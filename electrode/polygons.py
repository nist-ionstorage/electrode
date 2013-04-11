# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import logging

import numpy as np
from shapely import geometry, ops
from gdsii import library, structure, elements

from .system import System
from .electrode import PolygonPixelElectrode

logger = logging.getLogger()


class Polygons(list):
    @classmethod
    def from_system(cls, system):
        """
        convert a System() to a list of [("electrode name",
        MultiPolygon(...)), ...]
        """
        obj = cls()
        for e in system:
            if not hasattr(e, "paths"):
                continue
            # assert type(e) is PolygonPixelElectrode, (e, e.name)
            exts, ints = [], []
            for pi, ei in zip(e.paths, e.orientations()):
                # shapely ignores f-contiguous arrays so copy
                # https://github.com/sgillies/shapely/issues/26
                pi = geometry.Polygon(pi.copy("C"))
                {-1: ints, 0: [], 1: exts}[ei].append(pi)
            if not exts:
                continue
            mp = geometry.MultiPolygon(exts)
            if ints:
                mp = mp.difference(geometry.MultiPolygon(ints))
            obj.append((e.name, mp))
        return obj

    def to_system(self):
        s = System()
        for n, p in self:
            e = PolygonPixelElectrode(name=n, paths=[])
            s.append(e)
            if type(p) is geometry.Polygon:
                p = [p]
            for pi in p:
                ext = np.array(pi.exterior.coords)
                if not pi.exterior.is_ccw:
                    ext = ext[::-1]
                e.paths.append(ext[:-1, :2])
                for ii in pi.interiors:
                    int = np.array(ii.coords)
                    if ii.is_ccw:
                        int = int[::-1]
                    e.paths.append(int[:-1, :2])
        return s

    # attribute namespaces anyone?
    attr_base = sum(ord(i) for i in "electrode") # 951
    attr_name = attr_base + 0

    @classmethod
    def from_gds(cls, fil, scale=1., name=None, poly_layers=None,
            gap_layers=None, route_layers=[], bridge_layers=[], **kwargs):
        lib = library.Library.load(fil)
        polys = []
        gaps = []
        routes = []
        bridges = []
        for stru in lib:
            assert type(stru) is structure.Structure
            if name is None or name == stru.name:
                break
        for e in stru:
            ij = e.layer, e.data_type
            path = np.array(e.xy)*lib.physical_unit/scale
            props = dict(e.properties)
            name = props.get(cls.attr_name, "")
            if type(e) is elements.Boundary:
                if poly_layers is None or ij in poly_layers:
                    polys.append((name, path))
            elif type(e) is elements.Path:
                if gap_layers is None or ij in gap_layers:
                    gaps.append(path)
                elif ij in route_layers:
                    routes.append(path)
                elif ij in bridge_layers:
                    bridges.append(path)
                else:
                    logger.debug("%s skipped", e)
            else:
                logger.debug("%s skipped", e)
        return cls.from_data(polys, gaps, routes, bridges, **kwargs)

    @classmethod
    def from_data(cls, polys=[], gaps=[], routes=[],
            bridges=[], edge=40., buffer=1e-10):
        """
        start with a edge by edge square, and cut it according to
        gaps. then undo the fragmentation that is fully encircled by
        routes and bridges. then undo fragmentation within polys and
        then fragment along the poly boundaries. finally fragment
        along routes.

        the result is a Polygons() that contains the fragmented area of
        the original square, with polys as some of the fragments and the
        rest fragmented along routes and those gaps that are
        not encircled in routes and bridges.

        buffer is a small size that is used to associate a finite width
        to gaps and routes. there should be no features smaller than
        10*buffer.
        """
        fragments = cls()
        field = geometry.Polygon([[edge/2, edge/2], [-edge/2, edge/2],
                                  [-edge/2, -edge/2], [edge/2, -edge/2]])
        gaps = geometry.MultiLineString(gaps)
        field = field.difference(gaps.buffer(buffer, 1))
        if routes:
            routes = geometry.MultiLineString(routes)
        if routes and bridges:
            bridges = geometry.MultiLineString(bridges)
            for poly in ops.polygonize(routes.union(bridges)):
                field = field.union(poly)
        # field = field.buffer(0, 1)
        for name, poly in polys:
            poly = geometry.Polygon(poly)
            assert poly.is_valid, poly
            poly = geometry.polygon.orient(poly, 1)
            poly = poly.intersection(field)
            fragments.append((name, poly))
            field = field.difference(poly)
        if routes:
            field = field.difference(routes.buffer(buffer, 1))
        assert field.is_valid, field
        if type(field) is geometry.Polygon:
            field = [field]
        for i in field:
            # assume that buffer is much smaller than any relevant
            # distance and round coordinates to 10*buffer, then simplify
            i = np.array(i.exterior.coords)
            i = np.around(i, int(-np.log10(buffer)-1))
            i = geometry.Polygon(i)
            fragments.append(("", i))
        return fragments

    def to_gds(self, scale=1., poly_layer=(0, 0), gap_layer=(1, 0),
            text_layer=(0, 0), phys_unit=1e-9, name="trap_electrodes"):
        lib = library.Library(version=5, name=bytes(name),
                physical_unit=phys_unit, logical_unit=1e-3)
        stru = structure.Structure(name=bytes(name))
        lib.append(stru)
        #stru.append(elements.Node(layer=layer, node_type=0, xy=[(0, 0)]))
        for name, polys in self:
            props = {self.attr_name: bytes(name)}
            if type(polys) is geometry.Polygon:
                polys = [polys]
            for p in polys:
                assert p.is_valid, p
                if p.is_empty:
                    continue
                xy = np.array(p.exterior.coords.xy).copy()
                xy = xy.T[:, :2]*scale/phys_unit
                xyb = np.r_[xy, xy[:1]]
                if poly_layer is not None:
                    p = elements.Boundary(layer=poly_layer[0],
                            data_type=poly_layer[1], xy=xy)
                    p.properties = props.items()
                    stru.append(p)
                if gap_layer is not None:
                    p = elements.Path(layer=gap_layer[0],
                            data_type=gap_layer[1], xy=xyb)
                    p.properties = props.items()
                    stru.append(p)
                if text_layer is not None:
                    p = elements.Text(layer=text_layer[0],
                            text_type=text_layer[1], xy=xy[:1],
                            string=bytes(name))
                    p.properties = props.items()
                    stru.append(p)
        return lib

    def validate(self):
        """
        asserts geometric validity of all electrodes
        """
        for ni, pi in self:
            if not pi.is_valid:
                raise ValueError("%s %s" % (ni, pi))

    def remove_overlaps(self):
        """
        successively removes overlaps with preceeding electrodes
        """
        p = Polygons()
        acc = geometry.Point()
        for ni, pi in self:
            pa = acc.intersection(pi)
            if pa.is_valid and pa.area > np.finfo(np.float32).eps:
                pc = pi.difference(pa)
                if pc.is_valid:
                    pi = pc
            acca = acc.union(pi)
            if acca.is_valid:
                acc = acca
            p.append((ni, pi))
        return p

    def add_gaps(self, gapsize=0):
        """
        shrinks each electrode by adding a gapsize buffer around it.
        gaps between previously touching electrodes will be gapsize wide
        electrodes must not be overlapping
        """
        p = Polygons()
        for ni, pi in self:
            pb = pi.buffer(-gapsize/2., 1)
            if pb.is_valid:
                pi = pb
            p.append((ni, pi))
        return p

    def simplify(self, buffer=0, preserve_topology=False):
        if buffer == 0:
            return self.add_gaps(buffer)
        p = Polygons()
        for ni, pi in self:
            p.append((ni, pi.simplify(buffer,
                preserve_topology=preserve_topology)))
        return p
        
    def assign_to_pad(self, pads):
        """given a list of polygons or multipolygons and a list
        of pad xy coordinates, yield tuples of
        (pad number, polygon index, polygon)"""
        polys = range(len(self))
        for pad, (x, y) in enumerate(pads):
            p = geometry.Point(x, y)
            for i in polys:
                name, poly = self[i]
                if p.intersects(poly):
                    yield pad, i
                    polys.remove(i)
                    break
            if not polys:
                break
        # assert not polys, polys

    def gaps_union(self):
        """returns the union of the boundaries of the polygons.
        if the boundaries of adjacent polygons coincide, this returns
        only the gap paths.

        polys is a list of multipolygons or polygons"""
        gaps = []
        for name, multipoly in self:
            if type(multipoly) is geometry.Polygon:
                multipoly = [multipoly]
            for poly in multipoly:
                gaps.append(poly.exterior)
                gaps.extend(poly.interiors)
        #return ops.cascaded_union(gaps) # segfaults
        g = geometry.LineString()
        for i in gaps:
            g = g.union(i)
        return g


def square_pads(step=10., edge=200., odd=False, start_corner=0):
    """generates a (n, 2) array of xy coordinates of pad centers
    pad are spaced by `step`, on the edges with edge length `edge`.
    if odd=True, there is a pad the center of an edge. The corner to
    start is given in `start_corner`. 0 is top left (-x, +y). counter
    clockwise from that"""
    n = int(edge/step)
    if odd: n += (n % 2) + 1
    p = np.arange(-n/2.+.5, n/2.+.5)*step
    assert len(p) == n, (p, n)
    # top left as origin is common for packages
    q = (edge/2.-step/2.)*np.ones_like(p)
    edges = [(-q, -p), (p, -q), (q, p), (-p, q)]
    xy = np.concatenate(edges[start_corner:] + edges[:start_corner], axis=1)
    assert xy.shape == (2, 4*n), xy.shape
    return xy.T


if __name__ == "__main__":
    import cPickle as pickle
    s = pickle.load(open("rfjunction.pickle", "rb"))
    p = Polygons.from_system(s)
    p = p.remove_overlaps()
    p = p.add_gaps(.05)
    s1 = p.to_system()
    for si in s, s1:
        for ei in si:
            if not hasattr(ei, "paths"):
                continue
            for pi, oi in zip(ei.paths, ei.orientations()):
                print(ei.name, pi, oi)
        print()
    pickle.dump(s1, open("rfjunction1.pickle", "wb"))

    import sys
    from matplotlib import pyplot as plt
    with open(sys.argv[1], "rb") as fil:
        s = from_gds(fil)
    fig, ax = plt.subplots()
    s.plot(ax)
    fig.savefig("gds_to_system.pdf")
    l = to_gds(s)
    with open("system_to_gds.gds", "wb") as fil:
        l.save(fil)
