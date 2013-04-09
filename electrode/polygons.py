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

import numpy as np

from shapely import geometry, ops

from .system import System
from .electrode import PolygonPixelElectrode


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

    @classmethod
    def from_boundaries_routes(cls, boundaries=[], routes=[], edge=40.,
            buffer=1e-12):
        """
        start with a edge by edge square,
        subtract boundaries and put them in polygon
        electrodes named by their layer/datatype then pattern
        the rest using the routes (taken to be near-zero-width) and
        add one polygon electrode for each resulting fragment
        """
        field = geometry.Polygon([[edge/2, edge/2], [-edge/2, edge/2],
                                  [-edge/2, -edge/2], [edge/2, -edge/2]])
        p = cls()
        for name, polys in boundaries:
            mp = map(geometry.Polygon, polys)
            mp = reduce(lambda a, b: a.union(b), mp)
            if not type(mp) is geometry.MultiPolygon:
                mp = geometry.MultiPolygon([mp])
            assert mp.is_valid, polys
            assert field.is_valid, field
            mp = mp.intersection(field)
            field = field.difference(mp)
            p.append((name, mp))
        gaps = map(geometry.LineString, routes)
        gaps = reduce(lambda a, b: a.union(b), gaps)
        #gaps = ops.cascaded_union(gaps).intersection(field) # segfaults       
        fields = field.difference(gaps.buffer(buffer, 1))
        if type(fields) is geometry.Polygon:
            fields = geometry.MultiPolygon([fields])
        for fragment in fields:
            # assume that buffer is much smaller than any relevant
            # distance and round coordinates to 10*buffer, then simplify
            fragment = np.array(fragment.exterior.coords)
            fragment = np.around(fragment, int(-np.log10(buffer)-1))
            fragment = geometry.Polygon(fragment).buffer(0, 1)
            p.append(("", geometry.MultiPolygon([fragment])))
        return p

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

    def add_gaps(self, gapsize):
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

    def simplify(self, buffer=0):
        return self.add_gaps(-2*buffer)
        
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
