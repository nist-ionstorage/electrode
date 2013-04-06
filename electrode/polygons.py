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

from __future__ import absolute_import, print_function, unicode_literals

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
        for e in system.electrodes:
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
            s.electrodes.append(e)
            if type(p) is geometry.Polygon:
                p = [p]
            for pi in p:
                pi = geometry.polygon.orient(pi, 1)
                ext = np.array(pi.exterior.coords)[:-1, :2]
                e.paths.append(ext)
                for ii in pi.interiors:
                    int = np.array(ii.coords)[-2::-1, :2]
                    e.paths.append(int)
        return s

    def validate(self):
        """
        asserts geometric validity of all electrodes
        """
        for ni, pi in self:
            if not pi.is_valid:
                raise ValueError, (ni, pi)

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
        shrinks each electrode by adding a gapsize buffer around it
        gaps between previously touching electrodes will be 2*gapsize wide
        electrodes must not be overlapping
        """
        p = Polygons()
        for ni, pi in self:
            pb = pi.boundary.buffer(gapsize, 0)
            if pb.is_valid:
                pc = pi.difference(pb)
                if pc.is_valid:
                    pi = pc
            p.append((ni, pi))
        return p

    def simplify(self, buffer=0.):
        p = Polygons()
        for ni, pi in self:
            p.append((ni, pi.buffer(buffer, 0)))
        return p
        
    def assign_to_pad(self, pads):
        """given a list of polygons or multipolygons and a list
        of pad xy coordinates, yield tuples of
        (pad number, polygon index, polygon)"""
        polys = list(enumerate(self))
        for pad, (x, y) in enumerate(pads):
            p = geometry.Point(x, y)
            for i in range(len(polys)):
                j, (name, poly) = polys[i]
                if p.intersects(poly):
                    yield pad, j, poly
                    del polys[i]
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
    q = (edge/2-step/2)*np.ones_like(p)
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
        for ei in si.electrodes:
            if not hasattr(ei, "paths"):
                continue
            for pi, oi in zip(ei.paths, ei.orientations()):
                print(ei.name, pi, oi)
        print()
    pickle.dump(s1, open("rfjunction1.pickle", "wb"))
