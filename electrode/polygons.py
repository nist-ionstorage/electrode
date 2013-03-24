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

import numpy as np

from shapely import geometry, ops

from .system import System
from .electrode import PolygonPixelElectrode


def polygons_to_system(polygons):
    """
    convert a list of [("electrode name", MultiPolygon(...)), ...] to a
    System()
    """
    s = System()
    for n, p in polygons:
        e = PolygonPixelElectrode(name=n, paths=[])
        s.electrodes.append(e)
        if type(p) is geometry.Polygon:
            p = [p]
        for pi in p:
            pi = geometry.polygon.orient(pi, 1)
            ext = np.array(pi.exterior.coords[:-1]).copy()
            ext.resize(ext.shape[0], 3)
            e.paths.append(ext)
            for ii in pi.interiors:
                int = np.array(ii.coords)[-1::-1].copy()
                int.resize(int.shape[0], 3)
                e.paths.append(int)
    return s

def system_to_polygons(system, pixel_threshold=np.sqrt(.5)):
    """
    convert a System() to a list of [("electrode name",
    MultiPolygon(...)), ...]
    """
    p = []
    for e in system.electrodes:
        if not hasattr(e, "paths"):
            continue
        # assert type(e) is PolygonPixelElectrode, (e, e.name)
        exts, ints = [], []
        for pi, ei, qi in zip(e.paths, e.orientations(), e.pixel_factors):
            if qi < pixel_threshold:
                continue
            # shapely ignores f-contiguous arrays
            # https://github.com/sgillies/shapely/issues/26
            {-1: ints, 1: exts}[ei].append(pi.copy("C"))
        mp = []
        for ei in exts:
            # FIXME: only those in ints that intersect ei
            mp.append((ei, ints))
        p.append((e.name, geometry.MultiPolygon(mp)))
    return p

def check_validity(polygons):
    """
    asserts geometric validity of all electrodes
    """
    for ni, pi in polygons:
        if not pi.is_valid:
            raise ValueError, (ni, pi)

def remove_overlaps(polygons):
    """
    successively removes overlaps with preceeding electrodes
    """
    p = []
    acc = geometry.Point()
    for ni, pi in polygons:
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

def add_gaps(polygons, gapsize):
    """
    shrinks each electrode by adding a gapsize buffer around it
    gaps between previously touching electrodes will be 2*gapsize wide
    electrodes must not be overlapping
    """
    p = []
    for ni, pi in polygons:
        pb = pi.boundary.buffer(gapsize, 0)
        if pb.is_valid:
            pc = pi.difference(pb)
            if pc.is_valid:
                pi = pc
        p.append((ni, pi))
    return p

def simplify_polygons(polygons, theshold=.5**.5):
    out = []
    for name, poly in polygons:
        poly = poly.buffer(0)
        out.append((name, poly))
    return out


if __name__ == "__main__":
    import cPickle as pickle
    s = pickle.load(open("rfjunction.pickle", "rb"))
    p = system_to_polygons(s)
    p = remove_overlaps(p)
    p = add_gaps(p, .05)
    s1 = polygons_to_system(p)
    for si in s, s1:
        for ei in si.electrodes:
            if not hasattr(ei, "paths"):
                continue
            for pi, oi in zip(ei.paths, ei.orientations()):
                print ei.name, pi, oi
        print
    pickle.dump(s1, open("rfjunction1.pickle", "wb"))
