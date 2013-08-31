# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011-2013 Robert Jordens <jordens@phys.ethz.ch>
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

from __future__ import print_function, division, absolute_import

from functools import partial

import numpy as np

from .system import System
from .electrode import PolygonPixelElectrode
from .utils import area_centroid

from bem.pytriangle import triangulate


def paths_to_mesh(paths):
    points = np.concatenate(paths, axis=0)
    segments = []
    segmentmarkers = []
    n = 0
    for i, path in enumerate(paths):
        si = n + np.arange(path.shape[0])
        segments.append((np.roll(si, 1, 0), si))
        n += path.shape[0]
        segmentmarkers.append((i + 2)*np.ones_like(si))
    segments = np.concatenate(segments, axis=1).T
    segmentmarkers = np.concatenate(segmentmarkers, axis=0)
    args = {"points": points.astype(np.double, order="C"),
            "segments": segments.astype(np.intc, order="C"),
            "segmentmarkers": segmentmarkers.astype(np.intc, order="C")}
    return args


def transformed_copy(transformations, path):
    p4 = np.c_[path, np.zeros(len(path)), np.ones(len(path))]
    return [np.dot(p4, t.T)[::sign, :2] for t, sign in transformations]


def transformer(transformations):
    return partial(transformed_copy, transformations)


def adapt_mesh(constraints, variable, fixed=[], threshold=.5,
        nmax=int(2**15.5), a=1, q=20, up=16., down=4., verbose=False,
        symmetry=lambda p: [p], **kwargs):
    opts = "Qq%fa%fn" % (q, a)
    args = paths_to_mesh(variable)
    s = v = c = None
    while True:
        args = triangulate(opts=opts, **args)
        n = args["triangles"].shape[0]
        if n > nmax:
            break
        if verbose:
            print("triangles:", n, ", ", end="")
        areas = np.empty(n, dtype=np.double)
        centroids = np.empty((n, 2), dtype=np.double)
        paths = args["points"][args["triangles"], :]
        s = System(fixed)
        for i, p in enumerate(paths):
            areas[i], centroids[i] = area_centroid(p)
            s.append(PolygonPixelElectrode(paths=symmetry(p), **kwargs))
        v, c = s.optimize(constraints, verbose=False) #verbose)
        if verbose: 
            print("objective:", c)
        potentials = v[len(fixed):]
        neighbors = args.pop("neighbors")
        edge_changes = potentials[neighbors, :] - potentials[:, None]
        refine = np.fabs(edge_changes).max(1) > threshold
        args["triangleareas"] = np.where(refine, areas/down, areas*up)
    return s, v, c

