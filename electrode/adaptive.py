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

"""Tools to perform adaptive mesh refinement based on System.optimize
output.

.. note::
    Needs the python triangulate() wrapper.
"""


def paths_to_mesh(paths):
    """Converts a coordinates to `triangulate()` input data.

    """
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
    """Adaptively refines the electrode boundaries based on the
    incremental `System.optimize()` results.

    Parameters
    ----------
    constraints : list of `pattern_constraint.Constraint`
        Constraints and and objectives from `pattern_constraints` 
        to be passed to `System.optimize()`.
    variable : list of array_like
        List of xy arrays (n, 2) shaped boundaries of the electrodes to
        be subdivided and incrementally segmented.
    fixed : list of array_like
        List of xy arrays (n, 2) shaped electrode boundaries that are
        not to be changed in shaped but can be changed in voltage.
    threshold : float
        Potential jump threshold above which to detect an electrode
        boundary. Refinement happens at boundaries where electrodes of
        potentials that are differing by more than `threshold` are
        adjacent.
    nmax : int
        Number of triangles to stop the iterative refinemement at.
    a : float
        Initial may triangle area for first triangulation.
    q : float
        Minimum triangle corner angle during triangulation (constrained
        Delaunay, see triangulate() documentation).
    up : float
    down : float
        Factors to increase/decrease the triangle area by before each
        re-triangulation. Increase happens at triangles that are
        surrounded by triangles with similar potential. Decrease
        otherwise.
    verbose : bool
        Print triangulation information. Also passed to
        System.optimize()
    symmetry : callable
        Symmetry operation to transform every triangle with. Generates
        Segmentations that strictly adhere to this symmetry. Called as
        `symmetry(triangle)`. With triangle being a (3, 2) array of
        corners. See `transformed_copy()` and `transformer()` for some
        utilities how to write these symmetry functions.

    Returns
    -------
    s : System
        The `System()` instance with the polygonal electrodes.
    v : array_like
        The array of potentials for the electrodes in `s`.
    c : float
        Final strength of the constraints. See `System.optimize()`.
    """
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
        edge_changes = potentials.take(neighbors) - potentials[:, None]
        refine = np.fabs(edge_changes).max(1) > threshold
        args["triangleareas"] = np.where(refine, areas/down, areas*up)
    return s, v, c

