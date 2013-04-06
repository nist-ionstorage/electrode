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

import warnings

import numpy as np
import matplotlib as mpl
from scipy.ndimage.interpolation import map_coordinates

from .utils import area_centroid, derive_map

try:
    if False: # test slow python only expressions
        raise ImportError
    from .cexpressions import (point_potential, polygon_potential,
            mesh_potential)
except ImportError:
    from .expressions import (point_potential, polygon_potential,
            mesh_potential)


class Electrode(object):
    def __init__(self, name="", dc=0., rf=0.):
        self.name = name
        self.dc = dc
        self.rf = rf

    def potential(self, x, derivative=0, potential=1., out=None):
        """return the specified derivative of the eletrical potential,
        units are volts"""
        raise NotImplementedError

    def orientations(self):
        """return the orientation of pathes (positive orientation
        yields positive potential for positive voltage and z>0"""
        return np.array([])

    def plot(self, ax, label=None, color=None, **kw):
        """plot this electrode's in the supplied axes"""
        pass


class CoverElectrode(Electrode):
    height = 50.
    # also adjust cover_height in
    # the other electrodes to include the cover's effect on their
    # potentials

    def __init__(self, height=50., **kwargs):
        super(CoverElectrode, self).__init__(**kwargs)
        self.height = height

    def potential(self, x, derivative=0, potential=1., out=None):
        if out is None:
            out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        if derivative == 0:
            out[:, 0] += potential*x[:, 2]/self.height
        elif derivative == 1:
            out[:, 2] += potential*1/self.height
        else:
            pass
        return out


class SurfaceElectrode(Electrode):
    """
    Gapless surface electrode patch set.

    Parts of the PixelElectrode code are based on:

    Roman Schmied, SurfacePattern software package
    http://atom.physik.unibas.ch/people/romanschmied/code/SurfacePattern.php

    [1] R. Schmied, "Electrostatics of gapped and finite surface
    electrodes", New J. Phys. 12:023038 (2010),
    http://dx.doi.org/10.1088/1367-2630/12/2/023038

    [2] R. Schmied, J. H. Wesenberg, and D. Leibfried, "Optimal
    Surface-Electrode Trap Lattices for Quantum Simulation with Trapped
    Ions", PRL 102:233002 (2009),
    http://dx.doi.org/10.1103/PhysRevLett.102.233002
    """

    def __init__(self, cover_height=50., cover_nmax=0, **kwargs):
        super(SurfaceElectrode, self).__init__(**kwargs)
        # cover plane height
        self.cover_height = cover_height
        # max components in cover plane potential expansion
        self.cover_nmax = cover_nmax


class PointPixelElectrode(SurfaceElectrode):
    def __init__(self, points=[], areas=[], **kwargs):
        super(PointPixelElectrode, self).__init__(**kwargs)
        self.points = np.asanyarray(points, np.double)
        self.areas = np.asanyarray(areas, np.double)

    def orientations(self):
        return np.ones_like(self.areas)

    def plot(self, ax, label=None, color=None, **kw):
        # color="red"?
        p = self.points
        a = (self.areas/np.pi)**.5*2
        col = mpl.collections.EllipseCollection(
                edgecolors="none",
                #cmap=plt.cm.binary, norm=plt.Normalize(0, 1.),
                facecolor=color,
                widths=a, heights=a, units="x", # FIXME xy in matplotlib>r8111
                angles=np.zeros(a.shape),
                offsets=p[:, (0, 1)], transOffset=ax.transData)
        ax.add_collection(col)
        if label is None:
            label = self.name
        if label:
            ax.text(p[:,0].mean(), p[:,1].mean(), label)

    def potential(self, x, derivative=0, potential=1., out=None):
        return point_potential(x, self.points, self.areas, potential,
                derivative, self.cover_nmax, self.cover_height, out)


class PolygonPixelElectrode(SurfaceElectrode):
    def __init__(self, paths=[], **kwargs):
        super(PolygonPixelElectrode, self).__init__(**kwargs)
        self.paths = [np.asanyarray(i, np.double) for i in paths]

    def orientations(self):
        return np.sign([area_centroid(pi)[0] for pi in self.paths])

    def plot(self, ax, label=None, color=None, **kw):
        if label is None:
            label = self.name
        for p in self.paths:
            ax.fill(p[:,0], p[:,1],
                    edgecolor=kw.get("edgecolor", "none"),
                    color=color, **kw)
            if label:
                ax.text(p[:,0].mean(), p[:,1].mean(), label)

    def to_points(self):
        a, c = [], []
        for p in self.paths:
            ai, ci = area_centroid(p)
            a.append(ai)
            c.append(ci)
        e = PointPixelElectrode(name=self.name, dc=self.dc, rf=self.rf,
                cover_nmax=self.cover_nmax, cover_height=self.cover_height,
                areas=a, points=c)
        return e

    def potential(self, x, derivative=0, potential=1., out=None):
        return polygon_potential(x, self.paths, potential, derivative,
                self.cover_nmax, self.cover_height, out)


class MeshPixelElectrode(SurfaceElectrode):
    def __init__(self, points=[], edges=[], polygons=[], potentials=[],
            **kwargs):
        super(MeshPixelElectrode, self).__init__(**kwargs)
        self.points = np.asanyarray(points, np.double)
        self.edges = np.asanyarray(edges, np.intc)
        self.polygons = np.asanyarray(polygons, np.intc)
        self.potentials = np.asanyarray(potentials, np.double)

    @classmethod
    def from_polygon_system(cls, s):
        points = []
        edges = []
        polygons = []
        potentials = []
        for p in s.electrodes:
            assert isinstance(p, PolygonPixelElectrode), p
            for i in p.paths:
                ei = len(points)+np.arange(len(i))
                points.extend(i)
                edges.extend(np.c_[np.roll(ei, 1, 0), ei])
                polygons.extend(len(potentials)*np.ones(len(ei)))
            potentials.append(p.dc)
        return cls(dc=1, points=points, edges=edges, polygons=polygons,
                potentials=potentials)

    def potential(self, x, derivative=0, potential=1., out=None):
        return mesh_potential(x, self.points, self.edges, self.polygons,
                self.potentials*potential, derivative,
                self.cover_nmax, self.cover_height, out)


class GridElectrode(Electrode):
    def __init__(self, data=[], origin=(0, 0, 0), spacing=(1, 1, 1),
            **kwargs):
        super(GridElectrode, self).__init__(**kwargs)
        self.data = [np.asanyarray(i, np.double) for i in data]
        self.origin = np.asanyarray(origin, np.double)
        self.spacing = np.asanyarray(spacing, np.double)

    @classmethod
    def from_result(cls, result, maxderiv=3):
        obj = cls()
        obj.origin = result.grid.get_origin()
        obj.spacing = result.grid.step
        obj.data.append(result.potential[:, :, :, None])
        if result.field is not None:
            obj.data.append(result.field.transpose(1, 2, 3, 0))
        obj.generate(maxderiv)
        return obj

    @classmethod
    def from_vtk(cls, fil):
        """load grid potential data from vtk StructuredPoints file "fil"
        and return a GridElectrode instance"""
        from tvtk.api import tvtk
        #sgr = tvtk.XMLImageDataReader(file_name=fil)
        sgr = tvtk.StructuredPointsReader(file_name=fil)
        sgr.update()
        sg = sgr.output
        pot = [None, None]
        for i in range(sg.point_data.number_of_arrays):
            name = sg.point_data.get_array_name(i)
            if "_pondpot" in name:
                continue # not harmonic, do not use it
            elif name not in ("potential", "field"):
                continue
            sp = sg.point_data.get_array(i)
            data = sp.to_array()
            spacing = sg.spacing
            origin = sg.origin
            dimensions = tuple(sg.dimensions)
            dim = sp.number_of_components
            data = data.reshape(dimensions[::-1]+(dim,)).transpose(2, 1, 0, 3)
            pot[(dim-1)/2] = data
        obj = cls(origin=origin, spacing=spacing, data=pot)
        return obj

    def generate(self, maxderiv=3):
        for deriv in range(maxderiv+1):
            if len(self.data) < deriv+1:
                self.data.append(self.derive(deriv))
            ddata = self.data[deriv]
            assert ddata.ndim == 4, ddata.ndim
            assert ddata.shape[-1] == 2*deriv+1, ddata.shape
            if deriv > 0:
                assert ddata.shape[:-1] == self.data[deriv-1].shape[:-1]

    def derive(self, deriv):
        odata = self.data[deriv-1]
        ddata = np.empty(odata.shape[:-1] + (2*deriv+1,), np.double)
        for i in range(2*deriv+1):
            (e, j), k = derive_map[(deriv, i)]
            # TODO triple work
            grad = np.gradient(odata[..., j], *self.spacing)[k]
            ddata[..., i] = grad
        return ddata

    def potential(self, x, derivative=0, potential=1., out=None):
        x = (x - self.origin[None, :])/self.spacing[None, :]
        if out is None:
            out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        dat = self.data[derivative]
        for i in range(2*derivative+1):
            map_coordinates(dat[..., i], x.T, order=1, mode="nearest",
                    output=out[:, i])
        return out
