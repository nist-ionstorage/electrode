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

from traits.api import HasTraits, Array, Float, Int

from .electrode import Electrode


class GridElectrode(Electrode):
    origin = Array(dtype=np.float64, shape=(3, ), value=(0, 0, 0))
    spacing = Array(dtype=np.float64, shape=(3, ), value=(1, 1, 1))
    data = Array(dtype=np.float64, shape=(None, None, None))
    voltage = Float(1.)
    delta = Int(1)

    def orientations(self):
        """dummy"""
        return np.array([1])

    def plot(self, ax, text=None, *a, **kw):
        """dummy, TODO: use geometry"""
        pass

    def electrical_potential(self, x):
        """return linearly interpolated potential"""
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        #x, y, z = np.floor(x).astype(np.int).T
        #p = self.data[x, y, z]
        p = ndimage.map_coordinates(self.data, x.T,
                order=1, mode="nearest")
        return self.voltage*p

    def electrical_gradient(self, x):
        """finite differences gradient around x"""
        k = self.delta
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        x, y, z = np.floor(x).astype(np.int).T
        dx = self.data[x+k, y, z] - self.data[x, y, z]
        dy = self.data[x, y+k, z] - self.data[x, y, z]
        dz = self.data[x, y, z+k] - self.data[x, y, z]
        dp = np.array([dx, dy, dz])/self.spacing/k
        return self.voltage*dp.T

    def electrical_curvature(self, x):
        """finite differences curvature around x"""
        k = self.delta
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        x, y, z = np.round(x).astype(np.int).T
        c = self.data[x, y, z]
        cxx = self.data[x+k, y, z] + self.data[x-k, y, z] - 2*c
        cyy = self.data[x, y+k, z] + self.data[x, y-k, z] - 2*c
        czz = self.data[x, y, z+k] + self.data[x, y, z-k] - 2*c
        #assert np.allclose(cxx+.5*cyy, -czz-.5*cyy)
        cxy = self.data[x+k, y+k, z] + self.data[x-k, y-k, z] - (
            self.data[x-k, y+k, z] + self.data[x+k, y-k, z])
        cxz = self.data[x+k, y, z+k] + self.data[x-k, y, z-k] - (
            self.data[x-k, y, z+k] + self.data[x+k, y, z-k])
        cyz = self.data[x, y+k, z+k] + self.data[x, y-k, z-k] - (
            self.data[x, y-k, z+k] + self.data[x, y+k, z-k])
        cv = np.array([
            [cxx, cxy/4, cxz/4],
            [cxy/4, cyy, cyz/4],
            [cxz/4, cyz/4, czz]])/np.outer(self.spacing,
                    self.spacing)/k**2
        return self.voltage*cv.transpose(2, 0, 1)



