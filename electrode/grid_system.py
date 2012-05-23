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


import itertools, sys, warnings

import numpy as np
from scipy import constants as ct

from traits.api import HasTraits, List, Instance

from .system import System
from .grid_electrode import GridElectrode


class GridSystem(System):
    rf = List(Instance(GridElectrode))
    dc = List(Instance(GridElectrode))

    def potential(self, x):
        p = sum(e.electrical_potential(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return p

    def gradient(self, x):
        f = sum(e.electrical_gradient(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return f

    def curvature(self, x):
        c = sum(e.electrical_curvature(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return c

    @classmethod
    def from_vtk(cls, geom, data, scale=1e-6):
        """load grid potential data from vti file "data" and geometry
        from "geom" and return a GridSystem instance, scale length
        units to scale"""
        from enthought.tvtk.api import tvtk
        o = cls()
        sgr = tvtk.XMLImageDataReader(file_name=data)
        sgr.update()
        sg = sgr.output
        for i in range(sg.point_data.number_of_arrays):
            name = sg.point_data.get_array_name(i)
            sp = sg.point_data.get_array(i)
            data = sp.to_array()
            spacing = sg.spacing
            origin = sg.origin
            dimensions = sg.dimensions
            # print name, spacing, origin, dimensions
            if sp.number_of_components == 1:
                data = data.reshape(dimensions[::-1]).transpose(2, 1, 0)
            else:
                continue # ignore fields for now
                data = data.reshape(tuple(dimensions) +
                    (sp.number_of_components, ))
            if "_pondpot_1V1MHz1amu" in name:
                # convert to DC electrode equivalent potential
                data /= ct.elementary_charge**2/(4*ct.atomic_mass*(1e6*2*np.pi)**2
                        )/scale**2/ct.elementary_charge
                name = name[:-len("_pondpot_1V1MHz1amu")]
            else:
                data /= 1.
            el = GridElectrode(name=name, origin=origin/scale,
                    spacing=spacing/scale, data=data)
            o.electrodes.append(el)
        return o


