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

from .utils import select_tensor, expand_tensor, rotate_tensor


class Constraint(HasTraits):
    def objective(self, electrode, variables):
        return
        yield

    def constraints(self, electrode, variables):
        return
        yield


class PatternValueConstraint(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    d = Int
    v = Array(dtype=np.float64)
    r = Array(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    def objective(self, electrode, variables):
        v = select_tensor(self.v[..., None]).ravel()
        c, = electrode.value(self.x, self.d)
        c = expand_tensor(c[..., 0])
        c = np.array(rotate_tensor(c, self.r))
        c = select_tensor(c).reshape((v.shape[0], -1))
        return zip(c, v)


class PatternRangeConstraint(Constraint):
    min = Float
    max = Float

    def constraints(self, electrode, variables):
        if self.min is not None:
            yield variables >= self.min
        if self.max is not None:
            yield variables <= self.max


class PotentialObjective(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    d = Int # derivative order
    e = Int # derivative component
    v = Float # value

    def objective(self, electrode, variables):
        c = electrode.value(self.x, self.d)[0][self.e, :, 0]
        return [(c, self.v)]


class PotentialConstraint(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    d = Int # derivative order
    e = Int # derivative component
    min = Float # value
    max = Float # value

    def constraints(self, electrode, variables):
        c = electrode.value(self.x, self.d)[0][self.e, :, 0]
        if self.min is not None:
            yield c*variables >= self.min
        if self.max is not None:
            yield c*variables <= self.max
