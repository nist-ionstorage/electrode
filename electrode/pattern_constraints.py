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

from traits.api import (HasTraits, Array, Float, Int, List, Instance,
    Str, Trait)

from .utils import (select_tensor, expand_tensor, rotate_tensor,
        name_to_deriv)


class Constraint(HasTraits):
    def objective(self, system, variables):
        return
        yield

    def constraints(self, system, variables):
        return
        yield


class PatternValueConstraint(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    d = Int
    v = Array(dtype=np.float64)
    r = Array(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    def objective(self, system, variables):
        v = select_tensor(self.v[None, ...]) # TODO: no select
        c = system.individual_potential(self.x, self.d)[:, 0, :]
        c = select_tensor(rotate_tensor(expand_tensor(c), self.r))
        return zip(c.T, v[0])


class PatternRangeConstraint(Constraint):
    min = Float
    max = Float
    index = Trait(None, Int)

    def constraints(self, system, variables):
        if self.index is not None:
            variables = variables[self.index]
        if self.min is not None:
            yield variables >= self.min
        if self.max is not None:
            yield variables <= self.max


class PotentialObjective(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    derivative = Str # derivative name
    value = Float # value
    rotation = Array(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    def objective(self, system, variables):
        d, e = name_to_deriv(self.derivative)
        c = system.individual_potential(self.x, d)[:, 0, :]
        if not np.allclose(self.rotation, np.identity(3)):
            c = expand_tensor(c)
            c = rotate_tensor(c, self.rotation, d)
            c = select_tensor(c)
        c = c[:, e]
        return [(c, self.value)]


class MultiPotentialObjective(Constraint):
    components = List(Instance(PotentialObjective))
    # component values are weights
    value = Float

    def objective(self, system, variables):
        c = 0.
        for oi in self.components:
            for ci, vi in oi.objective(system, variables):
                c += vi*ci
        return [(c, self.value)]


class PotentialConstraint(Constraint):
    x = Array(dtype=np.float64, shape=(3,))
    derivative = Str
    min = Float # value
    max = Float # value
    rotation = Array(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    def constraints(self, system, variables):
        d, e = name_to_deriv(self.derivative)
        c = system.individual_potential(self.x, d)[:, 0, :]
        if not np.allclose(self.rotation, np.identity(3)):
            c = expand_tensor(c)
            c = rotate_tensor(c, self.rotation, d)
            c = select_tensor(c)
        c = c[:, e]
        if self.min is not None:
            yield c*variables >= self.min
        if self.max is not None:
            yield c*variables <= self.max

