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

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)

from .utils import (select_tensor, expand_tensor, rotate_tensor,
        name_to_deriv)


class Constraint(object):
    def objective(self, system, variables):
        return
        yield

    def constraints(self, system, variables):
        return
        yield


class PatternValueConstraint(Constraint):
    def __init__(self, x, d, v, r=None):
        warnings.warn("use PotentialObjective and MultiPotentialObjective",
                DeprecationWarning)
        self.x = np.asanyarray(x, np.double)
        self.d = d
        self.v = np.asanyarray(v, np.double)
        self.r = np.asanyarray(r, np.double) if r is not None else None

    def objective(self, system, variables):
        v = select_tensor(self.v[None, ...]) # TODO: no select
        c = system.individual_potential(self.x, self.d)[:, 0, :]
        if self.r is not None:
            c = select_tensor(rotate_tensor(expand_tensor(c), self.r,
                self.d))
        return zip(c.T, v[0])


class PatternRangeConstraint(Constraint):
    def __init__(self, min=None, max=None, index=None):
        self.min = min
        self.max = max
        self.index = index

    def constraints(self, system, variables):
        if self.index is not None:
            variables = variables[self.index]
        if self.min is not None:
            yield variables >= self.min
        if self.max is not None:
            yield variables <= self.max


class SingleValueConstraint(Constraint):
    def __init__(self, value=None, min=None, max=None):
        self.value = value
        self.min = min
        self.max = max

    def get(self, system, variables):
        raise NotImplementedError

    def objective(self, system, variables):
        if self.value is not None:
            c = self.get(system, variables)
            yield (c, self.value)
    
    def constraints(self, system, variables):
        if self.min is not None or self.max is not None:
            c = self.get(system, variables)
            d = cvxopt.matrix(np.ascontiguousarray(c))
            v = cvxopt.modeling.dot(d, variables)
            if self.min is not None:
                yield v >= self.min
            if self.max is not None:
                yield v <= self.max


class PotentialObjective(SingleValueConstraint):
    def __init__(self, x, derivative, rotation=None, **kwargs):
        super(PotentialObjective, self).__init__(**kwargs)
        self.x = np.asanyarray(x, np.double)
        self.derivative = derivative
        self.rotation = (np.asanyarray(rotation, np.double)
                if rotation is not None else None)

    def get(self, system, variables):
        d, e = name_to_deriv(self.derivative)
        c = system.individual_potential(self.x, d)[:, 0, :]
        if self.rotation is not None:
            c = select_tensor(rotate_tensor(expand_tensor(c),
                self.rotation, d))
        return c[:, e]
    

class MultiPotentialObjective(SingleValueConstraint):
    def __init__(self, components=[], **kwargs):
        super(MultiPotentialObjective, self).__init__(**kwargs)
        self.components = components
        # component values are weights

    def get(self, system, variables):
        c = 0.
        for oi in self.components:
            for ci, vi in oi.objective(system, variables):
                c = c+vi*ci
        return c
