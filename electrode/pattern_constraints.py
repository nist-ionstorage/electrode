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

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import warnings
import numpy as np

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)

from .utils import (select_tensor, expand_tensor, rotate_tensor,
        name_to_deriv, deriv_to_reduced_idx)


"""Constraints and objectives to be used with `System.optimize()`

.. note::
    Needs cvxopt.
"""


class Constraint(object):
    def objective(self, system, variables):
        return
        yield

    def constraints(self, system, variables):
        return
        yield


class PatternRangeConstraint(Constraint):
    """Constrains the potential to lie within the given range

    Parameters
    ----------
    min : float or None
        Minimum potential value or unbounded below if None.
    max : float or None
        Maximum potential value or unbounded above if None.
    index : int or None
        Only affect the given electrode index or all if None.
    """
    def __init__(self, min=None, max=None, index=None):
        self.min = min
        self.max = max
        self.index = index

    def constraints(self, system, variables):
        if self.index is not None:
            variables = variables[self.index]
        if self.min is not None or self.max is not None:
            if self.min == self.max:
                yield variables == self.min
            else:
                if self.min is not None:
                    yield variables >= self.min
                if self.max is not None:
                    yield variables <= self.max


class SingleValueConstraint(Constraint):
    """Base class for Constraints/Objectives.

    Parameters
    ----------
    value : float or None
        If not None, the final value (the .get() of self) is optimized
        and kept proportional to `value`.
    min : float or None
        If not None, the value of this constraint is kept at or above
        `min.`
    max : float or None
        If not None, it is kept below or equal `max`.
    offset : float or None
        The value is forced exactly (not proportional) to `offset`.
    """
    def __init__(self, value=None, min=None, max=None, offset=None):
        self.value = value
        self.offset = offset
        self.min = min
        self.max = max

    def get(self, system, variables):
        raise NotImplementedError

    def objective(self, system, variables):
        if self.value is not None:
            c = self.get(system, variables)
            yield c, float(self.value)

    def constraints(self, system, variables):
        if (self.offset is not None
            or self.min is not None
            or self.max is not None):
            c = self.get(system, variables)
            d = cvxopt.matrix(np.ascontiguousarray(c))
            v = cvxopt.modeling.dot(d, variables)
            if self.offset is not None:
                yield v == float(self.offset)
            if self.min is not None:
                yield v >= float(self.min)
            if self.max is not None:
                yield v <= float(self.max)


class PotentialObjective(SingleValueConstraint):
    """Constrain or optimize potential.

    Parameters
    ----------
    x : array_like, shape (3,)
        Position where to evalue/constrain/optimize potential
    derivative : str
        Derivative to constrain/optimize. String of characters from
        "xyz". See `utils.name_to_deriv.keys()` for possible values.
        Not all possible cartesian derivatives are allowed, only those
        that are evaluated as the basis for the given order. Use
        `MultiPotentialObjective` to constrain sums or differences that
        make up the other derivatives.
    rotation : array_like, shape (3, 3)
        Rotation of the local coordinate system. np.eye(3) if None.
    **kwargs : any
        Passed to `SingleValueConstraint()`
    """
    def __init__(self, x, derivative, rotation=None, **kwargs):
        super(PotentialObjective, self).__init__(**kwargs)
        self.x = np.asanyarray(x, np.double)
        self.derivative = derivative
        self.order = len(derivative)
        self.reduced_idx = deriv_to_reduced_idx(derivative)
        self.rotation = (np.asanyarray(rotation, np.double)
                if rotation is not None else None)

    def get(self, system, variables):
        c = system.individual_potential(self.x, self.order)[:, 0, :]
        if self.rotation is not None:
            c = select_tensor(rotate_tensor(expand_tensor(c),
                self.rotation, self.order))
        c = c[:, self.reduced_idx]
        if type(self.reduced_idx) is tuple:
            c = -c.sum(1)
        return c


class MultiPotentialObjective(SingleValueConstraint):
    """Constrains or optimizes a linear combination of
    `PotentialObjective()` s.

    The value of this constraint (either used as a min/max or equal
    constraint or as part of the objective) is the sum of the
    constituents' `objective()` s. Thus the component `value` s are their
    weights.

    Parameters
    ----------
    components : list of `PotentialObjective()` s
    **kwargs : any
        Passed to `SingleValueConstraint()`.
    """
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


class VoltageDerivativeConstraint(Constraint):
    def __init__(self, order, weight=0, max=None, min=None,
                 smooth=False, delta=1, norm="one", abs=True):
        self.order = order
        self.weight = weight
        self.smooth = smooth
        self.delta = delta
        self.norm = norm
        self.abs = abs
        self.max = max
        self.min = min

    def get(self, system, variables):
        obj = variables
        for i in range(self.order):
            if self.smooth and i % 2 == 0:
                obj = obj[self.delta:0:-1] + obj + obj[-2:-2-self.delta:-1]
            obj = [(obj[i + self.delta] - obj[i]) for i in
                   range(len(obj) - self.delta)]
        return [v*(1./(self.delta**self.order)) for v in obj]

    def coef(self, system, variables):
        for v in self.get(system, variables):
            if self.abs:
                v = abs(v)
            if self.norm == "one":
                yield cvxopt.modeling.sum(v)
            elif self.norm == "inf":
                yield cvxopt.modeling.max(v)
            else:
                raise ValueError(self.norm)

    def objective(self, system, variables):
        if self.weight:
            for v in self.coef(system, variables):
                yield v, float(self.weight)

    def constraints(self, system, variables):
        if self.max is not None:
            for v in self.coef(system, variables):
                yield v <= float(self.max)
        if self.min is not None:
            for v in self.coef(system, variables):
                yield v >= float(self.min)


class SymmetryConstaint(Constraint):
    def __init__(self, a, b):
        raise NotImplementedError
