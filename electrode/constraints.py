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

import numpy as np
import warnings

from traits.api import HasTraits, Array, Float, Int, List, Bool, Trait, Enum

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)

from .utils import rotate_tensor


class Constraint(HasTraits):
    weight = Float(1.) # if ==0, this becomes a constraint, else objective
    equal = Bool(True) # if True this constraint is == 0, else <= 0
    only_variable = Bool(False) # only take the variable parts into
                                # account, not the offset

    def values(self, system, electrodes, x, variables, pots):
        return
        yield

    def constraints(self, system, electrodes, x, variables, pots):
        if self.weight == 0:
            for v in self.values(system, electrodes, x, variables, pots):
                if self.equal:
                    yield v <= 0
                    yield v >= 0
                else:
                    yield v <= 0

    def objective(self, system, electrodes, x, variables, pots):
        if self.weight > 0:
            for v in self.values(system, electrodes, x, variables, pots):
                yield self.weight*cvxopt.modeling.sum(v)


class SingleIndexConstraint(Constraint):
    index = Trait(None, None, List(Int))

    def values(self, system, electrodes, x, variables, pots):
        if self.index is not None:
            idx = self.index
        else:
            idx = range(len(variables))
        for i in idx:
            for v in self.single_value(system, electrodes, x[i],
                    variables[i], pots[i]):
                yield v

    def single_value(self, system, electrodes, xi, variables, pots):
        return
        yield


class VoltageConstraint(Constraint):
    norm = Enum("none", "one", "inf") # norm to use across x
    delta = Int(1) # finite differences step size
    order = Int(0) # finite differences derivative order
    smooth = Bool(False) # flatten odd derivatives at the beginning and
                         # the end by mirror-extending them
    weight = 1.
    equal = False
    factor = Float(1.)
    range = Trait(None, None, Float) # offset (use with weight=0,
                                     # equal=False, norm="none")
    variable = Trait(None, None, Int) # apply only to the given
                                      # electrode index

    def values(self, system, electrodes, x, variables, pots):
        if self.variable is not None:
            variables = [v[self.variable] for v in variables]
        if self.order == -1:
            obj = variables
        elif self.order == 0:
            obj = (abs(v) for v in variables)
        elif self.order == 1:
            obj = (abs(va-vb) for va, vb in
                    zip(variables[self.delta:],
                        variables[:-self.delta]))
        elif self.order == 2:
            obj = (abs(va-2*vb+vc) for va, vb, vc in
                    zip(variables[:-2*self.delta],
                        variables[self.delta:-self.delta],
                        variables[2*self.delta:]))
            if self.smooth:
                obj = tuple(obj) + (
                        abs(2*variables[0]-2*variables[self.delta]),
                        abs(2*variables[-1]-2*variables[-self.delta-1]))
        elif self.order == 3:
            obj = (abs(va-3*vb+3*vc-vd) for va, vb, vc, vd in
                    zip(variables[3*self.delta:],
                        variables[2*self.delta:-self.delta],
                        variables[1*self.delta:-2*self.delta],
                        variables[:-3*self.delta]))
        elif self.order == 4:
            obj = (abs(va-4*vb+6*vc-4*vd+ve) for va, vb, vc, vd, ve in
                    zip(variables[4*self.delta:],
                        variables[3*self.delta:-self.delta],
                        variables[2*self.delta:-2*self.delta],
                        variables[self.delta:-3*self.delta],
                        variables[:-4*self.delta]))
            if self.smooth:
                obj = tuple(obj) + (
                        abs(6*variables[0]-8*variables[self.delta]
                            +2*variables[2*self.delta]),
                        abs(6*variables[-1]-8*variables[-self.delta-1]
                            +2*variables[-2*self.delta-1]))
        for v in obj:
            if self.norm == "inf":
                v = cvxopt.modeling.max(v)
            elif self.norm == "one":
                v = cvxopt.modeling.sum(v)
            if self.range is not None:
                v = v - self.range
            yield v*self.factor


class SymmetryConstraint(SingleIndexConstraint):
    symmetry = Array(dtype=np.int, shape=(None, 2)) # variable mapping
    weight = 0.
    equal = True

    def single_value(self, system, electrodes, x, variables, pots):
        for a, b in self.symmetry:
            va = a >= 0 and variables[int(a)] or 0
            vb = b >= 0 and variables[int(b)] or 0
            yield va - vb


class PotentialConstraint(SingleIndexConstraint):
    pmin = Trait(None, None, Float)
    pmax = Trait(None, None, Float)
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert p0.shape == (), p0.shape
        assert p.shape == (m,), p.shape
        if self.only_variable:
            p0 *= 0
        # float() works around cvxopt bug with 0-dim arrays
        v = float(p0) + cvxopt.modeling.dot(cvxopt.matrix(p.copy()), variables)
        if self.pmax is not None:
            yield v - self.pmax
        if self.pmin is not None:
            yield self.pmin - v


class ForceConstraint(SingleIndexConstraint):
    fmax = Trait(None, None, Array(shape=(3,), dtype=np.float64))
    coord = Trait(None, None, Array(shape=(3, 3), dtype=np.float64))
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert f0.shape == (3,), f0.shape
        assert f.shape == (3, m), f.shape
        if self.only_variable:
            f0 *= 0
        if self.coord is not None:
            f0 = rotate_tensor(f0, self.coord, order=1)
            f = rotate_tensor(f, self.coord, order=1)
        v = cvxopt.matrix(f0.copy()) + cvxopt.matrix(f.copy())*variables
        if self.fmax is not None:
            yield abs(v) - cvxopt.matrix(self.fmax)
        else:
            yield v


class CurvatureConstraint(SingleIndexConstraint):
    cmin = Trait(None, None, Array(shape=(6,), dtype=np.float64))
    cmax = Trait(None, None, Array(shape=(6,), dtype=np.float64))
    coord = Trait(None, None, Array(shape=(3, 3), dtype=np.float64))
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert c0.shape == (3, 3), c0.shape
        assert c.shape == (3, 3, m), c.shape
        if self.only_variable:
            c0 *= 0
        if self.coord is not None:
            c0 = rotate_tensor(c0, self.coord, order=2)
            c = rotate_tensor(c, self.coord)
        a, b = np.triu_indices(3)
        c0 = c0[a, b]
        c = c[a, b, :]
        lhs = cvxopt.matrix(c0.copy()) + cvxopt.matrix(c.copy())*variables
        if self.cmin is not None:
            yield cvxopt.matrix(self.cmin) - lhs
        if self.cmax is not None:
            yield lhs - cvxopt.matrix(self.cmax)


class OffsetPotentialConstraint(SingleIndexConstraint):
    pmin = Trait(None, None, Float)
    pmax = Trait(None, None, Float)
    weight = 0.
    equal = False
    reference = Array(shape=(3,), dtype=np.float64)

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert p0.shape == (), p0.shape
        assert p.shape == (m,), p.shape
        pref0 = system.potential(self.reference)[0]
        pref = []
        for el in electrodes:
            v0, el.voltage_dc = el.voltage_dc, 1.
            pref.append(el.electrical_potential(self.reference)[0])
            el.voltage_dc = v0
        pref = np.array(pref)
        if self.only_variable:
            p0 = p0*0
            pref0 *= 0
        # float() works around cvxopt bug with 0-dim arrays
        v = float(pref0 - p0) + cvxopt.modeling.dot(cvxopt.matrix(pref-p), variables)
        if self.pmin is not None:
            yield self.pmin - v
        if self.pmax is not None:
            yield v - self.pmax
