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

import warnings, itertools
from contextlib import contextmanager
import logging

import numpy as np
from scipy import optimize, constants as ct
import matplotlib.pyplot as plt

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)

try:
    from qc.theory.gni import gni
except ImportError:
    warnings.warn("qc modules not found, some stuff will fail", ImportWarning)

from .transformations import euler_from_matrix
from .saddle import rfo
from .electrode import PolygonPixelElectrode
from .utils import (expand_tensor, norm, rotate_tensor,
    mathieu, name_to_deriv)
from .pattern_constraints import (PatternRangeConstraint,
        PotentialObjective)
from . import colors


logger = logging.getLogger()


class System(list):
    def __init__(self, electrodes=[], **kwargs):
        super(System, self).__init__(**kwargs)
        self.extend(electrodes)
   
    @property
    def names(self):
        return [el.name for el in self]
   
    @names.setter
    def names(self, names):
        for ei, ni in zip(self, names):
            ei.name = ni
    
    @property
    def dcs(self):
        return np.array([el.dc for el in self])

    @dcs.setter
    def dcs(self, voltages):
        for ei, vi in zip(self, voltages):
            ei.dc = vi

    @property
    def rfs(self):
        return np.array([el.rf for el in self])

    @rfs.setter
    def rfs(self, voltages):
        for ei, vi in zip(self, voltages):
            ei.rf = vi

    def __getitem__(self, name_or_index):
        """return the first electrode named name or None if not found"""
        try:
            return list.__getitem__(self, name_or_index)
        except TypeError:
            for ei in self:
                if ei.name == name_or_index:
                    return ei

    electrode = __getitem__

    @contextmanager
    def with_voltages(self, dcs=None, rfs=None):
        """contextmanager with temporary voltage setting"""
        if dcs is not None:
            odc = self.dcs
            self.dcs = dcs
        if rfs is not None:
            orf = self.rfs
            self.rfs = rfs
        yield
        if dcs is not None:
            self.dcs = odc
        if rfs is not None:
            self.rfs = orf

    def electrical_potential(self, x, typ="dc", derivative=0, expand=False):
        """
        return electrical potential derivative due to the electrodes at x
        electrode voltage given in typ names attribute, expand the
        tensor to full form if expand==True
        """
        x = np.asanyarray(x, dtype=np.double).reshape(-1, 3)
        pot = np.zeros((x.shape[0], 2*derivative+1), np.double)
        for ei in self:
            vi = getattr(ei, typ, None)
            if vi:
                ei.potential(x, derivative, potential=vi, out=pot)
        if expand:
            pot = expand_tensor(pot)
        return pot
    
    def individual_potential(self, x, derivative=0):
        """
        return derivatives of the electrical potential at x and
        contribution of each of the specified electrodes per volt
        O(len(electrodes)*len(x))
        """
        x = np.asanyarray(x, dtype=np.double).reshape(-1, 3)
        eff = np.zeros((len(self), x.shape[0], 2*derivative+1),
                np.double)
        for i, ei in enumerate(self):
            ei.potential(x, derivative, potential=1., out=eff[i])
        return eff

    def time_potential(self, x, derivative=0, t=0., alpha_rf=1., expand=False):
        """electrical field at an instant in time t (physical units:
        1/omega_rf), no adiabatic approximation here"""
        dc, rf = (self.electrical_potential(x, typ, derivative, expand)
                for typ in ("dc", "rf"))
        return dc + alpha_rf*np.cos(t)*rf

    def pseudo_potential(self, x, derivative=0):
        """return given derivative or the ponderomotive/pseudo potential
        at x"""
        p = [self.electrical_potential(x, "rf", i, expand=True)
                for i in range(1, derivative+2)]
        if derivative == 0:
            return np.einsum("ij,ij->i", p[0], p[0])
        elif derivative == 1:
            return 2*np.einsum("ij,ijk->ik", p[0], p[1])
        elif derivative == 2:
            return 2*(np.einsum("ijk,ijl->ikl", p[1], p[1])
                     +np.einsum("ij,ijkl->ikl", p[0], p[2]))
        elif derivative == 3:
            a = np.einsum("ij,ijklm->iklm", p[0], p[3])
            b = np.einsum("ijk,ijlm->iklm", p[1], p[2])
            a += b
            a += b.transpose(0, 2, 1, 3)
            a += b.transpose(0, 3, 2, 1)
            return 2*a
        elif derivative == 4:
            a = np.einsum("ij,ijklmn->iklmn", p[0], p[4])
            b = np.einsum("ijk,ijlmn->iklmn", p[1], p[3])
            a += b
            a += b.transpose(0, 4, 2, 3, 1)
            a += b.transpose(0, 3, 2, 1, 4)
            a += b.transpose(0, 2, 1, 3, 4)
            c = np.einsum("ijkl,ijmn->iklmn", p[2], p[2])
            a += c
            a += c.transpose(0, 1, 4, 3, 2)
            a += c.transpose(0, 1, 3, 2, 4)
            return 2*a

    def potential(self, x, derivative=0):
        """combined electrical and ponderomotive potential"""
        dc = self.electrical_potential(x, "dc", derivative,
                expand=True)
        rf = self.pseudo_potential(x, derivative)
        return dc + rf

    def plot(self, ax, alpha=.3, **kwargs):
        """plot electrodes with sequential colors"""
        for e, c in zip(self, itertools.cycle(colors.set3)):
            e.plot(ax, color=tuple(c/255.), alpha=alpha, **kwargs)

    def plot_voltages(self, ax, u=None, um=None, cmap=plt.cm.RdBu_r,
            **kwargs):
        """plot electrodes with color proportional to voltage 
        red for positive, blue for negative"""
        if u is None:
            u = np.array(self.dcs)
        if um is None:
            um = np.fabs(u).max() or 1.
        u = (u / um + 1)/2
        #colors = np.clip((u+.5, .5-np.fabs(u), -u+.5), 0, 1).T
        colors = [cmap(ui) for ui in u]
        for el, ci in zip(self, colors):
            el.plot(ax, color=ci, **kwargs)

    def minimum(self, x0, axis=(0, 1, 2), coord=np.identity(3)):
        """find a potential minimum near x0 searching along the
        specified axes in the orthonormal matrix coord"""
        x = np.array(x0)
        def p(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return self.potential(np.dot(coord, x), 0)[0]
        def g(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return rotate_tensor(self.potential(np.dot(coord, x), 1),
                    coord.T)[0, axis]
        # downhill simplex
        # bfgs seems better in test cases
        xs = optimize.fmin_bfgs(p, np.array(x0)[:, axis], fprime=g,
                disp=False)
        for i, ai in enumerate(axis):
            x[ai] = xs[i]
        return x

    def saddle(self, x0, axis=(0, 1, 2), coord=np.identity(3), **kw):
        """find a saddle point close to x0 along the specified axes in
        the coordinate system coord"""
        kwargs = dict(dx_max=.1, xtol=1e-5, ftol=1e-5)
        kwargs.update(kw)
        x = np.array(x0)
        def f(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return self.potential(np.dot(coord, x), 0)[0]
        def g(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return rotate_tensor(self.potential(np.dot(coord, x), 1),
                    coord.T)[0, axis]
        h = rotate_tensor(self.potential(np.dot(coord, x), 2),
                coord.T)[0, axis, :][:, axis]
        # rational function optimization
        xs, p, ret = rfo(f, g, np.array(x0)[:, axis], h=h, **kwargs)
        if not ret in ("ftol", "xtol"):
            raise ValueError("%s", ((x0, axis, x, xs, p, ret),))
        # f(xs) # update x
        return x, p

    def modes(self, x, sorted=True):
        """returns curvatures and eigenmode vectors at x
        physical units of the trap frequenzies (Hz):
        scale = (q*u/(omega*scale))**2/(4*m)
        (scale*ew/scale**2/m)**.5/(2*pi)
        """
        ew, ev = np.linalg.eigh(self.potential(x, 2)[0])
        if sorted:
            i = ew.argsort()
            ew, ev = ew[i], ev[:, i]
        return ew, ev

    def trajectory(self, x0, v0, axis=(0, 1, 2),
            t0=0, dt=.0063*2*np.pi, t1=1e4, nsteps=1,
            integ="gni_irk2", methc=2, field=None, *args, **kwargs):
        """integrate the trajectory with initial position and speed x0,
        v0 along the specified axes (use symmetry to eliminate one),
        time step dt, from time t0 to time t1, yielding current position
        and speed every nsteps time steps"""
        integ = getattr(gni, integ)
        alpha_rf = 1.
        if field is None:
            field = lambda x0, t: self.time_potential(x0, 1, t,
                    alpha_rf=alpha_rf, expand=True)[0]
        t, p, q = t0, v0[:, axis], x0[:, axis]
        x0 = np.array(x0)
        def ddx(t, q, f):
            for i, ai in enumerate(axis):
                x0[ai] = q[i]
            f[:len(axis)] = field(x0, t)[axis, :]
        while t < t1:
            integ(ddx, nsteps, t, p, q, t+dt, methc, *args, **kwargs)
            t += dt
            yield t, q.copy(), p.copy()

    def stability(self, maxx, *a, **k):
        """integrate the trajectory (see :meth:trajectory()) as long as
        the position remains within a maxx ball or t>t1"""
        t, q, p = [], [], []
        for ti, qi, pi in self.trajectory(*a, **k):
            t.append(ti), q.append(qi), p.append(pi)
            if (qi**2).sum() > maxx**2:
                break
        return map(np.array, (t, q, p))

    def shims(self, x_coord_deriv, objectives=[], constraints=None):
        """
        solve the shim equations simultaneously at all points 
        [(x, rotation, derivative), ...]
        """
        obj = [PotentialObjective(x=x, derivative=deriv, value=0,
            rotation=coord) for x, coord, deriv in x_coord_deriv]
        obj += objectives
        if constraints is None:
            constraints = [PatternRangeConstraint(min=-1, max=1)]
        vectors = np.empty((len(obj), len(self)),
                np.double)
        for i, objective in enumerate(obj):
            objective.value = 1
            p, c = self.optimize(constraints+obj, verbose=False)
            objective.value = 0
            vectors[i] = p/c
        return vectors

    def solve(self, x, constraints, verbose=True):
        """
        optimize dc voltages at positions x to satisfy constraints.

        O(len(constraints)*len(x)*len(electrodes)) if sparse (most of the time)
        """
        v0 = [el.voltage_dc for el in self]
        for el in self:
            el.voltage_dc = 0.
        p0, f0, c0 = self.potential(x), self.gradient(x), self.curvature(x)
        for el, vi in zip(self, v0):
            el.voltage_dc = vi
        p, f, c = (self.individual_potential(x, i) for i in range(3))

        variables = []
        pots = []
        for i, xi in enumerate(x):
            v = cvxopt.modeling.variable(len(self), "u%i" % i)
            v.value = cvxopt.matrix(
                    [float(el.voltage_dc) for el in self])
            variables.append(v)
            pots.append((p0[i], f0[:, i], c0[:, :, i],
                p[:, i], f[:, :, i], c[:, :, :, i]))

        # setup constraint equations
        obj = 0.
        ctrs = []
        for ci in constraints:
            obj += sum(ci.objective(self, self, x, variables, pots))
            ctrs.extend(ci.constraints(self, self, x, variables, pots))
        solver = cvxopt.modeling.op(obj, ctrs)

        if not verbose:
            cvxopt.solvers.options["show_progress"] = False
        else:
            logger.info("variables: %i", sum(v._size
                    for v in solver.variables()))
            logger.info("inequalities: %i", sum(v.multiplier._size
                    for v in solver.inequalities()))
            logger.info("equalities: %i", sum(v.multiplier._size
                    for v in solver.equalities()))

        solver.solve("sparse")

        u = np.array([np.array(v.value).ravel() for v in variables])
        p = np.array([p0i+np.dot(pi, ui)
            for p0i, ui, pi in zip(p0,
                u, p.transpose(-1, -2))])
        f = np.array([f0i+np.dot(fi, ui)
            for f0i, ui, fi in zip(f0.transpose(-1, 0),
                u, f.transpose(-1, 0, -2))])
        c = np.array([c0i+np.dot(ci, ui)
            for c0i, ui, ci in zip(c0.transpose(-1, 0, 1),
                u, c.transpose(-1, 0, 1, -2))])
        return u, p, f, c

    def optimize(self, constraints, rcond=1e-9, verbose=True):
        """optimize this electrode voltages with respect to
        constraints"""
        p = cvxopt.modeling.variable(len(self))
        obj = []
        ctrs = []
        for ci in constraints:
            obj.extend(ci.objective(self, p))
            ctrs.extend(ci.constraints(self, p))
        B = np.array([i[0] for i in obj])
        b = np.array([i[1] for i in obj])
        # the inhomogeneous solution
        Bp = np.linalg.pinv(B, rcond=rcond)
        g = np.dot(Bp, b)
        g2 = np.inner(g, g)
        B1 = B - np.outer(b, g)/g2 # B*g_perp
        obj = cvxopt.modeling.dot(cvxopt.matrix(g), p) # maximize this
        #FIXME: there is one singular value, drop one line
        B1 = B1[:-1]
        # B*g_perp*p == 0
        ctrs.append(cvxopt.modeling.dot(cvxopt.matrix(B1.T), p) == 0.)
        solver = cvxopt.modeling.op(-obj, ctrs)
        if not verbose:
            cvxopt.solvers.options["show_progress"] = False
        else:
            cvxopt.solvers.options["show_progress"] = True
            logger.info("variables: %i", sum(v._size
                    for v in solver.variables()))
            logger.info("inequalities: %i", sum(v.multiplier._size
                    for v in solver.inequalities()))
            logger.info("equalities: %i", sum(v.multiplier._size
                    for v in solver.equalities()))
        solver.solve("sparse")
        if not solver.status == "optimal":
            raise ValueError("solve failed: %s" % solver.status)
        p = np.array(p.value, np.double).ravel()
        c = np.inner(p, g)/g2
        return p, c

    def group(self, thresholds=[0], voltages=None):
        if voltages is None:
            voltages = self.dcs
        if thresholds is None:
            threshold = sorted(np.unique(voltages))
        ts = [-np.inf] + list(thresholds) + [np.inf]
        eles = []
        for i, (ta, tb) in enumerate(zip(ts[:-1], ts[1:])):
            good = (ta <= voltages) & (voltages < tb)
            #if not np.any(good):
            #    continue
            paths = []
            dcs = []
            rfs = []
            for j in np.argwhere(good):
                el = self[j]
                paths.extend(el.paths)
                dcs.append(el.dc)
                rfs.append(el.rf)
            eles.append(PolygonPixelElectrode(paths=paths,
                dc=np.mean(dcs), rf=np.mean(rfs)))
        return System(eles)

    def mathieu(self, x, u_dc, u_rf, r=2, sorted=True):
        """return characteristic exponents (mode frequencies) and
        fourier components"""
        a = 4*u_dc*self.electrical_potential(x, "dc", 2, expand=True)[0]
        q = 2*u_rf*self.electrical_potential(x, "rf", 2, expand=True)[0]
        mu, b = mathieu(r, a, q)
        if sorted:
            i = mu.imag >= 0
            mu, b = mu[i], b[:, i]
            i = mu.imag.argsort()
            mu, b = mu[i], b[:, i]
        return mu/2, b

    # FIXME
    def analyze_static(self, x, axis=(0, 1, 2), do_ions=False,
            m=ct.atomic_mass, q=ct.elementary_charge, u=1.,
            l=100e-6, o=2*np.pi*1e6):
        scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
        dc_scale = scale/q # dc energy scale
        yield "u = %.3g V, f = %.3g MHz, m = %.3g amu, "\
                 "q = %.3g qe, l = %.3g µm, axis=%s" % (
                u, o/(2e6*np.pi), m/ct.atomic_mass,
                q/ct.elementary_charge, l/1e-6, axis)
        yield "energy scale: %.3g eV" % dc_scale
        yield "analyze point: %s (%s µm)" % (x, x*l/1e-6)
        trap = self.minimum(x, axis=axis)
        yield " minimum is at offset: %s" % (trap - x)
        p_rf = self.pseudo_potential(x, 0)[0]
        p_dc = self.electrical_potential(x, "dc", 0)[0]
        yield " rf, dc potentials: %.2g, %.2g (%.2g eV, %.2g eV)" % (
            p_rf, p_dc, p_rf*dc_scale, p_dc*dc_scale)
        try:
            xs, xsp = self.saddle(x+1e-2, axis=axis)
            yield " saddle offset, height: %s, %.2g (%.2g eV)" % (
                xs - x, xsp - p_rf - p_dc, (xsp - p_rf - p_dc)*dc_scale)
        except:
            yield " saddle not found"
        curves, modes_pp = self.modes(x)
        freqs_pp = (scale*curves/l**2/m)**.5/(2*np.pi)
        q_o2l2m = q/((l*o)**2*m)
        mu, b = self.mathieu(x, u_dc=dc_scale*q_o2l2m, u_rf=u*q_o2l2m,
                r=4, sorted=True)
        freqs = mu[:3].imag*o/(2*np.pi)
        modes = b[len(b)/2-3:len(b)/2, :3].real
        yield " pp+dc normal curvatures: %s" % curves
        yield " motion is bounded: %s" % np.allclose(0, mu.real)
        for nj, fj, mj in (("pseudopotential", freqs_pp, modes_pp),
                ("mathieu", freqs, modes)):
            yield " %s modes:" % nj
            for fi, mi in zip(fj, mj.T):
                yield "  %.4g MHz, %s" % (fi/1e6, mi)
            yield "  euler angles: %s" % (
                    np.rad2deg(np.array(euler_from_matrix(mj, "rxyz"))))
        se = (self.individual_potential(x, 1)[:, 0]**2).sum(0)/l**2
        yield " heating for 1 nV²/Hz white on each electrode:"
        yield "  field-noise psd: %s V²/(m² Hz)" % (se*1e-9**2)
        for fi, mi in zip(freqs_pp, modes_pp.T):
            sej = (np.abs(mi)*se**.5).sum()**2
            ndot = sej*q**2/(4*m*ct.h*fi)
            yield "  %.4g MHz: ndot=%.4g/s, S_E*f=%.4g (V² Hz)/(m² Hz)" % (
                fi/1e6, ndot*1e-9**2, sej*fi*1e-9**2)
        if do_ions:
            xi = x+np.random.randn(2)[:, None]*1e-3
            qi = np.ones(2)*q/(scale*l*4*np.pi*ct.epsilon_0)**.5
            xis, cis, mis = self.ions(xi, qi)
            freqs_ppi = (scale*cis/l**2/m)**.5/(1e6*np.pi)
            r2 = norm(xis[1]-xis[0])
            r2a = ((q*l)**2/(2*np.pi*ct.epsilon_0*scale*curves[0]))**(1/3.)
            yield " two ion modes:"
            yield "  separation: %.3g (%.3g µm, %.3g µm harmonic)" % (
                r2, r2*l/1e-6, r2a/1e-6)
            for fi, mi in zip(freqs_ppi, mis.transpose(2, 0, 1)):
                yield "  %.4g MHz, %s/%s" % (fi/1e6, mi[0], mi[1])

    def ions(self, x0, q):
        """find the minimum energy configuration of several ions with
        normalized charges q and starting positions x0, return their
        equilibrium positions, the mode frequencies and vectors"""
        n = len(x0)
        qs = q[:, None]*q[None, :]

        def f(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = .5*qs/np.ma.array(
                    x**2+y**2+z**2)**(1/2.)
            return (p0+pi.sum(-1)).sum()

        def g(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 1).T
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = qs*[x, y, z]/np.ma.array(
                    x**2+y**2+z**2)**(3/2.)
            return (p0+pi.sum(-1)).T.ravel()

        def h(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 2).T
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            p = expand_tensor(
                (-qs*[2*x**2-y**2-z**2, 3*x*y, 3*x*z,
                    2*y**2-x**2-z**2, 3*y*z]/np.ma.array(
                    x**2+y**2+z**2)**(5/2.)).T)
            p = p.transpose(2, 0, 3, 1)
            for i, (p0i, pii) in enumerate(
                    zip(p0.transpose(2, 0, 1), p.sum(2))):
                p[i, :, i, :] += p0i-pii
            return p.reshape(p.shape[0]*p.shape[1], -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            x = optimize.fmin_ncg(f=f, fprime=g, fhess=h, x0=x0.ravel(),
                disp=0)
            #print warn
            #x1, e0, e1, e2, itf, itg, warn = optimize.fmin_bfgs(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x1)-np.sort(x))/np.sort(x)
            #x2, e0, itf, itg, warn = optimize.fmin_cg(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x2)-np.sort(x))/np.sort(x)
            c = h(x)
        ew, ev = np.linalg.eigh(c)
        i = np.argsort(ew)
        ew, ev = ew[i], ev[i, :].reshape(n, 3, -1)
        return x.reshape(-1, 3), ew, ev
