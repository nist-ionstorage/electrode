# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011 Robert Jordens <jordens@phys.ethz.ch>
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


import warnings, itertools

import numpy as np
from scipy import optimize, constants as ct

from traits.api import HasTraits, List, Instance, Property

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
from .electrode import Electrode
from .utils import (select_tensor, expand_tensor, norm, rotate_tensor,
    apply_method, mathieu)


class _DummyPool(object):
    def apply_async(self, func, args=(), kwargs={}):
        class _DummyRet(object):
            def get(self):
                return func(*args, **kwargs)
        return _DummyRet()

_dummy_pool = _DummyPool()


class System(HasTraits):
    electrodes = List(Instance(Electrode))
    names = Property()
    voltages = Property()

    def _get_names(self):
        return [el.name for el in self.electrodes]

    def _get_voltages(self):
        return [(el.voltage_dc, el.voltage_rf)
                for el in self.electrodes]

    def _set_voltages(self, voltages):
        for el, (v_dc, v_rf) in zip(self.electrodes, voltages):
            el.voltage_dc = v_dc
            el.voltage_rf = v_rf

    def electrode(self, name):
        """return the first electrode named name or None if not found"""
        return self.electrodes[self.names.index(name)]

    def with_voltages(self, voltages, function, *args, **kwargs):
        """execute function(*args, **kwargs) with voltages set to voltages"""
        voltages, self.voltages = self.voltages, voltages
        ret = function(*args, **kwargs)
        self.voltages = voltages
        return ret

    def get_potentials(self, x, typ, *d):
        x = np.atleast_2d(x)
        n = x.shape[0]
        e = [np.zeros((n)), np.zeros((3,n)), np.zeros((3,3,n)),
                np.zeros((3,3,3,n))]
        for el in self.electrodes:
            v = getattr(el, "voltage_"+typ)
            if v != 0:
                for di, pi in zip(d, el.potential(x, *d)):
                    e[di] += v*pi
        return [e[di] for di in d]

    def field_dc(self, x):
        return self.get_potentials(x, "dc", 1)[0]

    def field_rf(self, x, t=0.):
        return np.cos(t)*self.get_potentials(x, "rf", 1)[0]

    def field(self, x, t=0.):
        """electrical field at an instant in time t (physical units:
        1/omega_rf), no adiabatic approximation here"""
        return self.field_dc(x) + self.field_rf(x, t)

    def potential_dc(self, x):
        return self.get_potentials(x, "dc", 0)[0]

    def potential_rf(self, x):
        e_rf, = self.get_potentials(x, "rf", 1)
        return (e_rf**2).sum(axis=0)

    def potential(self, x):
        """combined electrical and ponderomotive potential"""
        return self.potential_dc(x) + self.potential_rf(x)

    def gradient_dc(self, x):
        return self.get_potentials(x, "dc", 1)[0]

    def gradient_rf(self, x):
        e_rf, c_rf = self.get_potentials(x, "rf", 1, 2)
        return 2*(e_rf[:, None]*c_rf).sum(axis=0)

    def gradient(self, x):
        """gradient of the combined electric and ponderomotive
        potential"""
        return self.gradient_dc(x) + self.gradient_rf(x)

    def curvature_dc(self, x):
        return self.get_potentials(x, "dc", 2)[0]

    def curvature_rf(self, x):
        e_rf, c_rf, d_rf = self.get_potentials(x, "rf", 1, 2, 3)
        return 2*(c_rf[:, :, None]*c_rf[:, None, :]+
                e_rf[:, None, None]*d_rf).sum(axis=0)

    def curvature(self, x):
        """curvature of the combined electric and ponderomotive
        potential"""
        return self.curvature_dc(x) + self.curvature_rf(x)

    def parallel(self, pool, x, y, z, fn="potential"):
        """paralelize the calculation of method fn over the indices of
        the x axis"""
        r = [pool.apply_async(apply_method, (self, fn,
                np.array([x[i], y[i], z[i]]).reshape(3, -1).T))
            for i in range(x.shape[0])]
        r = np.array([ri.get().reshape((x.shape[1], x.shape[2]))
            for ri in r])
        return r

    def plot(self, ax, *a, **k):
        """plot electrodes with random colors"""
        for e, c in zip(self.electrodes, itertools.cycle("bgrcmy")):
            e.plot(ax, color=c, alpha=.5, *a, **k)

    def plot_voltages(self, ax, u=None, els=None):
        """plot electrodes with alpha proportional to voltage (scaled to
        max abs voltage being opaque), red for positive, blue for
        negative"""
        if els is None:
            els = self.electrodes
        if u is None:
            u = np.array([el.voltage_dc for el in els])
        um = abs(u).max() or 1.
        for el, ui in zip(els, u):
            el.plot(ax, color=(ui > 0 and "red" or "blue"),
                    alpha=abs(ui)/um, text="")

    def minimum(self, x0, axis=(0, 1, 2), coord=np.identity(3)):
        """find a potential minimum near x0 searching along the
        specified axes in the orthonormal matrix coord"""
        x = np.array(x0)
        def p(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return float(self.potential(np.dot(coord, x)))
        # downhill simplex
        xs = optimize.fmin(p, np.array(x0)[:, axis], disp=False)
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
            return float(self.potential(np.dot(coord, x)))
        def g(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return rotate_tensor(self.gradient(np.dot(coord, x)),
                    coord.T)[axis, 0]
        h = rotate_tensor(self.curvature(np.dot(coord, x)),
                coord.T)[axis, :][:, axis, 0]
        # rational function optimization
        xs, p, ret = rfo(f, g, np.array(x0)[:, axis], h=h, **kwargs)
        if not ret in ("ftol", "xtol"):
            raise ValueError, (x0, axis, x, xs, p, ret)
        # f(xs) # update x
        return x, p

    def modes(self, x, sorted=True):
        """returns curvatures and eigenmode vectors at x
        physical units of the trap frequenzies (Hz):
        scale = (q*u/(omega*scale))**2/(4*m)
        (scale*ew/scale**2/m)**.5/(2*pi)
        """
        ew, ev = np.linalg.eigh(self.curvature(x)[:, :, 0])
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
        if field is None:
            field = self.field
        t, p, q = t0, v0[:, axis], x0[:, axis]
        x0 = np.array(x0)
        def ddx(t, q, f):
            for i, ai in enumerate(axis):
                x0[ai] = q[i]
            f[:len(axis)] = field(x0, t)[axis, 0]
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

    def effects(self, x, electrodes=None, pool=_dummy_pool):
        """
        return potential, gradient and curvature for the system at x and
        contribution of each of the specified electrodes per volt

        O(len(electrodes)*len(x))
        """
        if electrodes is None:
            electrodes = self.electrodes
        x = np.atleast_2d(x) # n,3 positions
        r = [pool.apply_async(apply_method,
            (el, "potential", x, 0, 1, 2)) for el in electrodes]
        p, f, c = zip(*map(lambda i: i.get(), r))
        # m,n = len(electrodes), len(x)
        # m,n potentials for electrodes at x0
        p = np.array(p)
        # 3,m,n forces for electrodes at x0
        f = np.array(f).transpose((1, 0, 2))
        # 3,3,m,n curvatures for electrodes at x0
        c = np.array(c).transpose((1, 2, 0, 3))
        return p, f, c

    def shims(self, x, electrodes=None, forces=None, curvatures=None,
            coords=None, rcond=1e-3):
        """
        solve the shim equations simultaneously at all points x0,
        return an array of voltages (x, y, z, xx, xy, yy, xz, yz) *
        len(x0). -zz=xx+yy is dropped.
        """
        x = np.atleast_2d(x)
        if electrodes is None:
            electrodes = self.electrodes
        p, f, c = self.effects(x, electrodes)
        f = f.transpose((-1, 0, 1)) # x,fi,el
        c = c.transpose((-1, 0, 1, 2)) # x,ci,cj,el
        dc = []
        for i, (fi, ci) in enumerate(zip(f, c)): # over x
            if coords is not None:
                fi = rotate_tensor(fi, coords[i], 1)
                ci = rotate_tensor(ci, coords[i], 2)
            ci = select_tensor(ci)
            if forces is not None:
                fi = fi[forces[i], ...]
            if curvatures is not None:
                ci = ci[curvatures[i], ...]
            dc.extend(fi)
            dc.extend(ci)
        ev = np.identity(len(dc))
        u, res, rank, sing = np.linalg.lstsq(dc, ev, rcond=rcond)
        return u, (res, rank, sing)

    def solve(self, x, constraints, 
            electrodes=None, verbose=True, pool=_dummy_pool):
        """
        optimize dc voltages at positions x to satisfy constraints.

        O(len(constraints)*len(x)*len(electrodes)) if sparse (most of the time)
        """
        if electrodes is None:
            electrodes = self.electrodes
        v0 = [el.voltage_dc for el in electrodes]
        for el in electrodes:
            el.voltage_dc = 0.
        p0, f0, c0 = self.potential(x), self.gradient(x), self.curvature(x)
        for el, vi in zip(electrodes, v0):
            el.voltage_dc = vi
        p, f, c = self.effects(x, electrodes, pool)

        variables = []
        pots = []
        for i, xi in enumerate(x):
            v = cvxopt.modeling.variable(len(electrodes), "u%i" % i)
            v.value = cvxopt.matrix(
                    [float(el.voltage_dc) for el in electrodes])
            variables.append(v)
            pots.append((p0[i], f0[:, i], c0[:, :, i],
                p[:, i], f[:, :, i], c[:, :, :, i]))

        # setup constraint equations
        obj = 0.
        ctrs = []
        for ci in constraints:
            obj += sum(ci.objective(self, electrodes, x, variables, pots))
            ctrs.extend(ci.constraints(self, electrodes, x, variables, pots))
        solver = cvxopt.modeling.op(obj, ctrs)

        if not verbose:
            cvxopt.solvers.options["show_progress"] = False
        else:
            print "variables:", sum(v._size
                    for v in solver.variables())
            print "inequalities", sum(v.multiplier._size
                    for v in solver.inequalities())
            print "equalities", sum(v.multiplier._size
                    for v in solver.equalities())

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

    def mathieu(self, x, u_dc, u_rf, r=2, sorted=True):
        """return characteristic exponents (mode frequencies) and
        fourier components"""
        c_rf, = self.get_potentials(x, "rf", 2)
        c_dc, = self.get_potentials(x, "dc", 2)
        a = 2*u_dc*c_dc[..., 0] # mathieu(*a) takes each ai *2!
        q = 2*u_rf*c_rf[..., 0]
        mu, b = mathieu(r, a, q)
        if sorted:
            i = mu.imag >= 0
            mu, b = mu[i], b[:, i]
            i = mu.imag.argsort()
            mu, b = mu[i], b[:, i]
        return mu/2, b
    
    def analyze_static(self, x, axis=(0, 1, 2), do_ions=False,
            m=1., q=1., u=1., l=1., o=1.):
        scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
        dc_scale = scale/q # dc energy scale
        yield "u = %.3g V, f = %.3g MHz, m = %.3g amu, "\
                 "q = %.3g qe, l = %.3g µm, axis=%s" % (
                u, o/(2e6*np.pi), m/ct.atomic_mass,
                q/ct.elementary_charge, l/1e-6, axis)
        yield "analyze point: %s (%s µm)" % (x, x*l/1e-6)
        trap = self.minimum(x, axis=axis)
        yield " minimum is at offset: %s" % (trap - x)
        p_rf, p_dc = self.potential_rf(x), self.potential_dc(x)
        yield " rf, dc potentials: %.2g, %.2g (%.2g eV, %.2g eV)" % (
            p_rf, p_dc, p_rf*scale/q, p_dc*dc_scale)
        try:
            xs, xsp = self.saddle(x+1e-2, axis=axis)
            yield " saddle offset, height: %s, %.2g (%.2g eV)" % (
                xs - x, xsp - p_rf - p_dc, (xsp - p_rf - p_dc)*scale/q)
        except:
            yield " saddle not found"
        curves, modes_pp = self.modes(x)
        freqs_pp = (scale*curves/l**2/m)**.5/(2*np.pi)
        q_o2l2m = q/((l*o)**2*m)
        mu, b = self.mathieu(x, u_dc=dc_scale*q_o2l2m, u_rf=u*q_o2l2m,
                r=3, sorted=True)
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
                    np.array(euler_from_matrix(mj, "rxyz"))*180/np.pi)
        se = sum(list(el.potential(x, 1))[0][:, 0]**2
                for el in self.electrodes)/l**2
        ee = sum(list(el.potential(x, 0))[0][0]**2
                for el in self.electrodes)
        yield " heating for 1 nV²/Hz white on each electrode:"
        yield "  field-noise psd: %s V²/(m² Hz)" % (se*1e-9**2)
        yield "  pot-noise psd: %s V²/Hz" % (ee*1e-9**2)
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
            yield "  separation: %.3g (%.3g µm, %.3g µm analytic)" % (
                r2, r2*l/1e-6, r2a/1e-6)
            for fi, mi in zip(freqs_ppi, mis.transpose(2, 0, 1)):
                yield "  %.4g MHz, %s/%s" % (fi/1e6, mi[0], mi[1])

    def analyze_shims(self, x, electrodes=None,
            forces=None, curvatures=None, use_modes=True):
        x = np.atleast_2d(x)
        if electrodes is None:
            electrodes = [e.name for e in self.electrodes
                    if e.voltage_rf == 0.]
        els = [self.electrode(n) for n in electrodes]
        if use_modes:
            coords = [self.modes(xi)[1] for xi in x]
        else:
            coords = None
        fn = "x y z".split()
        cn = "xx xy xz yy yz".split()
        if forces is None:
            forces = [fn] * len(x)
        if curvatures is None:
            curvatures = [cn] * len(x)
        fx = [[fn.index(fi) for fi in fj] for fj in forces]
        cx = [[cn.index(ci) for ci in cj] for cj in curvatures]
        us, (res, rank, sing) = self.shims(x, els, fx, cx, coords)
        yield "shim analysis for points: %s" % x
        yield " forces: %s" % forces
        yield " curvatures: %s" % curvatures
        yield " matrix shape: %s, rank: %i" % (us.shape, rank)
        yield " electrodes: %s" % np.array(electrodes)
        n = 0
        for i in range(len(x)):
            for ni in forces[i]+curvatures[i]:
                yield " sh_%i%-2s: %s" % (i, ni, us[:, n])
                n += 1
        yield forces, curvatures, us

    def ions(self, x0, q):
        """find the minimum energy configuration of several ions with
        normalized charges q and starting positions x0, return their
        equilibrium positions and the mode frequencies and vectors"""
        n = len(x0)
        qs = q[:, None]*q[None, :]

        def f(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = .5*qs/np.ma.array(
                    x**2+y**2+z**2)**(1/2.)
            return (p0+pi.sum(-1)).sum()

        def g(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.gradient(x0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = qs*[x, y, z]/np.ma.array(
                    x**2+y**2+z**2)**(3/2.)
            return (p0+pi.sum(-1)).T.ravel()

        def h(x0):
            x0 = x0.reshape(-1, 3)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            i, j = np.indices(x.shape)
            p0 = self.curvature(x0)
            p = expand_tensor(
                -qs*[2*x**2-y**2-z**2, 3*x*y, 3*x*z,
                    2*y**2-x**2-z**2, 3*y*z]/np.ma.array(
                    x**2+y**2+z**2)**(5/2.))
            p = p.transpose(2, 0, 3, 1)
            for i, (p0i, pii) in enumerate(
                    zip(p0.transpose(2, 0, 1), p.sum(2))):
                p[i, :, i, :] += p0i-pii
            return p.reshape(p.shape[0]*p.shape[1], -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            x, e0, itf, itg, ith, warn = optimize.fmin_ncg(
                f=f, fprime=g, fhess=h, x0=x0.ravel(), full_output=1,
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



