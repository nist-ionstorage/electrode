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

import warnings, itertools
from contextlib import contextmanager

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
    mathieu, name_to_deriv)
from .pattern_constraints import (PatternRangeConstraint,
        PotentialObjective)
from . import colors


class System(HasTraits):
    electrodes = List(Instance(Electrode))
    names = Property()
    dcs = Property()
    rfs = Property()

    def __init__(self, *electrodes, **kwargs):
        super(System, self).__init__(**kwargs)
        self.electrodes.extend(electrodes)
    
    def _get_names(self):
        return [el.name for el in self.electrodes]

    def _set_names(self, names):
        for ei, ni in zip(self.electrodes, names):
            ei.name = ni

    def _get_dcs(self):
        return [el.dc for el in self.electrodes]

    def _set_dcs(self, voltages):
        for ei, vi in zip(self.electrodes, voltages):
            ei.dc = vi

    def _get_rfs(self):
        return [el.rf for el in self.electrodes]

    def _set_rfs(self, voltages):
        for ei, vi in zip(self.electrodes, voltages):
            ei.rf = vi

    def electrode(self, name):
        """return the first electrode named name or None if not found"""
        for ei in self.electrodes:
            if ei.name == name:
                return ei

    __getitem__ = electrode
    by_name = electrode

    @contextmanager
    def with_voltages(self, dcs=None, rfs=None):
        """contextmanager with set to voltages"""
        orig = self.dcs, self.rfs
        if dcs is not None:
            self.dcs = dcs
        if rfs is not None:
            self.rfs = rfs
        yield
        self.dcs, self.rfs = orig

    def electrical_potential(self, x, typ="dc", derivative=0,
            expand=False):
        x = np.atleast_2d(x).astype(np.double)
        pot = np.zeros((x.shape[0], 2*derivative+1))
        for ei in self.electrodes:
            vi = getattr(ei, typ)
            if vi != 0.:
                pot += vi*ei.potential(x, derivative)
        if expand:
            pot = expand_tensor(pot)
        return pot

    def individual_potential(self, x, derivative=0):
        """
        return potential, gradient and curvature for the system at x and
        contribution of each of the specified electrodes per volt

        O(len(electrodes)*len(x))
        """
        x = np.atleast_2d(x) # n,3 positions
        eff = np.array([el.potential(x, derivative) for el in self.electrodes])
        assert eff.shape == (len(self.electrodes), x.shape[0],
                2*derivative+1)
        return eff

    def time_potential(self, x, t=0., derivative=0, expand=False):
        """electrical field at an instant in time t (physical units:
        1/omega_rf), no adiabatic approximation here"""
        dc, rf = (self.electrical_potential(x, typ, derivative, expand)
                for typ in ("dc", "rf"))
        return dc + np.cos(t)*rf

    def pseudo_potential(self, x, derivative=0):
        p = [self.electrical_potential(x, "rf", i, expand=True)
                for i in range(1, derivative+2)]
        if derivative == 0:
            return np.einsum("ij,ij->i", p[0], p[0])
        elif derivative == 1:
            return 2*np.einsum("ik,ijk->ij", p[0], p[1])
        elif derivative == 2:
            return 2*(np.einsum("ijl,ikl->ijk", p[1], p[1])
                    + np.einsum("il,ijkl->ijk", p[0], p[2]))
        elif derivative == 3:
            return 2*(np.einsum("im,ijklm->ijkl", p[0], p[3])
                    + np.einsum("ijm,iklm->ijkl", p[1], p[2])
                    + np.einsum("ikm,ijlm->ijkl", p[1], p[2])
                    + np.einsum("ilm,ijkm->ijkl", p[1], p[2]))
        elif derivative == 4:
            return 2*(np.einsum("in,ijklmn->ijklm", p[0], p[5])
                    + np.einsum("ijn,iklmn->ijklm", p[1], p[3])
                    + np.einsum("ikn,ijlmn->ijklm", p[1], p[3])
                    + np.einsum("iln,ijkmn->ijklm", p[1], p[3])
                    + np.einsum("imn,ijkln->ijklm", p[1], p[3])
                    + np.einsum("ijmn,ikln->ijklm", p[2], p[4])
                    + np.einsum("ikmn,ijln->ijklm", p[2], p[4])
                    + np.einsum("ilmn,ijkn->ijklm", p[2], p[4]))

    def potential(self, x, derivative=0):
        """combined electrical and ponderomotive potential"""
        dc = self.electrical_potential(x, "dc", derivative,
                expand=True)
        rf = self.pseudo_potential(x, derivative)
        return dc + rf

    def plot(self, ax, alpha=.3, *a, **k):
        """plot electrodes with sequential colors"""
        for e, c in zip(self.electrodes, itertools.cycle(colors.set3)):
            e.plot(ax, color=tuple(c/255.), alpha=alpha, *a, **k)

    def plot_voltages(self, ax, u=None, um=None, *a, **kw):
        """plot electrodes with alpha proportional to voltage (scaled to
        max abs voltage being opaque), red for positive, blue for
        negative"""
        if u is None:
            u = np.array([el.voltage_dc for el in els])
        if um is None:
            um = abs(u).max() or 1.
        for el, ui in zip(els, u):
            el.plot(ax, color=(ui > 0 and "red" or "blue"),
                    alpha=abs(ui)/um, text="", *a, **kw)

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
            raise ValueError, (x0, axis, x, xs, p, ret)
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
        if field is None:
            field = lambda x0, t: self.time_potential(x0, t, 1)[0]
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

    def shims(self, x_coord_deriv, constraints=None):
        """
        solve the shim equations simultaneously at all points 
        [(x, rotation, derivative), ...]
        """
        objectives = [PotentialObjective(x=x, derivative=deriv, value=0,
                rotation=coord) for x, coord, deriv in x_coord_deriv]
        if constraints is None:
            constraints = [PatternRangeConstraint(min=-1, max=1)]
        vectors = []
        for objective in objectives:
            objective.value = 1
            p, c = self.optimize(constraints+objectives, verbose=False)
            objective.value = 0
            vectors.append(p/c)
        return np.array(vectors)

    def solve(self, x, constraints, verbose=True):
        """
        optimize dc voltages at positions x to satisfy constraints.

        O(len(constraints)*len(x)*len(electrodes)) if sparse (most of the time)
        """
        v0 = [el.voltage_dc for el in self.electrodes]
        for el in self.electrodes:
            el.voltage_dc = 0.
        p0, f0, c0 = self.potential(x), self.gradient(x), self.curvature(x)
        for el, vi in zip(self.electrodes, v0):
            el.voltage_dc = vi
        p, f, c = (self.individual_potential(x, i) for i in range(3))

        variables = []
        pots = []
        for i, xi in enumerate(x):
            v = cvxopt.modeling.variable(len(self.electrodes), "u%i" % i)
            v.value = cvxopt.matrix(
                    [float(el.voltage_dc) for el in self.electrodes])
            variables.append(v)
            pots.append((p0[i], f0[:, i], c0[:, :, i],
                p[:, i], f[:, :, i], c[:, :, :, i]))

        # setup constraint equations
        obj = 0.
        ctrs = []
        for ci in constraints:
            obj += sum(ci.objective(self, self.electrodes, x, variables, pots))
            ctrs.extend(ci.constraints(self, self.electrodes, x, variables, pots))
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

    def optimize(self, constraints, verbose=True):
        """optimize this electrode voltages with respect to
        constraints"""
        p = cvxopt.modeling.variable(len(self.electrodes))
        obj = []
        ctrs = []
        for ci in constraints:
            obj.extend(ci.objective(self, p))
            ctrs.extend(ci.constraints(self, p))
        B = np.matrix([i[0] for i in obj])
        b = np.matrix([i[1] for i in obj])
        # the inhomogeneous solution
        g = b*np.linalg.pinv(B).T
        # maximize this
        obj = cvxopt.matrix(g)*p
        # B*g_perp
        B1 = B - b.T*g/(g*g.T)
        if False:
            u, l, v = np.linalg.svd(B1)
            li = np.argmin(l)
            print li, l[li], v[li], B1*v[li].T
            return np.array(v)[li], 0
        #FIXME: there is one singular value, drop one constraint
        B1 = B1[:-1]
        # B*g_perp*p == 0
        ctrs.append(cvxopt.matrix(B1)*p == 0.)
        solver = cvxopt.modeling.op(-obj, ctrs)
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
        c = float(np.matrix(p.value).T*g.T/(g*g.T))
        p = np.array(p.value).ravel()
        return p, c

    def split(self, thresholds=[0]):
        if thresholds is None:
            threshold = sorted(np.unique(self.pixel_factors))
        ts = [-np.inf] + thresholds + [np.inf]
        eles = []
        for i, (ta, tb) in enumerate(zip(ts[:-1], ts[1:])):
            good = (ta <= self.pixel_factors) & (self.pixel_factors < tb)
            if not np.any(good):
                continue
            paths = [self.paths[j] for j in np.argwhere(good)]
            name = "%s_%i" % (self.name, i)
            eles.append(self.__class__(name=name, paths=paths,
                voltage_dc=self.voltage_dc, voltage_rf=self.voltage_rf))
        return eles

    def mathieu(self, x, u_dc, u_rf, r=2, sorted=True):
        """return characteristic exponents (mode frequencies) and
        fourier components"""
        c_rf = self.electrical_potential(x, "rf", 2)
        c_dc = self.electrical_potential(x, "dc", 2)
        a = 4*u_dc*c_dc[..., 0]
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
        yield "energy scale: %.3g eV" % dc_scale
        yield "analyze point: %s (%s µm)" % (x, x*l/1e-6)
        trap = self.minimum(x, axis=axis)
        yield " minimum is at offset: %s" % (trap - x)
        p_rf = self.pseudo_potential(x, 0)
        p_dc = self.electrical_potential(x, "dc", 0)
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
        se = sum(list(el.potential(x, 1))[0]**2
                for el in self.electrodes)/l**2
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

    def analyze_shims(self, x, forces=None, curvatures=None,
            use_modes=True, **kwargs):
        x = np.atleast_2d(x)
        els = self.electrodes
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
        us, (res, rank, sing) = self.shims(x, els, fx, cx, coords,
                **kwargs)
        yield "shim analysis for points: %s" % x
        yield " forces: %s" % forces
        yield " curvatures: %s" % curvatures
        yield " matrix shape: %s, rank: %i" % (us.shape, rank)
        yield " electrodes: %s" % np.array(self.names)
        n = 0
        for i in range(len(x)):
            for ni in forces[i]+curvatures[i]:
                yield " sh_%i%-2s: %s" % (i, ni, us[:, n])
                n += 1
        yield us

    def ions(self, x0, q):
        """find the minimum energy configuration of several ions with
        normalized charges q and starting positions x0, return their
        equilibrium positions and the mode frequencies and vectors"""
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
            p0 = self.potential(x0, 1)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = qs*[x, y, z]/np.ma.array(
                    x**2+y**2+z**2)**(3/2.)
            return (p0+pi.sum(-1)).T.ravel()

        def h(x0):
            x0 = x0.reshape(-1, 3)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            i, j = np.indices(x.shape)
            p0 = self.potential(x0, 2)
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
