# -*- coding: utf8 -*-
#
#   electrode.py: numeric tools for Paul traps
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

from __future__ import absolute_import, print_function, unicode_literals

import unittest
from numpy import testing as nptest

import numpy as np
from scipy import constants as ct
import matplotlib.pyplot as plt

from electrode import (transformations, utils, electrode, system,
    pattern_constraints)


class ThreefoldOptimizeCase(unittest.TestCase):
    def hextess(self, n, points=False):
        x = np.array(sum(([np.array([i+j*.5, j*3**.5*.5])/(n+.5)
            for j in range(-n-min(0, i), n-max(0, i)+1)]
            for i in range(-n, n+1)), []))
        if points:
            a = np.ones(len(x))*3**.5/(n+.5)**2/2
            return [electrode.PointPixelElectrode(points=[xi],
                areas=[ai]) for xi, ai in zip(x, a)]
        else:
            a = 1/(3**.5*(n+.5)) # edge length
            p = x[:, None, :] + [[[a*np.cos(phi), a*np.sin(phi)] for phi in
                np.arange(np.pi/6, 2*np.pi, np.pi/3)]]
            return [electrode.PolygonPixelElectrode(paths=[i]) for i in p]

    def get(self, n=12, h=1/8., d=1/4., H=25/8., nmax=1, points=True):
        s = system.System(electrodes=self.hextess(n, points))
        for ei in s.electrodes:
            ei.cover_height = H
            ei.cover_nmax = nmax
        self.s = s

        ct = []
        ct.append(pattern_constraints.PatternRangeConstraint(min=0, max=1.))
        for p in 0, 4*np.pi/3, 2*np.pi/3:
            x = np.array([d/3**.5*np.cos(p), d/3**.5*np.sin(p), h])
            r = transformations.euler_matrix(p, np.pi/2, np.pi/4, "rzyz")[:3, :3]
            for i in "x y z xy xz yz".split():
                ct.append(pattern_constraints.PotentialObjective(derivative=i,
                    x=x, rotation=r, value=0))
            for i in "xx yy".split():
                ct.append(pattern_constraints.PotentialObjective(derivative=i,
                        x=x, rotation=r, value=2**(-1/3.)))
        s.rfs, self.c = s.optimize(ct, verbose=False)
        self.h = h
        
        self.x0 = np.array([d/3**.5, 0, h])
        self.r = transformations.euler_matrix(0, np.pi/2, np.pi/4, "rzyz")[:3, :3]

    def test_points(self):
        self.get(points=True)
        nptest.assert_allclose(self.c*self.h**2, .16381, rtol=1e-4)
        nptest.assert_allclose(
                self.s.electrical_potential(self.x0, "rf", 1), 0, atol=1e-9)

    def test_poly(self):
        self.get(points=False)
        nptest.assert_allclose(self.c*self.h**2, .13943, rtol=1e-4)
        nptest.assert_allclose(
                self.s.electrical_potential(self.x0, "rf", 1), 0, atol=1e-9)

    def test_curve(self):
        self.get()
        c = self.s.electrical_potential(self.x0, "rf", 2, expand=True)
        c = utils.rotate_tensor(c, self.r)
        nptest.assert_almost_equal(c[0]/self.c,
            2**(-1/3.)*np.eye(3)*[1, 1, -2])

    def test_main_saddle(self):
        self.get()
        xs, xsp = self.s.saddle((0, 0, .5), axis=(0, 1, 2,))
        nptest.assert_almost_equal(xs, [0, 0, .5501], decimal=4)
        nptest.assert_almost_equal(xsp, .1662, decimal=4)

    def test_single_saddle(self):
        self.get()
        xs, xsp = self.s.saddle(self.x0+[.02, 0, .02], axis=(0, 2), dx_max=.02)
        nptest.assert_almost_equal(xs, [.145, 0, .156], decimal=3)
        nptest.assert_almost_equal(xsp, .0109, decimal=3)
        xs1, xsp1 = self.s.saddle(self.x0+[.0, 0, .02], axis=(0, 1, 2), dx_max=.02)
        nptest.assert_almost_equal(xs, xs1, decimal=3)
        nptest.assert_almost_equal(xsp, xsp1, decimal=3)


class FiveWireCase(unittest.TestCase):
    def trap(self, tw=np.pi/4, t0=5*np.pi/8):
        s = system.System()
        rmax = 1e4
        # Janus H. Wesenberg, Phys Rev A 78, 063410 ͑2008͒
        def patches(n, tw, t0):
            for i in range(n):
                a = t0 + 2*np.pi/n*i
                ya, yb = np.tan((a-tw/2)/2), np.tan((a+tw/2)/2)
                yield np.array([[rmax, ya], [rmax, yb],
                     [-rmax, yb], [-rmax, ya]])
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
            rf=1, paths=list(patches(n=2, tw=tw, t0=t0))))
        return s

    def setUp(self):
        self.s = self.trap()

    def test_minimum(self):
        x0 = self.s.minimum((0,0,1.), axis=(1, 2))
        nptest.assert_almost_equal(x0, [0, 0, 1.], decimal=3)

    def test_low_rf(self):
        p = self.s.potential((0,0,1.))
        nptest.assert_almost_equal(p, 0, decimal=3)

    def test_saddle(self):
        xs, xsp = self.s.saddle((0,0,1.1), axis=(1, 2))
        nptest.assert_almost_equal(xs, [0, -.125, 1.8], decimal=3)
        nptest.assert_almost_equal(xsp, .0036, decimal=4)

    def test_scale(self):
        q = 1*ct.elementary_charge
        u = 100.
        m = 10*ct.atomic_mass
        d = 100e-6
        o = 100e6*2*np.pi
        scale = q**2*u**2/(4*m*o**2*d**2)
        nptest.assert_almost_equal(scale, 6.1*ct.electron_volt, decimal=3)
        nptest.assert_almost_equal(.0036*scale,
                2*55.8e-3*ct.electron_volt, decimal=4)
        nptest.assert_almost_equal((.1013*scale/m)**.5/(d*2*np.pi),
                3.889e6, decimal=-3)

    def test_modes(self):
        o0, e0 = self.s.modes([0, 0, 1])
        nptest.assert_almost_equal(o0, [0, .1013, .1013], decimal=3)
        abc = np.array(transformations.euler_from_matrix(e0))/2/np.pi
        nptest.assert_allclose(abc, [0, 0, 0], atol=1, rtol=1e-3)

    def test_integrate(self):
        h = 1.
        x0 = np.array([0, 0, 1.])
        dt, vu0, uu = .01, .01, .5
        self.s.electrode("rf").rf = uu*h**2
        t, x, v = [], [], []
        for ti, xi, vi in self.s.trajectory(
                x0, np.array([0, 0, vu0*uu*h]), axis=(1, 2),
                t1=20*2*np.pi, dt=dt*2*np.pi):
            t.append(ti)
            x.append(xi)
            v.append(vi)

        t = np.array(t)
        x = np.array(x)
        v = np.array(v)

        self.assertEqual(np.alltrue(utils.norm(x, axis=1)<3), True)
        self.assertEqual(np.alltrue(utils.norm(v, axis=1)<1), True)

        avg = int(1/dt)
        kin = (((x[:-avg]-x[avg:])/(2*np.pi))**2).sum(axis=-1)/2*4 # 4?
        pot = self.s.potential(np.array([x0[0]+0*x[:,0], x[:,0], x[:,1]]).T)
        pot = pot[avg/2:-avg/2]
        t = t[avg/2:-avg/2]
        do_avg = lambda ar: ar[:ar.size/avg*avg].reshape(
                (-1, avg)).mean(axis=-1)
        t, kin, pot = map(do_avg, (t, kin, pot))

        self.assertEqual(np.alltrue(np.std(kin+pot)/np.mean(kin+pot)<.01),
                True)

    def test_minimum_four(self):
        # four wire trap has max depth
        s = self.trap(tw=np.pi/2-1e-6, t0=np.pi/4-1e-6)
        x0 = self.s.minimum((0,0,1.), axis=(1, 2))
        nptest.assert_almost_equal(x0, [0, 0, 1.], decimal=3)

    def test_saddle_four(self):
        # four wire trap has max depth
        s = self.trap(tw=np.pi/2-1e-6, t0=np.pi/4-1e-6)
        xs, xsp = s.saddle((0,0,1.1), axis=(1, 2))
        nptest.assert_almost_equal(xs, [0, 0, (2+5**.5)**.5], decimal=4)
        nptest.assert_almost_equal(xsp, (5*5**.5-11)/(2*np.pi**2), decimal=4)

    def test_shaper(self):
        x = np.mgrid[:3, :4, :5]
        s = self.trap()
        a = utils.shaper(s.potential, x, 2)
        b = s.potential(x.reshape(3, -1).T, 2)
        nptest.assert_allclose(a, b.reshape(x.shape[1:] + b.shape[1:]))



class MagtrapCase(unittest.TestCase):
    def magtrap(self):
        s = system.System()
        rmax = 1e3
        a, b, c, d, e = -1.3, 1.3, .78, 2.5, 3.
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
                rf=1., paths=[
                    [[rmax,0], [-rmax,0], [-rmax,a], [rmax,a]],
                    [[rmax,b+c], [-rmax,b+c], [-rmax,b], [rmax,b]],
                ]))
        for cx, cy, w, n in [
            (d/2+e/2, b+2*c, d, "c1"),
            (0, b+2*c, e, "c2"),
            (-d/2-e/2, b+2*c, d, "c3"),
            (d/2+e/2, a, d, "c6"),
            (0, a, e, "c5"),
            (-d/2-e/2, a, d, "c4")]:
                s.electrodes.append(electrode.PolygonPixelElectrode(
                    name=n, paths=[[
                        [cx-w/2,cy], [cx-w/2,rmax*np.sign(cy)],
                        [cx+w/2,rmax*np.sign(cy)], [cx+w/2,cy],
                    ]]))
        for e in s.electrodes:
            e.paths = [p[::o] for p, o in zip(e.paths, e.orientations())]
        return s

    def setUp(self):
        self.s = self.magtrap()
        self.s["c1"].dc = 1e-4 # lift degeneracy
        self.x0 = self.s.minimum((.01, .5, 1.), axis=(1,2))

    def test_minimum(self):
        x0 = self.s.minimum(self.x0, axis=(1, 2))
        nptest.assert_allclose(self.s.potential(x0, 0)[0], 0., atol=1e-5)
        nptest.assert_allclose(x0, [.01, .8125, 1.015], atol=1e-3)

    def test_saddle(self):
        xs, xsp = self.s.saddle(self.x0+[0,0,.1], axis=(1, 2))
        nptest.assert_almost_equal(xs, [.01, .883, 1.828], decimal=3)
        nptest.assert_almost_equal(xsp, .00421, decimal=4)
    
    def test_modes(self):
        """fails sometimes due to near degeneracy"""
        o, e = self.s.modes(self.x0)
        nptest.assert_almost_equal(o, [0, .1164, .1164], decimal=4)
        a = -86.89*np.pi/180
        nptest.assert_almost_equal(e[1:, 1:],
                [[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]], decimal=3)
    
    def test_mathieu(self):
        mu, b = self.s.mathieu(self.x0, 4*.018, 30*.018)
        nptest.assert_almost_equal(mu.real, 0., 9)

    def test_with(self):
        n = len(self.s.electrodes)
        dcs, rfs = np.arange(n), np.arange(n, 2*n)
        self.s.voltages = np.zeros((n, 2))
        nptest.assert_allclose(self.s.voltages, np.zeros((n, 2)))
        with self.s.with_voltages(dcs=dcs, rfs=rfs):
            nptest.assert_allclose(self.s.dcs, dcs)
            nptest.assert_allclose(self.s.rfs, rfs)
        nptest.assert_allclose(self.s.voltages, np.zeros((n, 2)))

    def test_shims_shift(self):
        x = self.x0
        eln = "c1 c2 c3 c4 c5 c6".split()
        s = system.System([self.s[n] for n in eln])
        derivs = "x y z xx yy yz".split()
        vectors = s.shims([(x, None, d) for d in derivs])
        self.assertEqual(vectors.shape, (len(derivs), len(eln)))
        for v, n in zip(vectors, derivs):
            d, e = utils.name_to_deriv(n)
            s.dcs = v
            p = s.electrical_potential(x, "dc", d)[0]
            #v = np.identity(2*d+1)[e]
            #nptest.assert_almost_equal(p, v)
            nptest.assert_allclose(p[e], 1, rtol=1e-5)

    def test_shims_shift_modes(self):
        x = self.x0
        eln = "c1 c2 c3 c4 c5 c6".split()
        els = [self.s.electrode(n) for n in eln]
        o0, e0 = self.s.modes(x)
        us, (res, rank, sing) = self.s.shims([x], els, curvatures=[[]],
                coords=[e0])
        for i, usi in enumerate(us.T):
            for ui, el in zip(usi, els):
                el.voltage_dc = ui
            g = self.s.gradient(x)[..., 0]
            g = np.dot(e0.T, g)
            nptest.assert_almost_equal(g[i], 1, decimal=3)

    def test_shims_curve(self):
        x = self.x0
        eln = "c1 c2 c3 c4 c5 c6".split()
        els = [self.s.electrode(n) for n in eln]
        o0, e0 = self.s.modes(x)
        us, (res, rank, sing) = self.s.shims([x], els,
                forces=[[0, 1, 2]],
                curvatures=[[0, 1, 4]],
                coords=None)
        self.assertEqual(us.shape, (len(eln), 6))
        self.assertEqual(rank, 6)
        c0 = utils.select_tensor(self.s.curvature(x))[:, 0]
        for i, usi in enumerate(us.T):
            for ui, el in zip(usi, els):
                el.voltage_dc = ui
            if i in (0, 1, 2): # force
                nptest.assert_almost_equal(self.s.gradient(x)[i, 0], 1,
                    decimal=2)
            if i in (3, 4, 5): # curve
                c1 = utils.select_tensor(self.s.curvature(x))[:, 0]
                ax = {3:0, 4:1, 5:4}[i]
                nptest.assert_almost_equal(c1[ax]-c0[ax], 1,
                    decimal=2)
                nptest.assert_almost_equal(self.s.gradient(x)[:, 0],
                        [0., 0, 0], decimal=5)
            if i == 0:
                nptest.assert_almost_equal(usi[(0, 3), :], -usi[(2, 5), :])
            elif i in (1, 2):
                nptest.assert_almost_equal(usi[(0, 3), :], usi[(2, 5), :])
            if i == 2:
                self.assertEqual(np.alltrue(usi > 0), True)

    def test_shims_build(self):
        x = self.x0
        eln = "c1 c2 c3 c4 c5 c6".split()
        els = [self.s.electrode(n) for n in eln]
        us, (res, rank, sing) = self.s.shims([x], els,
                forces=[[0, 1, 2]],
                curvatures=[[0, 1, 4]],
                coords=None)
        us = us.T
        u = .01*us[3]
        for ui, el in zip(u, els):
            el.voltage_dc = ui
        o0, e0 = self.s.modes(x)
        nptest.assert_almost_equal(o0, [.01, .095, .128], decimal=3)
        nptest.assert_almost_equal(self.s.potential_rf(x), 0.)
        mu, b = self.s.mathieu(x, 4*.018, 30*.018)
        nptest.assert_almost_equal(mu.real, 0., 9)

    def test_ions_simple(self):
        self.test_shims_build()
        x = self.x0
        n = 3
        xi = x+np.random.randn(n)[:, None]*1e-3
        qi = np.ones((n))*1e-3
        xis, ois, vis = self.s.ions(xi, qi)
        nptest.assert_almost_equal(
                xis[np.argmin(np.abs(xis)[:, 0])], x, 3)

    def test_ions_modes(self):
        """fails sometimes due to near degeneracy"""
        self.test_shims_build()
        x = self.x0
        n = 2
        xi = x+np.random.randn(n)[:, None]*1e-3
        qi = np.ones((n))*1e-3
        xis, ois, vis = self.s.ions(xi, qi)
        nptest.assert_almost_equal(
                ois, [.01, .03, .085, .095, .105, .128], 2)

    def test_analyze_static(self):
        s = list(self.s.analyze_static(self.x0))
        self.assertEqual(len(s), 23)

    def test_analyze_shims(self):
        s = list(self.s.analyze_shims([self.x0]))
        self.assertEqual(len(s), 14)


class RingtrapCase(unittest.TestCase):
    def ringtrap(self):
        s = system.System()
        n = 100
        p = np.exp(1j*np.linspace(0, 2*np.pi, 100))
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
            rf=1, paths=[np.r_[
                3.38*np.array([p.real, p.imag]).T,
                .68*np.array([p.real, p.imag]).T[::-1]]]
            ))
        return s

    def setUp(self):
        self.s = self.ringtrap()

    def test_minimum(self):
        x0 = self.s.minimum((0,0,1.), axis=(1, 2))
        nptest.assert_almost_equal(x0, [0, 0, 1.], decimal=3)

    def test_low_rf(self):
        p = self.s.potential((0,0,1.))
        nptest.assert_almost_equal(p, 0, decimal=3)

    def test_curve(self):
        x = self.s.minimum([0, 0, 1.])
        c0 = self.s.electrical_potential(x, "dc", 2)
        nptest.assert_allclose(c0, 0)
        c0 = self.s.electrical_potential(x, "rf", 2, expand=True)
        c1 = self.s.pseudo_potential(x, 2)
        c2 = self.s.potential(x, 2)
        nptest.assert_allclose(c1, c2)
        nptest.assert_allclose(2*np.dot(c0[0], c0[0]), c1[0], rtol=1e-3,
                atol=1e-9)

    def test_saddle(self):
        xs, xsp = self.s.saddle((0,.1,1.1), axis=(1, 2))
        nptest.assert_almost_equal(xs, [0, 0, 1.933], decimal=3)
        nptest.assert_almost_equal(xsp, .0196, decimal=4)
        nptest.assert_almost_equal(.1196, .0196*6.1, decimal=4)

    def test_modes(self):
        o0, e0 = self.s.modes([0, 0, 1])
        nptest.assert_almost_equal(o0, [.1114, .1114, .4491], decimal=3)
        abc = np.array(transformations.euler_from_matrix(e0))/2/np.pi
        nptest.assert_allclose(abc, 0, atol=1, rtol=1e-3)

    def test_plot(self):
        fig, ax = plt.subplots()
        self.s.plot(ax)

    def test_plot_voltages(self):
        fig, ax = plt.subplots()
        self.s.plot_voltages(ax)



if __name__ == "__main__":
    unittest.main()


