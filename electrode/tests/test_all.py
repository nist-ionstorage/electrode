#!/usr/bin/python
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

import unittest
from numpy import testing as nptest

import numpy as np
from numpy import (cos, sin, pi, tan, array, matrix, mgrid, dot, arange,
        log, linspace, arctan2, zeros, ones, arange, identity)
from scipy import constants as ct
import multiprocessing

import electrode
from electrode.transformations import euler_matrix, euler_from_matrix


class BasicFunctionsTestCase(unittest.TestCase):
    def test_dummy_pool(self):
        f = lambda x, y=1, *a, **k: (x, y, a, k)
        r = electrode.dummy_pool.apply_async(f, (2, 3, 4), {"a": 5})
        self.assertEqual(r.get(), (2, 3, (4,), {"a": 5}))

    def test_apply_method(self):
        class C:
            def m(self, a):
                return a
        self.assertEqual(electrode.apply_method(C(), "m", 1), 1)

    def test_norm(self):
        self.assertEqual(electrode.norm([1,2,3.]), 14**.5)
        self.assertEqual(electrode.norm([[1,2,3.]], 1), 14**.5)

    def test_dot(self):
        self.assertEqual(electrode.dot([1,2,3], [4,5,6]), 32)
        self.assertEqual(electrode.dot([[1,2,3]], [[4,5,6]], 1), 32)

    def test_triple(self):
        self.assertEqual(electrode.triple([1,2,3], [4,5,6], [7,8,10]),
                -3)
        self.assertEqual(electrode.triple([[1,2,3]], [[4,5,6]],
            [[7,8,10]], 1), -3)

    def test_expand_tensor(self):
        a = np.array([1, 2, 3.])[:, None]
        nptest.assert_equal(electrode.expand_tensor(a), a)
        b = np.array([1, 2, 3, 4, 5])[:, None]
        b1 = np.array([1, 2, 3, 2, 4, 5, 3, 5, -5] # triu
                )[:, None].reshape((3, 3, 1))
        nptest.assert_equal(electrode.expand_tensor(b), b1)
        c = np.random.random(5)[:, None]
        ti, tj = np.triu_indices(3)
        ce = electrode.expand_tensor(c)[ti, tj]
        nptest.assert_equal(ce[:5], c)
        nptest.assert_equal(ce[5], -c[0]-c[3])
    
    def test_expand_select_tensor(self):
        for n in 3, 5, 7:
            d = np.random.random(n)[:, None]
            de = electrode.expand_tensor(d)
            ds = electrode.select_tensor(de)
            nptest.assert_equal(d, ds)

    def test_expand_tensor_trace(self):
        d = np.random.random(5)[:, None]
        de = electrode.expand_tensor(d)
        nptest.assert_equal(de.trace(), 0)
        d = np.random.random(7)[:, None]
        de = electrode.expand_tensor(d)
        nptest.assert_almost_equal(de.trace(), np.zeros((3,1)))

    def test_rotate_tensor_identity(self):
        dr = np.identity(3)
        d = np.arange(3).reshape((3,))
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 1))
        d = np.arange(3**2).reshape((3,3))
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 2))
        d = np.arange(3**3).reshape(3,3,3)
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 3))
        d = np.arange(3**4).reshape(3,3,3,3)
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 4))
        d = np.arange(3**2*5).reshape(3,3,5)
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 2))
        d = np.arange(3**4*5).reshape(3,3,3,3,5)
        nptest.assert_almost_equal(d, electrode.rotate_tensor(d, dr, 4))
    
    def test_rotate_tensor_rot(self):
        r = euler_matrix(*np.random.random(3))[:3, :3]
        d = np.arange(3**3*5).reshape(3,3,3,5)
        dr = electrode.rotate_tensor(d, r, 3)
        drr = electrode.rotate_tensor(dr, r.T, 3)
        nptest.assert_almost_equal(d, drr)

    def test_rotate_tensor_simple(self):
        r = euler_matrix(0, 0, np.pi/2, "sxyz")[:3, :3]
        d = np.arange(3)
        nptest.assert_almost_equal(d[(1, 0, 2), :],
                electrode.rotate_tensor(d, r, 1))
        d = np.arange(9).reshape(3,3)
        nptest.assert_almost_equal([[4, -3, 5], [-1, 0, -2], [7, -6, 8]],
                electrode.rotate_tensor(d, r, 2))

    def test_centroid_area(self):
        p = np.array([[1, 0, 0], [2, 3, 0], [2, 7, 0], [3, 8, 0],
            [-2, 8, 0], [-5, 2, 0]])
        a, c = electrode.area_centroid(p)
        nptest.assert_almost_equal(a, 40)
        nptest.assert_almost_equal(c, [-1, 4, 0])

    def test_mathieu(self):
        a = np.array([.005])
        q = np.array([.2**.5])
        mu, b = electrode.mathieu(1, a, q)
        nptest.assert_almost_equal(mu.real, 0., 9)
        mui = sorted(mu.imag[mu.imag > 0])
        nptest.assert_almost_equal(mui[0], (a+q**2/2)**.5, 2)
        nptest.assert_almost_equal(mui[0], [.33786], 5)
        n = 3
        a = np.arange(n**2).reshape(n,n)
        q = np.arange(n**2)[::-1].reshape(n,n)*10
        mu, b = electrode.mathieu(3, a, q)
        #nptest.assert_almost_equal(mu, [.1, .2, .3])
        #nptest.assert_almost_equal(b, )


class CoverTestCase(unittest.TestCase):
    def setUp(self):
        self.c = electrode.CoverElectrode(voltage_dc=2,
                cover_height=20)

    def test_pot(self):
        nptest.assert_almost_equal(self.c.electrical_potential(
            [1, 2, 3]), 2*3/20.)

    def test_grad(self):
        nptest.assert_almost_equal(self.c.electrical_gradient(
            [1, 2, 3])[:, 0], [0, 0, 2/20.])

    def test_curve(self):
        nptest.assert_almost_equal(self.c.electrical_curvature(
            [1, 2, 3])[:, 0].sum(), 0.)

    def test_orientation(self):
        nptest.assert_almost_equal(self.c.orientations(), [1.])


class LargeElectrodeTestCase(unittest.TestCase):
    def setUp(self):
        r = 1e9
        self.e = electrode.PolygonPixelElectrode(paths=[[
            [r, r, 0], [-r, r, 0], [-r, -r, 0], [r, -r, 0]]],
            voltage_dc=2, voltage_rf=3)
    
    def test_null(self):
        self.e.voltage_dc = 0.
        nptest.assert_almost_equal(self.e.electrical_potential(
            [1, 2, 3]), 0)
        self.e.voltage_dc = 2

    def test_pot(self):
        nptest.assert_almost_equal(self.e.electrical_potential(
            [1, 2, 3]), 2)

    def test_pot_cover(self):
        self.e.nmax = 3
        self.e.cover_height = 100.
        nptest.assert_almost_equal(self.e.electrical_potential(
            [1, 2, 3]), 2)
        nptest.assert_almost_equal(self.e.electrical_potential(
            [1, 2, -3]), -2)

    def test_z_symmetry(self):
        for i, s in enumerate([-1, (-1, -1, 1), (-1, -1, 1, -1, 1),
                (-1, 1, 1, -1, -1, -1, 1)]):
            a = self.e.value([1, 2, -3], i)[0].T
            b = self.e.value([1, 2, 3], i)[0].T
            nptest.assert_almost_equal(s*a, b)
      
    def test_grad(self):
        nptest.assert_almost_equal(self.e.electrical_gradient(
            [1, 2, 3])[:, 0], [0, 0, 0.])

    def test_curve(self):
        nptest.assert_almost_equal(self.e.electrical_curvature(
            [1, 2, 3])[:, 0].sum(), 0.)

    def test_rf_null(self):
        self.e.voltage_rf = 0.
        nptest.assert_almost_equal(self.e.pseudo_potential(
            [1, 2, 3]), 0)
        self.e.voltage_rf = 3

    def test_rf_pot(self):
        nptest.assert_almost_equal(self.e.pseudo_potential(
            [1, 2, 3]), 0)

    def test_rf_grad(self):
        nptest.assert_almost_equal(self.e.pseudo_gradient(
            [1, 2, 3])[:, 0], [0, 0, 0.])

    def test_rf_curve(self):
        nptest.assert_almost_equal(self.e.pseudo_curvature(
            [1, 2, 3])[:, 0].sum(), 0.)

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])


class PixelElectrodeTestCase(unittest.TestCase):
    def setUp(self):
        self.r = r = 4e-5
        self.p = electrode.PolygonPixelElectrode(paths=[[
            [1+r, 2+r, 0], [1-r, 2+r, 0], [1-r, 2-r, 0], [1+r, 2-r, 0]]],
            voltage_dc=2e6/r, voltage_rf=3e6/r)
        a = (2*r)**2
        self.e = electrode.PointPixelElectrode(areas=[a], points=[[1, 2,
            0]], voltage_dc=2e6/r, voltage_rf=3e6/r)

    def test_convert(self):
        c = self.p.to_points()
        nptest.assert_almost_equal(c.areas, self.e.areas, 13)
        nptest.assert_almost_equal(c.points, self.e.points)

    def test_null(self):
        self.e.voltage_dc, self.e.voltage_rf = 0, 0
        self.p.voltage_dc, self.p.voltage_rf = 0, 0
        self.test_pots()

    def test_pots(self):
        p = "electrical pseudo".split()
        n = "potential gradient curvature".split()
        for ni in n:
            for pi in p:
                a = getattr(self.p, pi+"_"+ni)([1,2,3])
                b = getattr(self.e, pi+"_"+ni)([1,2,3])
                nptest.assert_almost_equal(a, b, decimal=5)

    def test_z_symmetry(self):
        for i, s in enumerate([-1, (-1, -1, 1), (-1, -1, 1, -1, 1),
                (-1, 1, 1, -1, -1, -1, 1)]):
            a = self.e.value([1, 2, -3], i)[0].T
            b = self.e.value([1, 2, 3], i)[0].T
            nptest.assert_almost_equal(s*a, b)
 
    def test_bare_pots(self):
        p, e = self.p.potential([1,2,3], 0, 1, 2, 3),\
                self.e.potential([1,2,3], 0, 1, 2, 3)
        for pi, ei in zip(p, e):
            nptest.assert_almost_equal(pi, ei)

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])
        nptest.assert_almost_equal(self.p.orientations(), [1.])


class PolygonTestCase(unittest.TestCase):
    def setUp(self):
        p = np.array([[1, 0, 0], [2, 3, 0], [2, 7, 0], [3, 8, 0],
            [-2, 8, 0], [-5, 2, 0]])
        self.e = electrode.PolygonPixelElectrode(paths=[p], 
                voltage_dc=1, voltage_rf=1)

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])

    def test_simple_pot(self):
        for zi in 3, -3:
            nptest.assert_almost_equal(
                    self.e.electrical_potential([1, 2, zi]),
                    electrode.potential(np.array([1, 2, zi])[None,:],
                        self.e.paths[0][::-1]))
         
    def test_simple_field(self):
        for zi in 3, -3:
            nptest.assert_almost_equal(
                    self.e.electrical_gradient([1, 2, zi]),
                    electrode.field([1, 2, zi],
                        self.e.paths[0][::-1]).T)

    def test_known_pot(self):
        nptest.assert_almost_equal(
                self.e.electrical_potential([1,2,3]),
                .24907)

    def test_known_grad(self):
        nptest.assert_almost_equal(
                self.e.electrical_gradient([1,2,3])[:, 0],
                [-0.0485227, 0.0404789, -0.076643])

    def test_known_curve(self):
        nptest.assert_almost_equal(
                electrode.select_tensor(
                    self.e.electrical_curvature([1,2,3]))[:, 0],
                [-0.0196946, -0.00747322, 0.0287624, -0.014943, -0.0182706])

    def test_known_curve_direct(self):
        nptest.assert_almost_equal(
                self.e.value([1,2,3], 2)[0][:, 0, 0],
                [-0.0196946, -0.00747322, 0.0287624, -0.014943, -0.0182706])


class ThreefoldOptimizeTestCase(unittest.TestCase):
    def hextess(self, n, points=False):
        x = array(sum(([array([i+j*.5, j*3**.5*.5, 0])/(n+.5)
            for j in range(-n-min(0, i), n-max(0, i)+1)]
            for i in range(-n, n+1)), []))
        if points:
            a = ones((len(x),))*3**.5/(n+.5)**2/2
            return electrode.PointPixelElectrode(points=x, areas=a)
        else:
            a = 1/(3**.5*(n+.5)) # edge length
            p = x[:, None, :] + [[[a*cos(phi), a*sin(phi), 0] for phi in
                arange(pi/6, 2*pi, pi/3)]]
            return electrode.PolygonPixelElectrode(paths=list(p))

    def setUp(self, n=12, h=1/8., d=1/4., H=25/8., nmax=1, points=True):
        rf = self.hextess(n, points)
        rf.voltage_rf = 1.
        rf.cover_height = H
        rf.nmax = nmax
        self.rf = rf

        ct = []
        ct.append(electrode.PatternRangeConstraint(min=0, max=1.))
        for p in 0, 4*pi/3, 2*pi/3:
            x = array([d/3**.5*cos(p), d/3**.5*sin(p), h])
            r = euler_matrix(p, pi/2, pi/4, "rzyz")[:3, :3]
            ct.append(electrode.PatternValueConstraint(d=1, x=x, r=r,
                v=[0, 0, 0]))
            ct.append(electrode.PatternValueConstraint(d=2, x=x, r=r,
                v=2**(-1/3.)*np.eye(3)*[1, 1, -2]))
        rf.pixel_factors, self.c = rf.optimize(ct, verbose=False)
        self.h = h
        
        self.x0 = array([d/3**.5, 0, h])
        self.r = euler_matrix(0, pi/2, pi/4, "rzyz")[:3, :3]

    def test_factor(self):
        nptest.assert_almost_equal(self.c*self.h**2, .159853, decimal=2)

    def test_potential(self):
        nptest.assert_almost_equal(
                self.rf.potential(self.x0, 1)[0][:, 0], [0, 0, 0])

    def test_curve(self):
        c = electrode.rotate_tensor(self.rf.potential(self.x0,
            2)[0]/self.c, self.r)
        nptest.assert_almost_equal(c[:, :, 0],
            2**(-1/3.)*np.eye(3)*[1, 1, -2])

    def test_poly_potential(self):
        self.setUp(points=False)
        nptest.assert_almost_equal(
                self.rf.potential(self.x0, 1)[0][:, 0], [0, 0, 0])
        nptest.assert_almost_equal(self.c*self.h**2, .13943, decimal=4)

    def test_main_saddle(self):
        s = electrode.System(electrodes=[self.rf])
        xs, xsp = s.saddle((0, 0, .5), axis=(0, 1, 2,))
        nptest.assert_almost_equal(xs, [0, 0, .5501], decimal=4)
        nptest.assert_almost_equal(xsp, .1662, decimal=4)

    def test_single_saddle(self):
        s = electrode.System(electrodes=[self.rf])
        xs, xsp = s.saddle(self.x0+[.02, 0, .02], axis=(0, 2), dx_max=.02)
        nptest.assert_almost_equal(xs, [.145, 0, .156], decimal=3)
        nptest.assert_almost_equal(xsp, .0109, decimal=3)
        xs1, xsp1 = s.saddle(self.x0+[.0, 0, .02], axis=(0, 1, 2), dx_max=.02)
        nptest.assert_almost_equal(xs, xs1, decimal=3)
        nptest.assert_almost_equal(xsp, xsp1, decimal=3)


class FourWireTestCase(unittest.TestCase):
    def simpletrap(self):
        s = electrode.System()
        rmax = 1e3
        def patches(n, tw, t0):
            for i in range(n):
                a = t0 - tw/2 + 2*np.pi/n*i
                ya, yb = np.tan(a/2), np.tan((a+tw)/2)
                yield np.array([[rmax, ya, 0], [rmax, yb, 0],
                     [-rmax, yb, 0], [-rmax, ya, 0]])
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
            voltage_rf=1, paths=list(patches(n=2, tw=np.pi/4,
                t0=5*np.pi/8))))
        return s

    def setUp(self):
        self.s = self.simpletrap()

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
        abc = np.array(euler_from_matrix(e0))/2/pi
        nptest.assert_allclose(abc, [0, 0, 0], atol=1, rtol=1e-3)

    def test_parallel(self):
        n = 10
        xyz = np.mgrid[-1:1:1j*n, -1:1:1j*n, .5:1.5:1j*n]
        r = self.s.parallel(electrode.dummy_pool, *xyz)

    def test_parallel_pool(self):
        import multiprocessing
        pool = multiprocessing.Pool()
        n = 10
        xyz = np.mgrid[-1:1:1j*n, -1:1:1j*n, .5:1.5:1j*n]
        r = self.s.parallel(pool, *xyz)

    def test_integrate(self):
        h = 1.
        x0 = np.array([0, 0, 1.])
        dt, vu0, uu = .01, .01, .5
        self.s.electrode("rf").voltage_rf = uu*h**2
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

        self.assertEqual(np.alltrue(electrode.norm(x, axis=1)<3), True)
        self.assertEqual(np.alltrue(electrode.norm(v, axis=1)<1), True)

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


class MagtrapTestCase(unittest.TestCase):
    def magtrap(self):
        s = electrode.System()
        rmax = 1e3
        a, b, c, d, e = -1.3, 1.3, .78, 2.5, 3.
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
            voltage_rf=1., paths=[
               [[rmax,0,0], [-rmax,0,0], [-rmax,a,0], [rmax,a,0]],
               [[rmax,b,0], [rmax,b+c,0], [-rmax,b+c,0], [-rmax,b,0]],
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
                        [cx+w/2,cy,0], [cx-w/2,cy,0],
                        [cx-w/2,rmax*np.sign(cy),0],
                        [cx+w/2,rmax*np.sign(cy),0]
                    ]]))
        for e in s.electrodes:
            for p in e.paths:
                if e.orientations()[0] == -1:
                    e.paths[0] = e.paths[0][::-1]
        return s

    def setUp(self):
        self.s = self.magtrap()
        self.x0 = self.s.minimum((0, .5, 1.), axis=(1,2))

    def test_minimum(self):
        x0 = self.s.minimum(self.x0, axis=(1, 2))
        nptest.assert_almost_equal(self.s.potential(x0), 0.)
        nptest.assert_almost_equal(x0, [0, .8125, 1.015], decimal=3)

    def test_saddle(self):
        xs, xsp = self.s.saddle(self.x0+[0,0,.1], axis=(1, 2))
        nptest.assert_almost_equal(xs, [0, .883, 1.828], decimal=3)
        nptest.assert_almost_equal(xsp, .00421, decimal=4)
    
    def test_modes(self):
        o, e = self.s.modes(self.x0)
        nptest.assert_almost_equal(o, [0, .1164, .1164], decimal=4)
        a = -116.7*np.pi/180
        nptest.assert_almost_equal(e[1:, 1:],
                [[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]], decimal=3)
    
    def test_mathieu(self):
        mu, b = self.s.mathieu(self.x0, 4*.018, 30*.018)
        nptest.assert_almost_equal(mu.real, 0., 9)

    def test_shims_shift(self):
        x = self.x0
        eln = "c1 c2 c3 c4 c5 c6".split()
        els = [self.s.electrode(n) for n in eln]
        us, (res, rank, sing) = self.s.shims([x] , els, curvatures=[[]])
        self.assertEqual(us.shape, (len(eln), 3))
        self.assertEqual(rank, 3)
        c0 = self.s.curvature(x)[..., 0]
        for i, usi in enumerate(us.T):
            for ui, el in zip(usi, els):
                el.voltage_dc = ui
            nptest.assert_almost_equal(self.s.gradient(x)[i, 0], 1,
                    decimal=2)
            if i == 0:
                nptest.assert_almost_equal(usi[(0, 3), :], -usi[(2, 5), :])
            elif i in (1, 2):
                nptest.assert_almost_equal(usi[(0, 3), :], usi[(2, 5), :])
            if i == 2:
                self.assertEqual(np.alltrue(usi > 0), True)
                nptest.assert_almost_equal(usi[:3]/usi[3:], 1,
                        decimal=1)

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
        c0 = electrode.select_tensor(self.s.curvature(x))[:, 0]
        for i, usi in enumerate(us.T):
            for ui, el in zip(usi, els):
                el.voltage_dc = ui
            if i in (0, 1, 2): # force
                nptest.assert_almost_equal(self.s.gradient(x)[i, 0], 1,
                    decimal=2)
            if i in (3, 4, 5): # curve
                c1 = electrode.select_tensor(self.s.curvature(x))[:, 0]
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


class RingtrapTestCase(unittest.TestCase):
    def ringtrap(self):
        s = electrode.System()
        n = 100
        p = np.exp(1j*np.linspace(0, 2*np.pi, 100))
        s.electrodes.append(electrode.PolygonPixelElectrode(name="rf",
            voltage_rf=1, paths=[np.r_[
                3.38*np.array([p.real, p.imag, 0*p.real]).T,
                .68*np.array([p.real, p.imag, 0*p.real]).T[::-1]]]
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

    def test_saddle(self):
        xs, xsp = self.s.saddle((0,.1,1.1), axis=(1, 2))
        nptest.assert_almost_equal(xs, [0, 0, 1.933], decimal=3)
        nptest.assert_almost_equal(xsp, .0196, decimal=4)
        nptest.assert_almost_equal(.1196, .0196*6.1, decimal=4)

    def test_modes(self):
        o0, e0 = self.s.modes([0, 0, 1])
        nptest.assert_almost_equal(o0, [.1114, .1114, .4491], decimal=3)
        abc = np.array(euler_from_matrix(e0))/2/pi
        nptest.assert_allclose(abc, [0, 0, 0], atol=1, rtol=1e-3)

        
if __name__ == "__main__":
    unittest.main()
