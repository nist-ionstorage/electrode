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

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import os
import unittest
from numpy import testing as nptest

import numpy as np

from electrode import utils, electrode, system


class CoverCase(unittest.TestCase):
    def setUp(self):
        self.c = electrode.CoverElectrode(height=20)
        self.x = np.array([[1, 2, 3.]])

    def test_pot(self):
        nptest.assert_almost_equal(self.c.potential(
            self.x, 0), [[3/20.]])

    def test_grad(self):
        nptest.assert_almost_equal(self.c.potential(
            self.x, 1), [[0, 0, 1/20.]])

    def test_curve(self):
        nptest.assert_almost_equal(self.c.potential(
            self.x, 2).sum(), 0.)

    def test_orientation(self):
        nptest.assert_almost_equal(self.c.orientations(), [])


class LargeElectrodeCase(unittest.TestCase):
    def setUp(self):
        r = 1e9
        self.e = electrode.PolygonPixelElectrode(paths=[[
            [r, r], [-r, r], [-r, -r], [r, -r]]])
        self.x = np.array([[1, 2, 3.]])
    
    def test_pot(self):
        nptest.assert_almost_equal(self.e.potential(
            self.x, 0)[0], 1)

    def test_pot_cover(self):
        #need large numbers due to large electrode
        self.e.cover_nmax = 1e5
        self.e.cover_height = 1e6
        x = np.array([[0, 0, self.e.cover_height]])
        nptest.assert_allclose(self.e.potential(x, 0), 0, atol=1e-4)

    def test_z_symmetry(self):
        x = self.x*[[1, 1, 1,], [1, 1, -1]]
        for i, deriv in enumerate(utils.derivative_names):
            s = -(-1)**np.array([_.count("z") for _ in deriv])
            p = self.e.potential(x, i)
            nptest.assert_allclose(s*p[0], p[1])

    def test_grad(self):
        nptest.assert_allclose(self.e.potential(
            self.x, 1), 0, atol=1e-9)

    def test_curve(self):
        nptest.assert_allclose(self.e.potential(
            self.x, 2)[0], 0., atol=1e-9)

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])


class PixelElectrodeCase(unittest.TestCase):
    def setUp(self):
        self.r = r = 4e-5
        self.p = electrode.PolygonPixelElectrode(paths=[[
            [1+r, 2+r], [1-r, 2+r], [1-r, 2-r], [1+r, 2-r]]])
        a = (2*r)**2
        self.e = electrode.PointPixelElectrode(areas=[a], points=[[1, 2]])
        self.x = np.array([[3, 6, 5.]])

    def test_convert(self):
        c = self.p.to_points()
        nptest.assert_almost_equal(c.areas, self.e.areas, 13)
        nptest.assert_almost_equal(c.points, self.e.points)

    def test_pots(self):
        for di in range(6):
            a = self.p.potential(self.x, di)
            b = self.e.potential(self.x, di)
            nptest.assert_allclose(a, b, rtol=1e-4)

    def test_z_symmetry(self):
        x = self.x*[[1, 1, 1,], [1, 1, -1]]
        for i, deriv in enumerate(utils.derivative_names):
            s = -(-1)**np.array([_.count("z") for _ in deriv])
            p = self.e.potential(x, i)
            nptest.assert_allclose(s*p[0], p[1])
            q = self.p.potential(x, i)
            nptest.assert_allclose(s*q[0], q[1])

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])
        nptest.assert_almost_equal(self.p.orientations(), [1.])

    def test_derivs(self):
        ns = range(1, 6)
        for ee, d in (self.e, 1e-6), (self.p, 1e-6):
            xd = self.x + [[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d]]
            for n in ns:
                p = utils.expand_tensor(ee.potential(self.x, n))[0]
                pd = utils.expand_tensor(ee.potential(xd, n-1))
                pd = (pd[1:] - pd[0])/d
                nptest.assert_allclose(p, pd, rtol=d, atol=1e-15,
                      err_msg="ee=%s, n=%i" % (ee, n))


class PolygonTestCase(unittest.TestCase):
    def setUp(self):
        p = np.array([[1, 0], [2, 3], [2, 7], [3, 8],
            [-2, 8], [-5, 2]])
        self.e = electrode.PolygonPixelElectrode(paths=[p])
        self.x = np.array([[1, 2, 3.]])

    def test_orientation(self):
        nptest.assert_almost_equal(self.e.orientations(), [1.])

    def test_pot_cover(self):
        self.e.cover_nmax = 10
        self.e.cover_height = 50.
        x = np.array([[0, 0, self.e.cover_height]])
        # phi = 0
        nptest.assert_allclose(self.e.potential(x, 0), 0, atol=1e-5)
        # E_perp == 0
        nptest.assert_allclose(self.e.potential(x, 1)[:, :2], 0,
                atol=1e-5)

    def test_known_pot(self):
        nptest.assert_almost_equal(
                self.e.potential(self.x, 0)[0], .24907)

    def test_known_grad(self):
        nptest.assert_almost_equal(
                self.e.potential(self.x, 1)[0],
                [-0.0485227, 0.0404789, -0.076643])

    def test_known_curve(self):
        nptest.assert_almost_equal(
                self.e.potential(self.x, 2)[0],
                [-0.0196946, -0.00747322, 0.0287624, -0.014943, -0.0182706])

    def test_derivs(self):
        ns = range(1, 6)
        d = 1e-7
        xd = self.x + [[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d]]
        for n in ns:
            p = utils.expand_tensor(self.e.potential(self.x, n))[0]
            pd = utils.expand_tensor(self.e.potential(xd, n-1))
            pd = (pd[1:] - pd[0])/d
            for k, (i, j) in enumerate(zip(p.ravel(), pd.ravel())):
                nptest.assert_allclose(i, j, rtol=d, atol=d/100,
                      err_msg="n=%i, k=%i" % (n,k))
                
    def test_pseudopotential_derivs(self):
        ns = range(1, 5)
        d = 1e-7
        xd = self.x + [[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d]]
        s = system.System([self.e])
        self.e.rf = 1.
        for n in ns:
            p = s.potential(self.x, n)[0]
            pd = s.potential(xd, n-1)
            pd = (pd[1:] - pd[0])/d
            for k, (i, j) in enumerate(zip(p.ravel(), pd.ravel())):
                nptest.assert_allclose(i, j, rtol=d, atol=d/100,
                      err_msg="n=%i, k=%i" % (n,k))

    def test_spherical_harmonics(self):
        ns = range(6)
        v = [self.e.potential(self.x, i).T for i in ns]
        for i, vi in enumerate(v):
            s = utils.cartesian_to_spherical_harmonics(vi)
            self.assertEqual(s.shape, vi.shape)

    def test_z_symmetry(self):
        x = self.x*[[1, 1, 1,], [1, 1, -1]]
        for i, deriv in enumerate(utils.derivative_names):
            s = -(-1)**np.array([_.count("z") for _ in deriv])
            p = self.e.potential(x, i)
            nptest.assert_allclose(s*p[0], p[1])


class GridElectrodeCase(unittest.TestCase):
    def setUp(self):
        p = np.array([[1, 0], [2, 3], [2, 7], [3, 8],
            [-2, 8], [-5, 2]])
        self.p = electrode.PolygonPixelElectrode(paths=[p])
        spacing = .11, .13, .11
        origin = -1.5, -1.7, .9
        shape = 31, 32, 33
        x = np.mgrid[[slice(o, o+(s-.5)*d, d) for o, s, d in zip(origin,
            shape, spacing)]]
        xt = x.reshape(3, -1).T
        assert x.shape == (3,) + shape, x.shape
        pot0 = self.p.potential(x.reshape(3, -1).T,
                0).reshape(x[0].shape+(1,))
        pot1 = self.p.potential(x.reshape(3, -1).T,
                1).reshape(x[0].shape+(3,))
        self.e = electrode.GridElectrode(data=[pot0, pot1],
            origin=origin, spacing=spacing)

    def test_gen(self):
        for i in 0, 5:
            self.e.generate(maxderiv=i)

    def test_pot(self):
        self.e.generate(4)
        x = np.array([[1.4567, 1.67858, 1.49533]])
        for d, r in (0, 1e-3), (1, 2e-3), (2, 5e-3), (3, .1), (4, .5):
            pe = self.e.potential(x, d)
            pp = self.p.potential(x, d)
            nptest.assert_allclose(pe, pp, rtol=r, atol=1e-4)


class GridElectrodeVtkCase(unittest.TestCase):
    fil = "~/work/nist/qc-tools/trunk/bin/threefold_2_sim_dense_rf.vtk"
    fil = os.path.expanduser(fil)
    @unittest.skipUnless(os.path.exists(fil), "no dataset")
    def test_load_vtk(self):
        e = electrode.GridElectrode.from_vtk(os.path.expanduser(fil))


class MeshElectrodeCase(unittest.TestCase):
    def setUp(self):
        p = np.array([[0, 0], [1, 0], [1, 2], [2, 2.]])
        a = electrode.PolygonPixelElectrode(paths=[p, p+[[3, 3.]]],
                dc=1.)
        b = electrode.PolygonPixelElectrode(paths=[p-[[3, 3]]],
                dc=2.)
        self.s = system.System([a, b])
        self.m = electrode.MeshPixelElectrode.from_polygon_system(
            self.s)
        self.x = np.array([[1,2,3.]])

    def test_create(self):
        for di in range(6):
            a = self.s.electrical_potential(self.x, "dc", di)
            b = self.m.potential(self.x, di)
            nptest.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
