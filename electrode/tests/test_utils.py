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

from electrode import transformations, utils, electrode


class BasicFunctionsCase(unittest.TestCase):
    def test_dummy_pool(self):
        f = lambda x, y=1, *a, **k: (x, y, a, k)
        r = utils.DummyPool().apply_async(f, (2, 3, 4), {"a": 5})
        self.assertEqual(r.get(), (2, 3, (4,), {"a": 5}))

    def test_apply_method(self):
        class C:
            def m(self, a):
                return a
        self.assertEqual(utils.apply_method(C(), "m", 1), 1)

    def test_norm(self):
        self.assertEqual(utils.norm([1,2,3.]), 14**.5)
        self.assertEqual(utils.norm([[1,2,3.]], 1), 14**.5)

    def test_expand_tensor(self):
        a = np.array([1, 2, 3.])[None, :]
        nptest.assert_equal(utils.expand_tensor(a), a)
        b = np.array([1, 2, 3, 4, 5])[None, :]
        b1 = np.array([1, 2, 3, 2, 4, 5, 3, 5, -5] # triu
                ).reshape((1, 3, 3))
        nptest.assert_equal(utils.expand_tensor(b), b1)
        c = np.random.random(5)
        ti, tj = np.triu_indices(3)
        ce = utils.expand_tensor(c[None, :])[0, ti, tj]
        nptest.assert_equal(ce[:5], c)
        nptest.assert_equal(ce[5], -c[0]-c[3])
    
    def test_expand_select_tensor(self):
        for n in 3, 5, 7, 9, 11:
            d = np.random.random(n)[None, :]
            de = utils.expand_tensor(d)
            ds = utils.select_tensor(de)
            nptest.assert_equal(d, ds)

    def test_expand_tensor_trace(self):
        d = np.random.random(5)[None, :]
        de = utils.expand_tensor(d)
        nptest.assert_equal(de[0].trace(), 0)
        d = np.random.random(7)[None, :]
        de = utils.expand_tensor(d)
        nptest.assert_almost_equal(de[0].trace(), np.zeros((3)))
        d = np.random.random(9)[None, :]
        de = utils.expand_tensor(d)
        nptest.assert_almost_equal(de[0].trace(), np.zeros((3,3)))
        d = np.random.random(11)[None, :]
        de = utils.expand_tensor(d)
        nptest.assert_almost_equal(de[0].trace(), np.zeros((3,3,3)))

    def test_rotate_tensor_identity(self):
        dr = np.identity(3)
        d = np.arange(3).reshape((1,3,))
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 1))
        d = np.arange(3**2).reshape((1,3,3))
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 2))
        d = np.arange(3**3).reshape(1,3,3,3)
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 3))
        d = np.arange(3**4).reshape(1,3,3,3,3)
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 4))
        d = np.arange(3**2*5).reshape(5,3,3)
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 2))
        d = np.arange(3**4*5).reshape(5,3,3,3,3)
        nptest.assert_almost_equal(d, utils.rotate_tensor(d, dr, 4))
    
    def test_rotate_tensor_rot(self):
        r = transformations.euler_matrix(*np.random.random(3))[:3, :3]
        d = np.arange(3**3*5).reshape(5,3,3,3)
        dr = utils.rotate_tensor(d, r, 3)
        drr = utils.rotate_tensor(dr, r.T, 3)
        nptest.assert_almost_equal(d, drr)

    def test_rotate_tensor_simple(self):
        r = transformations.euler_matrix(0, 0, np.pi/2, "sxyz")[:3, :3]
        d = np.arange(3)
        nptest.assert_almost_equal(d[(1, 0, 2), :],
                utils.rotate_tensor(d, r, 1))
        d = np.arange(9).reshape(1,3,3)
        nptest.assert_almost_equal([[[4, -3, 5], [-1, 0, -2], [7, -6, 8]]],
                utils.rotate_tensor(d, r, 2))

    def test_centroid_area(self):
        p = np.array([[1, 0, 0], [2, 3, 0], [2, 7, 0], [3, 8, 0],
            [-2, 8, 0], [-5, 2, 0]])
        a, c = utils.area_centroid(p)
        nptest.assert_almost_equal(a, 40)
        nptest.assert_almost_equal(c, [-1, 4, 0])

    def test_mathieu(self):
        a = np.array([.005])
        q = np.array([.2**.5])
        mu, b = utils.mathieu(1, a, q)
        nptest.assert_almost_equal(mu.real, 0., 9)
        mui = sorted(mu.imag[mu.imag > 0])
        nptest.assert_almost_equal(mui[0], (a+q**2/2)**.5, 2)
        nptest.assert_almost_equal(mui[0], [.33786], 5)
        n = 3
        a = np.arange(n**2).reshape(n,n)
        q = np.arange(n**2)[::-1].reshape(n,n)*10
        mu, b = utils.mathieu(3, a, q)
        #nptest.assert_almost_equal(mu, [.1, .2, .3])
        #nptest.assert_almost_equal(b, )

    def test_polygon_value(self):
        p = np.array([[1., 0], [2, 3], [2, 7], [3, 8],
            [-2, 8], [-5, 2]])
        x = np.array([[1,2,3.]])
        nptest.assert_almost_equal(
                electrode.polygon_potential(x, [p], 1, 0, 0, 0, None),
                [[.24907]])
    
    def test_polygon_value_grad(self):
        p = np.array([[1., 0], [2, 3], [2, 7], [3, 8],
            [-2, 8], [-5, 2]])
        x = np.array([[1,2,3.]])
        nptest.assert_almost_equal(
                electrode.polygon_potential(x, [p], 1, 1, 0, 0, None),
                [[-0.0485227, 0.0404789, -0.076643]])


if __name__ == "__main__":
    unittest.main()
