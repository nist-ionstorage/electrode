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
from scipy import constants as ct

from electrode import (electrode, system, polygons, gds)


class PolygonsCase(unittest.TestCase):
    def ringtrap(self):
        s = system.System()
        n = 100
        p = np.exp(1j*np.linspace(0, 2*np.pi, 100))
        s.append(electrode.PolygonPixelElectrode(name="rf",
            rf=1, paths=[np.r_[
                3.38*np.array([p.real, p.imag]).T,
                .68*np.array([p.real, p.imag]).T[::-1]]]
            ))
        return s

    def setUp(self):
        self.s = self.ringtrap()
        self.p = polygons.Polygons.from_system(self.s)

    def test_create(self):
        pass

    def test_to_system(self, p=None):
        if p is None:
            p = self.p
        s1 = p.to_system()
        self.assertEqual(len(s1), len(self.s))
        for a, b in zip(s1, self.s):
            self.assertEqual(a.name, b.name)
            self.assertEqual(len(a.paths), len(b.paths))
            for c, d in zip(a.paths, b.paths):
                nptest.assert_allclose(c, d)

    def test_validate(self):
        self.p.validate()

    def test_remove_overlaps(self):
        p1 = self.p.remove_overlaps()
        self.test_to_system(p1)

    def test_add_gaps(self):
        p1 = self.p.add_gaps(0.)
        self.test_to_system(p1)

    def test_simplify(self):
        p1 = self.p.simplify()
        self.test_to_system(p1)
    
    def test_gaps_union(self):
        g = self.p.gaps_union()


class PolygonsReadGdsCase(unittest.TestCase):
    fil = "test.gds"
    @unittest.skipUnless(os.path.exists(fil), "no example gds")
    def test_read_boundaries_gaps(self):
        b, r = gds.from_gds_gaps(open(self.fil, "rb"), scale=100e-6)
        p = polygons.Polygons.from_boundaries_routes(b, r, edge=30.)
        names = "c0 m1 m2 m3 m4 m5 m6 rf a1 a2 p6 p5 p4 p3 p2 p1".split()
        pads = polygons.square_pads(step=.5, edge=30)
        for pad, poly in p.assign_to_pad(pads):
            p[poly] = names.pop(0), p[poly][1]
        assert not names
        s = p.to_system()
        assert len(s) == 16


if __name__ == "__main__":
    unittest.main()
