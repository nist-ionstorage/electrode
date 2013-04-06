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

import unittest
from numpy import testing as nptest

import numpy as np
from scipy import constants as ct

from electrode import (electrode, system, polygons)


class PolygonsCase(unittest.TestCase):
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
        self.p = polygons.Polygons.from_system(self.s)

    def test_create(self):
        pass

    def test_to_system(self, p=None):
        if p is None:
            p = self.p
        s1 = p.to_system()
        self.assertEqual(len(s1.electrodes), len(self.s.electrodes))
        for a, b in zip(s1.electrodes, self.s.electrodes):
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
        p1 = self.p.add_gaps(0)
        self.test_to_system(p1)

    def test_simplify(self):
        p1 = self.p.simplify()
        self.test_to_system(p1)
    
    def test_gaps_union(self):
        g = self.p.gaps_union()


if __name__ == "__main__":
    unittest.main()
