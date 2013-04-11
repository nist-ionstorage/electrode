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
import matplotlib.pyplot as plt

from electrode import electrode, system, polygons


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


fil1 = "test.gds"

@unittest.skipUnless(os.path.exists(fil1), "no example gds")
class GdsPolygonsCase(unittest.TestCase):
    def test_read_simple(self):
        p = polygons.Polygons.from_gds(open(fil1, "rb"),
                scale=100e-6)
        s = p.to_system()
        fig, ax = plt.subplots(figsize=(17, 11))
        s.plot(ax)
        #fig.savefig("gds_polygons.pdf")
        p = p.add_gaps(.01)
        p = p.simplify(1e-3)
        g = p.to_gds(scale=40e-6)
        #g.save(open("test2a.gds", "wb"))

    def test_read(self):
        p = polygons.Polygons.from_gds(open(fil1, "rb"), scale=100e-6,
                edge=30.)
        s = p.to_system()
        names = "c0 m1 m2 m3 m4 m5 m6 rf a1 a2 rf p6 p5 p4 p3 p2 p1".split()
        pads = polygons.square_pads(step=.5, edge=30.)
        for pad, poly in p.assign_to_pad(pads):
            s[poly].name = names.pop(0)
        assert not names
        fig, ax = plt.subplots(figsize=(17, 11))
        s.plot(ax)
        #fig.savefig("gds_polygons.pdf")
        p = p.add_gaps(.01)
        p = p.simplify(1e-3)
        g = p.to_gds(scale=40e-6)
        #g.save(open("test2a.gds", "wb"))

fil2 = "test2.gds"

@unittest.skipUnless(os.path.exists(fil2), "no example gds")
class GdsComplicatedCase(unittest.TestCase):
    def test_read_complicated(self):
        p = polygons.Polygons.from_gds(open(fil2, "rb"), scale=40e-6,
                poly_layers=[(0, 0), (13, 1), (13, 2)],
                route_layers=[(12, 2)], bridge_layers=[(13, 5)],
                gap_layers=[(1, 0)], edge=35.)
        names = ("m1 m2 m3 m4 m5 m6 m7 rf a1 a2 "
                 "rf p7 p6 p5 p4 p3 p2 p1 c1").split()
        pads = polygons.square_pads(step=.25, edge=35, odd=True)
        for pad, poly in p.assign_to_pad(pads):
            p[poly] = names.pop(0), p[poly][1]
        assert not names, names
        fig, ax = plt.subplots(figsize=(17, 11))
        p.to_system().plot(ax)
        #fig.savefig("gds_polygons.pdf")
        p = p.add_gaps(.01)
        p = p.simplify(1e-3)
        g = p.to_gds(scale=40e-6)
        #g.save(open("test2a.gds", "wb"))


if __name__ == "__main__":
    unittest.main()
