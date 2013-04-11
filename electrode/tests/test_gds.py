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

from electrode import electrode, system, gds, polygons


fil1 = "test.gds"

@unittest.skipUnless(os.path.exists(fil1), "no example gds")
class GdsPolygonsCase(unittest.TestCase):
    def test_read_simple(self):
        p = gds.GdsPolygons.from_gds_simple(open(fil1, "rb"),
                scale=100e-6)
        s = p.to_system()
        fig, ax = plt.subplots(figsize=(17, 11))
        s.plot(ax)
        #fig.savefig("gds_polygons.pdf")

    def test_read(self):
        p = gds.GdsPolygons.from_gds(open(fil1, "rb"), scale=100e-6,
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


fil2 = "test2.gds"

@unittest.skipUnless(os.path.exists(fil2), "no example gds")
class GdsComplicatedCase(unittest.TestCase):
    def test_read_complicated(self):
        p = gds.GdsPolygons.from_gds(open(fil2, "rb"), scale=40e-6,
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


if __name__ == "__main__":
    unittest.main()
