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

import numpy as np
from numpy import (cos, sin, pi, array, ones, mgrid, log, arange)
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import multiprocessing
from scipy import constants

from transformations import euler_matrix
from electrode import (System,
        PointPixelElectrode, PolygonPixelElectrode,
        PatternValueConstraint, PatternRangeConstraint)


def hextess(n, points=False):
    x = array(sum(([array([i+j*.5, j*3**.5*.5, 0])/(n+.5)
        for j in range(-n-min(0, i), n-max(0, i)+1)]
        for i in range(-n, n+1)), []))
    if points:
        a = ones((len(x),))*3**.5/(n+.5)**2/2
        return PointPixelElectrode(points=x, areas=a)
    else:
        a = 1/(3**.5*(n+.5)) # edge length
        p = x[:, None, :] + [[[a*cos(phi), a*sin(phi), 0] for phi in
            arange(pi/6, 2*pi, pi/3)[::-1]]]
        return PolygonPixelElectrode(paths=list(p))

def threefold(n=12, h=1/8., d=1/4., H=25/8., nmax=1, points=True):
    pool = multiprocessing.Pool()

    s = System()
    rf = hextess(n, points)
    rf.voltage_rf = 1.
    rf.cover_height = H
    rf.nmax = nmax
    s.electrodes.append(rf)

    ct = []
    ct.append(PatternRangeConstraint(min=0, max=1.))
    for p in 0, 4*pi/3, 2*pi/3:
        x = array([d/3**.5*cos(p), d/3**.5*sin(p), h])
        r = euler_matrix(p, pi/2, pi/4, "rzyz")[:3, :3]
        ct.append(PatternValueConstraint(d=1, x=x, r=r,
            v=[0, 0, 0]))
        ct.append(PatternValueConstraint(d=2, x=x, r=r,
            v=2**(-1/3.)*np.eye(3)*[1, 1, -2]))
    rf.pixel_factors, c = rf.optimize(ct)

    x0 = array([d/3**.5, 0, h])
    print "c*h**2", c*h**2
    print "rf'", rf.potential(x0, 1)[0][:, 0]
    print "rf''", rf.potential(x0, 2)[0][(0, 1, 2), (0, 1, 2), 0]/c
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    s.plot_voltages(ax, u=array([1.]))
    fig.savefig("threefold_ele.pdf")

    l = 320e-6
    u = 20.
    m = 24*constants.atomic_mass
    q = 1*constants.elementary_charge
    o = 2*np.pi*50e6

    for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
	print line

    n = 20
    xyz = mgrid[-d:d:1j*n, -d:d:1j*n, h:h+1]
    xyzt = xyz.transpose((1, 2, 3, 0)).reshape((-1, 3))
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.contour(xyz[0].reshape((n,n)), xyz[1].reshape((n,n)),
               log(s.parallel(pool, *xyz)).reshape((n,n)),
               20, cmap=pl.cm.hot)
    fig.savefig("threefold.pdf")


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    threefold()
