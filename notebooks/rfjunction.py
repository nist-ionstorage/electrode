# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

#     Copyright (C) 2011-2012 Robert Jordens <jordens@phys.ethz.ch>,
#                             Roman Schmied <roman.schmied@unibas.ch>
#     
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#    
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# 
# Demo of the `electrode` package
# ===============================
# 
# This demo is similar to the "SurfacePattern demo: forward - finite - electric" of Roman Schmied,
# see http://atom.physik.unibas.ch/people/romanschmied/code/SurfacePattern.php

# <codecell>

from numpy import *
from matplotlib import pyplot as plt
from scipy import constants
from electrode.transformations import euler_matrix
from electrode import System, CoverElectrode, PolygonPixelElectrode
from electrode.utils import shaped
    
set_printoptions(precision=2)

# <codecell>

def rfjunction(l1, l2, a1, a2, a3):
    a0 = 2*(2**.5-1)
    b0 = 2.
    rmax = 5000
    J = 6
    D = 1
    nmax = 0
    cover_height = 150
    infp = lambda t: matrix([cos(t)*rmax, sin(t)*rmax])
    rot = lambda t: matrix([[cos(t), -sin(t)], [sin(t), cos(t)]])
    rot3 = lambda p: reduce(lambda a,b: a+b, ([[i*rot(t) for i in q] for q in p] for t in
        (0., 2*pi/3, 4*pi/3)), [])
    els = []
    els.append(["r", rot3([
        [infp(3*pi/2), (a0/2+b0, -l2), (a0/2, -l1)],
        [infp(3*pi/2), (-a0/2, -l1), (-a0/2-b0, -l2)],
        [(a0/2, -l1), (a1*3**.5/2, a1*-.5),
            (.5*(a0/2+3**.5*l1), .5*(-3**.5*a0/2+l1)),
            (a2*3**.5/2, a2*-.5)],
        ])])
    sect = []
    sect.append(["r", [
        [[[a0/2, -l1]], [[a0/2+b0, -l2]], [[a1*3**.5/2, -.5*a1]]],
        [[[-a0/2, -l1]], [[-a1*3**.5/2, -.5*a1]], [[-a0/2-b0, -l2]]],
        [(a0/2, -l1)*rot(4*pi/3), (a2*3**.5/2, -.5*a2)*rot(4*pi/3),
        (-1+2**.5+3**.5*l1, 3**.5-6**.5+l1)*rot(4*pi/3)*.5,
        (3**.5/2, .5)*rot(4*pi/3)*a3, (0, 0), (0,
            -a3)*rot(4*pi/3)]]])
    sect.append(["1", [
        [(0, a1), (.25*(-a0-2*b0+2*3**.5*l2), .5*(3**.5*(a0/2+b0)+l2)),
            (-.25*(-a0-2*b0+2*3**.5*l2),
                .5*(3**.5*(a0/2+b0)+l2))]]])
    sect.append(["2", [
        [(a0/4, -(l1+a3)/2), (0, -a3), (-a0/4, -(l1+a3)/2)]]])
    sect.append(["3", [
        [(3*a0/8, -(3*l1+a3)/4), (a0/4, -(l1+a3)/2),
            (-a0/4, -(l1+a3)/2), (-3*a0/8, -(3*l1+a3)/4)]]])
    sect.append(["4", [
        [(a0/2, -l1), (3*a0/8, -(3*l1+a3)/4),
            (-3*a0/8, -(3*l1+a3)/4), (-a0/2, -l1)]]])
    for j in range(J):
        sect.append(["%s" % (j+5), [
            [(-a0/2, -(l1+j*D)), (-a0/2, -(l1+(j+1)*D)),
                (a0/2, -(l1+(j+1)*D)), (a0/2, -(l1+j*D))]]])
    sect.append(["%s" % (J+5), [
        [(-a0/2, -(l1+J*D)), infp(3*pi/2), (a0/2, -(l1+J*D))]]])
    for n, t in zip("abc", (0, 2*pi/3, 4*pi/3)):
        for m, p in sect:
            els.append([n+m, [[r*rot(t) for r in q] for q in p]])
    els = [(n, [array(p)[:,0,:] for p in el]) for n, el in els]
    s = System()
    for n, paths in els:
        s.append(PolygonPixelElectrode(name=n, paths=paths,
                cover_nmax=nmax, cover_height=cover_height))
    s.append(CoverElectrode(name="m",
        height=cover_height))
    return s

# <codecell>

s = rfjunction(l1=3.7321851829486046, l2=3.0921935283018334,
                a1=1.5968461727578884, a2=0.7563967699214798,
                a3=0.2630201367283588)
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim((-10,10))
ax.set_ylim((-11,6))
s.plot(ax, label="")

# <codecell>

# connect b rf, c rf and main rf
s["r"].rf = 1
s["br"].rf = 1
s["cr"].rf = 1

def channel(x):
    """return channel minimum at x=x"""
    x1 = s.minimum((x, tan(pi/6)*abs(x), 1.), axis=(1,))
    x0 = s.minimum(x1, axis=(1, 2))
    return x0

xc = array(map(channel, linspace(-8, 8, 50)))

plt.figure()
_ = plt.plot(xc[:, 0], xc[:, 2])

plt.figure()
_ = plt.plot(xc[:, 0], s.potential(xc))

plt.figure()
m = map(s.modes, xc)
om = array([mi[0] for mi in m])
mm = array([mi[1] for mi in m])
_ = plt.plot(xc[:, 0], om[:, 0], "r-", xc[:, 0], om[:, 1], "g-", xc[:, 0], om[:, 2], "b-")

# <codecell>

n = 100
r = 7
xyz = mgrid[-r:r:1j*n, -r:r:1j*n, 1:2]
p = shaped(s.potential)(xyz)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
_ = ax.contour(xyz[0].reshape((n,n)), xyz[1].reshape((n,n)),
           log(p).reshape((n,n)),
           20, cmap=plt.cm.hot)

# <codecell>

x0a10, x0b10, x0c10 = (channel(s[e].paths[0].mean(axis=0)[0]) for e in "a10 b10 c10".split())
x0 = x0b10
els = "b7 b8 b9 b10 b11".split()
s1 = System(s[i] for i in els)
us = s1.shims([(x0, None, i) for i in "x z xx xz".split()])

# add some yy curvature
s1.dcs = .02*us[2]

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim((-12,0))
ax.set_ylim((0,8))
s.plot_voltages(ax, label="")
ax.plot(x0[0], x0[1], "kx")

l = 40e-6
u = 27.
m = 24*constants.atomic_mass
q = 1*constants.elementary_charge
o = 2*pi*55e6
scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
dc_scale = scale/q # dc energy scale

for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
    print line
    
# undo yy curvature at x0
s1 = 0*us[3]

# <codecell>

x0 = channel(0.) # center of b-c channel
els = "a1 a2 a3 b1 b2 b3 b4 b5 c1 c2 c3 c4 c5".split()
s1 = System(s[i] for i in els)
us = s1.shims([(x0, None, i) for i in "x y z xx yy yz".split()])

# add some xx curvature
s1.dcs = .02*us[3]

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim((-6,6))
ax.set_ylim((-3,5))
s.plot_voltages(ax, label="")
ax.plot(x0[0], x0[1], "kx")

l = 40e-6
u = 27.
m = 24*constants.atomic_mass
q = 1*constants.elementary_charge
o = 2*pi*55e6
scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
dc_scale = scale/q # dc energy scale

for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
    print line
    
# undo yy curvature at x0
s1.dcs = 0*us[3]

# <codecell>


