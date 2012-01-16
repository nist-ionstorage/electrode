# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

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
from electrode import (System, CoverElectrode,
        PolygonPixelElectrode)
    
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
    rot3 = lambda p: __builtins__.sum(([[i*rot(t) for i in q] for q in p] for t in
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
    els = [(n, [c_[array(p)[:,0,:], zeros((len(p),))]
            for p in el]) for n, el in els]
    s = System()
    # connect a rf, c rf and main rf already here
    for n, paths in els:
        if n in ("ar", "cr", "r"):
            s.electrodes.append(PolygonPixelElectrode(name=n, paths=paths,
                voltage_rf=1., nmax=nmax, cover_height=cover_height))
        elif n.endswith("r"):
            s.electrodes.append(PolygonPixelElectrode(name=n, paths=paths,
                nmax=nmax, cover_height=cover_height))
        else:
            s.electrodes.append(PolygonPixelElectrode(name=n, paths=paths,
                nmax=nmax, cover_height=cover_height))
    # join c7-11, b7-11 to a7-11
    #for i in range(7, 12):
    #    for a in "cb":
    #        el = s.electrode("%s%i" % (a, i))
    #        s.electrode("a%i" % i).paths.extend(el.paths)
    #        s.electrodes.remove(el)
    s.electrodes.append(CoverElectrode(name="m",
        cover_height=cover_height))
    return s

def channel(y):
    """return channel minimum at y=y"""
    x1 = s.minimum((0., y, 1.), axis=(0,))
    x0 = s.minimum(x1, axis=(0, 2))
    return x0

# <codecell>

s = rfjunction(l1=3.7321851829486046, l2=3.0921935283018334,
                a1=1.5968461727578884, a2=0.7563967699214798,
                a3=0.2630201367283588)

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_xlim((-10,10))
ax.set_ylim((-11,6))
s.plot(ax)

# <codecell>

coord = array([
    [.75**.5, .25**.5, 0], [-.25**.5, .75**.5, 0], [0, 0, 1]])
x0ac = dot(coord, s.minimum((.65, 0, .9), axis=(0, 2),
    coord=coord)) # ac channel center
x0a10 = channel(s.electrode("a10").paths[0].mean(axis=0)[1])
x0a4 = channel(s.electrode("a4").paths[0].mean(axis=0)[1])
x0a6 = channel(s.electrode("a6").paths[0].mean(axis=0)[1])
x0a5 = channel(s.electrode("a5").paths[0].mean(axis=0)[1])
x0a8 = channel(s.electrode("a8").paths[0].mean(axis=0)[1])
x0a9 = channel(s.electrode("a9").paths[0].mean(axis=0)[1])

# <codecell>

xc = array(map(channel, linspace(x0a10[1], x0ac[1], 50)))

# <codecell>

plt.plot(xc[:, 1], xc[:, 2])

# <codecell>

plt.plot(xc[:, 1], s.potential(xc))

# <codecell>

x0 = x0a10
els = "a7 a8 a9 a10 a11".split()
for line in s.analyze_shims([x0], electrodes=els, use_modes=False,
    forces=["y z".split()], curvatures=["yy yz".split()]):
    print line
forces, curvatures, us = line
# add some yy curvature
for eli, ui in zip(els, us[:, 2]):
    s.electrode(eli).voltage_dc = .02*ui

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_xlim((-8,8))
ax.set_ylim((-13,8))
s.plot_voltages(ax)

# <codecell>

l = 40e-6
u = 27.
m = 24*constants.atomic_mass
q = 1*constants.elementary_charge
o = 2*pi*55e6
scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
dc_scale = scale/q # dc energy scale

for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
    print line

