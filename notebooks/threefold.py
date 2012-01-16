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
# This demo is similar to the "SurfacePattern demo: backward - finite - electric" of Roman Schmied,
# see http://atom.physik.unibas.ch/people/romanschmied/code/SurfacePattern.php

# <codecell>

from numpy import *
from matplotlib import pyplot as plt
from scipy import constants
from electrode.transformations import euler_matrix
from electrode import (System,
        PointPixelElectrode, PolygonPixelElectrode,
        PatternValueConstraint, PatternRangeConstraint)
    
set_printoptions(precision=2)

# <markdowncell>

# Define two functions to build and optimize the pattern electrode:

# <codecell>

def hextess(n, points=False):
    """returns a hexagonal pixel electrode with unit radius
    and n pixels per unit length in a hexagonal pattern

    if points is True, each pixel is approximated as a point
    else each pixel is a hexagonal polygon"""
    x = vstack(array([[i+j*.5, j*3**.5*.5, 0]
        for j in range(-n-min(0, i), n-max(0, i)+1)])
        for i in range(-n, n+1))/(n+.5)
    if points:
        a = ones((len(x),))*3**.5/(n+.5)**2/2
        return PointPixelElectrode(points=x, areas=a)
    else:
        a = 1/(3**.5*(n+.5)) # edge length
        p = x[:, None, :] + [[[a*cos(phi), a*sin(phi), 0] for phi in
            arange(pi/6, 2*pi, pi/3)]]
        return PolygonPixelElectrode(paths=list(p))

# <codecell>

def threefold(n, h, d, H, nmax=1, points=True):
    """returns a System instance with a single hexagonal rf pixel electrode.

    The pixel factors (whether a pixel is grounded or at rf) are optimized
    to yield three trapping sites forming an equilateral triangle with

       n pixels per unit length,
       ion separation d,
       ion height h, and
       trapping frequencies with a ratio 2:1:1 (radial being the strongest).

    The effect of a grounded shielding cover at height H is accounted for
    up to nmax components in the expansion in h/H.
    """
    s = System()
    rf = hextess(n, points)
    rf.voltage_rf = 1.
    rf.name = "rf"
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
            v=2**(-1/3.)*eye(3)*[1, 1, -2]))
    rf.pixel_factors, c = rf.optimize(ct)
    return s, c

# <markdowncell>

# Create and optmimize the system:

# <codecell>

n=12
h=1/8.
d=1/4.
H=25/8.
x0 = array([d/3**.5, 0, h])
s, c = threefold(n, h, d, H, nmax=1, points=True)

# <markdowncell>

# Analysis of the result:

# <codecell>

 # optimal strength of the constraints
print "c*h**2*c:", h**2
# rf field should vanish
print "rf'/c:", s.electrode("rf").potential(x0, 1)[0][:, 0]/c
# rf curvature should be (2, 1, 1)/2**(1/3)
print "rf''/c:", s.electrode("rf").potential(x0, 2)[0][(0, 1, 2), (0, 1, 2), 0]/c

# <markdowncell>

# Plot the electrode pattern, white is ground, black/red is rf:

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_xlim((-1,1))
ax.set_ylim((-1,1))
s.plot_voltages(ax, u=array([1.]))

# <markdowncell>

# Get some physical quantities for a specific implementation:

# <codecell>

l = 320e-6 # length scale, hexagon radius
u = 20. # peak rf voltage
o = 2*pi*50e6 # rf frequency
m = 24*constants.atomic_mass # ion mass
q = 1*constants.elementary_charge # ion charge

for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
    print line

# <markdowncell>

# Plot the horizontal logarithmic pseudopotential at the ion height:

# <codecell>

n = 30
xyz = mgrid[-d:d:1j*n, -d:d:1j*n, h:h+1]
xyzt = xyz.transpose((1, 2, 3, 0)).reshape((-1, 3))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.contour(xyz[0].reshape((n,n)), xyz[1].reshape((n,n)),
           log(s.potential(xyzt)).reshape((n,n)),
           20, cmap=plt.cm.hot)

# <markdowncell>

# Plot the logarithmic pseudopotential in the xz plane.

# <codecell>

n = 30
xyz = mgrid[-d:d:1j*n, 0:1, .5*h:3*h:1j*n]
xyzt = xyz.transpose((1, 2, 3, 0)).reshape((-1, 3))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.contour(xyz[0].reshape((n,n)), xyz[2].reshape((n,n)),
           log(s.potential(xyzt)).reshape((n,n)),
           20, cmap=plt.cm.hot)

