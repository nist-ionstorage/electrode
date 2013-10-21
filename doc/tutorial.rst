Tutorial
========

.. ipython::

    In [1]: import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct

    In [1]: from electrode import (System, PolygonPixelElectrode, euler_matrix,
       ...:  PointPixelElectrode, PotentialObjective, PatternRangeConstraint, shaped)

    In [1]: np.set_printoptions(precision=2) # have numpy print fewer digits

Define a function that returns a hexagonal pixel electrode with unit radius
and n pixels per unit length in a hexagonal pattern.

If `points` is True, each pixel is approximated as a point
else each pixel is a hexagon

.. ipython::

    In [1]: def hextess(n, points=False):
       ...:     x = np.vstack(np.array([[i + j*.5, j*3**.5*.5]
       ...:         for j in range(-n - min(0, i), n - max(0, i) + 1)])
       ...:         for i in range(-n, n + 1))/(n + .5)
       ...:     if points:
       ...:         a = np.ones((len(x),))*3**.5/(n + .5)**2/2
       ...:         return [PointPixelElectrode(points=[xi], areas=[ai]) for
       ...:                 xi, ai in zip(x, a)]
       ...:     else:
       ...:         a = 1/(3**.5*(n + .5)) # edge length
       ...:         p = x[:, None, :] + [[[a*np.cos(phi), a*np.sin(phi)] for phi in
       ...:             np.arange(np.pi/6, 2*np.pi, np.pi/3)]]
       ...:         return [PolygonPixelElectrode(paths=[i]) for i in p]

    
Define a function that returns a System instance with a single hexagonal rf
pixel electrode.

The pixel factors (whether a pixel is grounded or at rf) are optimized
to yield three trapping sites forming an equilateral triangle with

* n pixels per unit length,
* ion separation d,
* ion height h, and
* trapping frequencies with a ratio 2:1:1 (radial being the strongest).

The effect of a grounded shielding cover at height H is accounted for
up to nmax components in the expansion in h/H.

.. ipython::

    In [1]: def threefold(n, h, d, H, nmax=1, points=True):
       ...:     s = System(hextess(n, points))
       ...:     ct = []
       ...:     ct.append(PatternRangeConstraint(min=0, max=1.))
       ...:     for p in 0, 4*np.pi/3, 2*np.pi/3:
       ...:         x = np.array([d/3**.5*np.cos(p), d/3**.5*np.sin(p), h])
       ...:         r = euler_matrix(p, np.pi/2, np.pi/4, "rzyz")[:3, :3]
       ...:         for i in "x y z xy xz yz".split():
       ...:             ct.append(PotentialObjective(derivative=i, x=x,
       ...:                 rotation=r, value=0))
       ...:         for i, v in ("xx", 1), ("yy", 1):
       ...:             ct.append(PotentialObjective(derivative=i, x=x,
       ...:                 rotation=r, value=2**(-1/3.)*v))
       ...:     s.rfs, c = s.optimize(ct)
       ...:     return s, c

Run the optimization.

.. ipython::

    In [1]: points, n, h, d, H = False, 50, 1/8., 1/4., 25/8.

    In [1]: n, points = 12, True # take a simpler, approximate problem

    In [1]: s, c = threefold(n, h, d, H, nmax=1, points=points)

Analysis of the result:

.. ipython::

    In [1]: x0 = np.array([d/3**.5, 0, h])
    
    In [1]: print "c*h**2*c:", h**2 # optimal strength of the constraints

    In [1]: print "rf'/c:", s.electrical_potential(x0, "rf", 1)[0]/c # rf field should vanish

    In [1]: print "rf''/c:", s.electrical_potential(x0, "rf", 2)[0]/c # rf curvature should be (2, 1, 1)/2**(1/3)

Plot the electrode pattern, white is ground, black/red is rf.

.. ipython::

    In [1]: fig, ax = plt.subplots()

    In [1]: ax.set_aspect("equal"), ax.set_xlim((-1,1)), ax.set_ylim((-1,1))

    @savefig threefold_ele.png width=6in
    In [1]: s.plot_voltages(ax, u=s.rfs)

Some textual analysis of one of the trapping sites.

.. ipython::

    In [1]: l = 320e-6 # length scale, hexagon radius

    In [1]: u = 20. # peak rf voltage

    In [1]: o = 2*np.pi*50e6 # rf frequency

    In [1]: m = 24*ct.atomic_mass # ion mass

    In [1]: q = 1*ct.elementary_charge # ion charge

    In [1]: # sphinx does not cope with unicode

    In [1]: for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
       ...:     print line.encode("ascii", errors="replace")


Plot the horizontal logarithmic pseudopotential at the ion height.

.. ipython::

    In [1]: n = 50

    In [1]: xyz = np.mgrid[-d:d:1j*n, -d:d:1j*n, h:h+1]

    In [1]: fig, ax = plt.subplots()

    In [1]: ax.set_aspect("equal")

    @savefig threefold_xy.png width=6in
    In [1]: ax.contour(xyz[0].reshape((n,n)), xyz[1].reshape((n,n)),
       ...:            np.log(shaped(s.potential)(xyz)).reshape((n,n)),
       ...:            20, cmap=plt.cm.hot)


Plot the logarithmic pseudopotential and the separatrix in the xz plane.

.. ipython::

    In [1]: xs1, ps1 = s.saddle(x0+1e-2)

    In [1]: xs0, ps0 = s.saddle([0, 0, .8])

    In [1]: print "main saddle:", xs0, ps0

    In [1]: n = 50

    In [1]: xyz = np.mgrid[-d:d:1j*n, 0:1, .7*h:3*h:1j*n]

    In [1]: p = shaped(s.potential)(xyz)

    In [1]: fig, ax = plt.subplots()

    In [1]: ax.set_aspect("equal")

    In [1]: ax.contour(xyz[0].reshape((n,n)), xyz[2].reshape((n,n)),
       ...:            np.log(p).reshape((n,n)),
       ...:            20, cmap=plt.cm.hot)

    @savefig threefold_xz.png width=6in
    In [1]: ax.contour(xyz[0].reshape((n,n)), xyz[2].reshape((n,n)),
       ...:            np.log(p).reshape((n,n)),
       ...:            [np.log(ps0), np.log(ps1)], color="black")


