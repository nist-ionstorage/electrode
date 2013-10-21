Tutorial
========

All parts of the tutorial are presented to be run in IPython. The
required packages are only imported ones but used in all sections.


Setup
-----

First import the needed packages and modules. From the `electrode`
package, a set of electrodes form a `System`. Here, the electrodes are
either `PointPixelElectrodes` for point approximations or
`PolygonPixelElectrodes` for polygonal structures.

.. ipython::

    In [1]: import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct
       ...: from electrode import (System, PolygonPixelElectrode, euler_matrix,
       ...:     PointPixelElectrode, PotentialObjective,
       ...:     PatternRangeConstraint, shaped)
       ...:
       ...: np.set_printoptions(precision=2) # have numpy print fewer digits


Linear surface trap
-------------------

Let's start with a very simple five-wire linear surface electrode trap.

We start with a function that returns a parametrized system of surface
electrodes. This way different designs can be compared quickly and the
design parameter space can be explored.

There ware two rf wires running along `x` with width in the `y`
direction of `top` and `bottom`. Between them, there is a long `dc`
electrode `c` of width `mid`. Above and below the rf electrodes there
are three dc electrodes `tl, tm, tr` and `bl, bm, br` to provide stray
field compensation, axial confinement and knobs to change the curvature
tensor.

.. ipython::

    In [1]: def five_wire(edge, width, top, mid, bot):
       ...:     e, r, t, m, b = edge, width/2, top + mid/2, mid/2, -bot - mid/2
       ...:     electrodes = [
       ...:         ("tl", [[(-e, e), (-e, t), (-r, t), (-r, e)]]),
       ...:         ("tm", [[(-r, e), (-r, t), (r, t), (r, e)]]),
       ...:         ("tr", [[(r, e), (r, t), (e, t), (e, e)]]),
       ...:         ("bl", [[(-e, -e), (-r, -e), (-r, b), (-e, b)]]),
       ...:         ("bm", [[(-r, -e), (r, -e), (r, b), (-r, b)]]),
       ...:         ("br", [[(r, -e), (e, -e), (e, b), (r, b)]]),
       ...:         ("r", [[(-e, t), (-e, m), (e, m), (e, t)],
       ...:                [(-e, b), (e, b), (e, -m), (-e, -m)]]),
       ...:         ("c",  [[(-e, m), (-e, -m), (e, -m), (e, m)]]),
       ...:         ]
       ...:     s = System([PolygonPixelElectrode(name=n, paths=map(np.array, p))
       ...:                 for n, p in electrodes])
       ...:     s["r"].rf = 1.
       ...:     return s

Now we can retrieve such a system and plot the electrodes' shapes, and
the rf voltages on them.

.. ipython::

    @savefig five_ele.png width=6in
    In [1]: s = five_wire(5, 2., 1., 1., 1.)
       ...: 
       ...: fig, ax = plt.subplots(1, 2)
       ...: s.plot(ax[0])
       ...: s.plot_voltages(ax[1], u=s.rfs)
       ...: 
       ...: r = 5
       ...: for axi in ax.flat:
       ...:     axi.set_aspect("equal")
       ...:     axi.set_xlim(-r, r)
       ...:     axi.set_ylim(-r, r)

To trap an ion in this trap, we find the potential minimum in `x0` the `yz`
plane (`axis=(1, 2)`) and perform a analysis of the potential landscape
at and around this minimum assuming some typical operating parameters.
Again, we constrain the search for the minimum and the saddle point to
the `yz` plane since there is no adequate axial confinement yet.

.. ipython::

    In [1]: l = 30e-6 # µm length scale
       ...: u = 20. # V rf peak voltage
       ...: m = 25*ct.atomic_mass # ion mass
       ...: q = 1*ct.elementary_charge # ion charge
       ...: o = 2*np.pi*100e6 # rf frequency in rad/s
       ...: 
       ...: x0 = s.minimum((0, 0, 1.), axis=(1, 2))
       ...:
       ...: for line in s.analyze_static(x0, axis=(1, 2), m=m, q=q, u=u, l=l, o=o):
       ...:     print line.encode("ascii", errors="replace")

The seven dc electrodes (three top, three bottom and the center wire)
can be used to apply electrical fields and electrical curvatures to
compensate stray fields and confine the ion axially.

The `shim()` method can be used to calculate voltage vectors for these
dc electrodes that are result in orthogonal effects with regards to some
cartesian partial derivatives at certain points. To use it we build a
temporary new system `s1` holding only the dc electrodes. Since these dc
electrode instances also appear in our primary system `s`, changes in
voltages are automatically synchronized between the two systems.

We then can calculate the shim voltage vectors that result in unit
changes of each of the partial derivatives `y, z, xx, xy, xz, yy` at
`x0` and plot the voltage distributions.

.. ipython::

    @savefig five_shim.png width=8in
    In [1]: s1 = System([e for e in s if not e.rf])
       ...: derivs = "y z xx xy xz yy".split()
       ...: u = s1.shims([(x0, None, deriv) for deriv in derivs])
       ...: 
       ...: fig, ax = plt.subplots(2, len(derivs)/2, figsize=(12, 10))
       ...: for d, ui, axi in zip(derivs, u, ax.flat):
       ...:     with s1.with_voltages(dcs=ui):
       ...:         s.plot_voltages(axi)
       ...:     axi.set_aspect("equal")
       ...:     axi.set_xlim(-r, r)
       ...:     axi.set_ylim(-r, r)
       ...:     um = ui[np.argmax(np.fabs(ui))]
       ...:     axi.set_title("%s, max=%g" % (d, um))


Rf/dc pattern optimization
--------------------------

Define a function that generates the pixels and electrode. Here 
we return pixel electrodes with `n` pixels per unit
length in a hexagonal pattern.

If `points` is True, each pixel is approximated as a point
else each pixel is a hexagon.

.. ipython::

    In [1]: def hextess(n, points):
       ...:     x = np.vstack([[i + j*.5, j*3**.5*.5]
       ...:         for j in range(-n - min(0, i), n - max(0, i) + 1)]
       ...:         for i in range(-n, n + 1))/(n + .5) # centers
       ...:     if points:
       ...:         a = np.ones(len(x))*3**.5/(n + .5)**2/2 # areas
       ...:         return [PointPixelElectrode(points=[xi], areas=[ai]) for
       ...:                 xi, ai in zip(x, a)]
       ...:     else:
       ...:         a = 1/(3**.5*(n + .5)) # edge length
       ...:         p = x[:, None] + [[a*np.cos(phi), a*np.sin(phi)] for phi in
       ...:             np.arange(np.pi/6, 2*np.pi, np.pi/3)]
       ...:         return [PolygonPixelElectrode(paths=[i]) for i in p]

Now define a function that returns a System instance with a single hexagonal
rf pixel electrode.

The pixel factors (whether a pixel is grounded or at rf) are optimized
to yield three trapping sites forming an equilateral triangle with

    * `n` pixels per unit length,

    * trap separation `d`,

    * trap height `h` above the surface electrodes, and

    * trapping frequencies with a ratio `2:1:1` (radial being the strongest).

.. ipython::

    In [1]: def threefold(n, h, d, points=True):
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
       ...:                 rotation=r, value=v))
       ...:     s.rfs, c = s.optimize(ct)
       ...:     return s, c

Run the optimization.

.. ipython::

    In [1]: points, n, h, d = True, 12, 1/8., 1/4.
       ...: s, c = threefold(n, h, d, points)

Analysis of the result. `c` is the obtained strength of the constraints,
the rf field should vanish and the rf curvature should be `(2, 1, 1)`.

.. ipython::

    In [1]: x0 = np.array([d/3**.5, 0, h])
       ...: print "c:", c
       ...: print "rf'/c:", s.electrical_potential(x0, "rf", 1)[0]/c
       ...: print "rf''/c:", s.electrical_potential(x0, "rf", 2)[0]/c

Plot the electrode pattern, white is ground, black/red is rf.

.. ipython::

    @savefig threefold_ele.png width=6in
    In [1]: fig, ax = plt.subplots()
       ...: ax.set_aspect("equal")
       ...: ax.set_xlim((-1,1))
       ...: ax.set_ylim((-1,1))
       ...: s.plot_voltages(ax, u=s.rfs)

Some textual analysis of one of the trapping sites.

.. sphinx does not cope with unicode

.. ipython::

    In [1]: l = 320e-6 # length scale, hexagon radius
       ...: u = 20. # peak rf voltage
       ...: o = 2*np.pi*50e6 # rf frequency
       ...: m = 24*ct.atomic_mass # ion mass
       ...: q = 1*ct.elementary_charge # ion charge
       ...:
       ...: for line in s.analyze_static(x0, l=l, u=u, o=o, m=m, q=q):
       ...:     print line.encode("ascii", errors="replace")


Plot the horizontal logarithmic pseudopotential at the ion height
and the logarithmic pseudopotential and the separatrix in the xz plane.

.. ipython::

    @savefig threefold_xy_xz.png width=6in
    In [1]: n = 50
       ...: xyz = np.mgrid[-d:d:1j*n, -d:d:1j*n, h:h+1]
       ...: fig, ax = plt.subplots(1, 2, subplot_kw=dict(aspect="equal"))
       ...: pot = shaped(s.potential)(xyz)
       ...: v = np.arange(-10, 3)
       ...: x, y, p = (_.reshape(n, n) for _ in (xyz[0], xyz[1], pot))
       ...: ax[0].contour(x, y, np.log2(p), v, cmap=plt.cm.hot)
       ...:
       ...: (xs1, ps1), (xs0, ps0) = s.saddle(x0+1e-2), s.saddle([0, 0, .8])
       ...: print "main saddle:", xs0, ps0
       ...: xyz = np.mgrid[-d:d:1j*n, 0:1, .7*h:3*h:1j*n]
       ...: pot = shaped(s.potential)(xyz)
       ...: x, z, p = (_.reshape(n, n) for _ in (xyz[0], xyz[2], pot))
       ...: ax[1].contour(x, z, np.log2(p), v, cmap=plt.cm.hot)
       ...: ax[1].contour(x, z, np.log2(p), np.log2((ps0, ps1)), color="black")
