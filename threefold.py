import numpy as np
from numpy import (cos, sin, pi, array, ones, mgrid, log, arange)
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import multiprocessing

from qc.theory.transformations import euler_matrix
from qc.traps.electrode import (System,
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

def threefold(n=120, h=1/8., d=1/4., H=25/8., nmax=1, points=True):
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

    from qc.data.atoms import Mg24p
    l = 320e-6
    u = 20.
    m = Mg24p.mass
    q = Mg24p.charge
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
