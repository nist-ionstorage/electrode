# -*- coding: utf8 -*-
#
#   saddle.py: Rational function optimization (RFO),
#              a saddlepoint search method
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

def rfo(fun, grad, x0, args=(), 
        xtol=1e-4, ftol=1e-4, maxiter=200, dx_max=1., h=None, cb=None):
    """Rational function optimization with approximate hessian.

    Finds a saddle point of fun near x0 using a modified Newton-Raphson.
    The Hessian is approximated with Bofill updates.

    See "Comparison of methods for finding saddle points without
    knowledge of the final states", R. A. Olsen et al., J. Chem. Phys.,
    121, 20, (2004).
    """

    x = np.matrix(x0).T.copy()
    f = fun(np.array(x).ravel())
    g = np.matrix(grad(np.array(x).ravel())).T
    if h is None:
        h = np.matrix(np.identity(x.size))
    else:
        h = np.matrix(h)

    d = np.ones((x.size,))
    d[0] *= -1

    it = 0

    ret = None

    while True:

        it += 1

        #print "i", it
        #print "x", x
        #print "f", f
        #print "g", g
        #print "h", h
        #print
        if cb is not None:
            cb(x, f, g, h)
        if ret:
            break

        l, v = np.linalg.eigh(h)
        i = np.argsort(l)
        l, v = l[i], v[i]
        v = np.matrix(v)

        gl = np.array(v.T*g).ravel()
        lmg = .5*d*(abs(l) + np.sqrt(l**2 + 4*gl**2))

        dx = -v*np.matrix(gl/lmg).T
        dx /= max(1., np.linalg.norm(dx)/dx_max)
        df = fun(np.array(x + dx).ravel()) - f
        dg = np.matrix(grad(np.array(x + dx).ravel())).T - g
        dp = dg - h*dx
        dh_powell = (dp*dx.T + dx*dp.T)/(dx.T*dx) - \
                float(dp.T*dx)*(dx*dx.T)/(dx.T*dx)**2
        dh_sr1 = (dp*dp.T)/(dp.T*dx)
        phi = float((dp.T*dx)**2/((dp.T*dp)*(dx.T*dx)))
        dh_bofill = phi*dh_sr1 + (1-phi)*dh_powell
        dh = dh_bofill

        if 2.*np.linalg.norm(dx) <= xtol * (
                np.linalg.norm(x+dx) + np.linalg.norm(x) + 1e-20):
            ret = "xtol"

        if 2.*np.abs(df) <= ftol * (abs(f+df) + abs(f) + 1e-20):
            ret = "ftol"

        if maxiter > 0 and it > maxiter:
            ret = "iter"

        x += dx
        f += df
        g += dg
        h += dh

    return np.array(x).ravel(), f, ret


if __name__ == "__main__":
    pot = "cerjan-miller"
    if pot == "saddle":
        f = lambda x: x[0]**2 - x[1]**2
        g = lambda x: [2*x[0], -2*x[1]]
        x0 = [1., 2.]
    elif pot == "random":
        f = lambda x: x[0]**2 - x[1]**2 + (x[2]-1)**3 + x[0]*x[2]
        g = lambda x: [2*x[0]+x[2], -2*x[1], 3*(x[2]-1)**2+x[0]]
        x0 = [1., 2., 3.]
    elif pot == "adams":
        f = lambda x: (2*x[0]**2*(4-x[0])
                +x[1]**2*(4+x[1])
                -x[0]*x[1]*(6-17*np.exp((-x[0]**2-x[1]**2)/4)))
        g = lambda x: [-6*x[1]+16*x[0]-6*x[0]**2+
            17*x[1]*(1-x[0]**2/2)*np.exp((-x[0]**2-x[1]**2)/4),
                       -6*x[0]+8*x[1]+3*x[1]**2+
            17*x[0]*(1-x[1]**2/2)*np.exp((-x[0]**2-x[1]**2)/4)]
        xi, yi = np.mgrid[-2.3:3.7:30j, -3:3:30j]
    elif pot == "cerjan-miller":
        f = lambda x: (1-x[1])*x[0]**2*np.exp(-x[0]**2)+x[1]**2/2
        g = lambda x: [2*x[0]*(1-x[1])*(-x[0]**2+1)*np.exp(-x[0]**2),
               x[1]-x[0]**2*np.exp(-x[0]**2)] 
        xi, yi = np.mgrid[-1.3:1.3:30j, -.7:1.9:30j]

    import pylab as pl
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1, aspect="equal")
    ax.contour(xi, yi, f([xi, yi]), 50, cmap=pl.cm.hot)
    gx, gy = g([xi, yi])
    ax.quiver(xi, yi, gx, gy, 50)
    
    for x0 in (np.random.rand(20, 2)-.5)/10:
        xs, fs, gs, hs = [], [], [], []
        def record(x, f, g, h):
            xs.append(x.copy())
            fs.append(f.copy())
            gs.append(g.copy())
            hs.append(h.copy())
        x, p, ret = rfo(f, g, x0, dx_max=.2, cb=record)
        xs, fs, gs, hs = map(lambda a: np.array(a).reshape((len(xs), -1)),
                (xs, fs, gs, hs))
        #print x, p, g(x)
        ax.plot(xs[:, 0], xs[:, 1], "k")
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())
    pl.show()

