# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@phys.ethz.ch>
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

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from traits.api import HasTraits, Array, Float, Int, Str, List

from .utils import norm, expand_tensor, area_centroid

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)


class Electrode(HasTraits):
    name = Str()
    voltage_dc = Float(0.)
    voltage_rf = Float(0.)

    def electrical_potential(self, x):
        """return the eletrical, units are volts"""
        raise NotImplementedError

    def electrical_gradient(self, x):
        """return the eletrical potential gradient,
        units are volts/length scale"""
        raise NotImplementedError

    def electrical_curvature(self, x):
        """return the eletrical potential curvature,
        units are volts/length scale^2"""
        raise NotImplementedError

    def orientations(self):
        """return the orientation of the patches (positive orientation
        yields positive potential for positive voltage and z>0"""
        raise NotImplementedError

    def plot(self, ax, text=None, *a, **kw):
        """plot this electrode's patches in the supplied axes"""
        raise NotImplementedError


class CoverElectrode(Electrode):
    voltage_dc = 0.
    voltage_rf = 0.
    cover_height = Float(100)
    # also adjust cover_height in
    # the other electrodes to include the cover's effect on their
    # potentials

    def potential(self, x, *d):
        x = np.atleast_2d(x)
        r = []
        if 0 in d:
            r.append(x[:, 2]/self.cover_height)
        if 1 in d:
            ri = np.zeros((3, x.shape[0]))
            ri[2] = 1/self.cover_height
            r.append(ri)
        if 2 in d:
            r.append(np.zeros((3, 3, x.shape[0])))
        return r

    def electrical_potential(self, x):
        return self.voltage_dc*self.potential(x, 0)[0]

    def electrical_gradient(self, x):
        return self.voltage_dc*self.potential(x, 1)[0]

    def electrical_curvature(self, x):
        return self.voltage_dc*self.potential(x, 2)[0]

    def orientations(self):
        return np.array([1])

    def plot(self, ax, text=None, *a, **kw):
        pass


class PixelElectrode(Electrode):
    """
    Parts of the PixelElectrode code are based on:

    Roman Schmied, SurfacePattern software package
    http://atom.physik.unibas.ch/people/romanschmied/code/SurfacePattern.php

    [1] R. Schmied, "Electrostatics of gapped and finite surface
    electrodes", New J. Phys. 12:023038 (2010),
    http://dx.doi.org/10.1088/1367-2630/12/2/023038

    [2] R. Schmied, J. H. Wesenberg, and D. Leibfried, "Optimal
    Surface-Electrode Trap Lattices for Quantum Simulation with Trapped
    Ions", PRL 102:233002 (2009),
    http://dx.doi.org/10.1103/PhysRevLett.102.233002
    """
    pixel_factors = Array(dtype=np.float64, shape=(None,))
    cover_height = Float # cover plane height
    nmax = Int(0) # max components in cover plane potential expansion

    def value_no_cover(self, x, *d):
        """bare pixel potential and derivative (d) value at x.
        indices are (components if d>0, pixel, x)"""
        raise NotImplementedError

    def value(self, x, *d):
        """potential and derivative value with cover plane"""
        x = np.atleast_2d(x).astype(np.double)
        r = self.value_no_cover(x, *d)
        for n in range(-self.nmax, 0) + range(1, self.nmax+1):
            xx = x + [[0, 0, 2*n*self.cover_height]]
            for i, ri in enumerate(self.value_no_cover(xx, *d)):
                r[i] += ri
        return r

    def potential(self, x, *d):
        """return the potential/its derivatives d at x with the pixel
        voltages multiplied and the tensor expanded to full form"""
        x = np.atleast_2d(x)
        v = self.value(x, *d)
        for i, vi in enumerate(v):
            p = self.pixel_factors[:, None]*vi
            p = expand_tensor(p.sum(axis=-2))
            v[i] = p
        return v

    def electrical_potential(self, x):
        e, = self.potential(x, 0)
        return self.voltage_dc*e

    def electrical_gradient(self, x):
        e, = self.potential(x, 1)
        return self.voltage_dc*e

    def electrical_curvature(self, x):
        e, = self.potential(x, 2)
        return self.voltage_dc*e

    def electrical_thirdderiv(self, x):
        e, = self.potential(x, 3)
        return self.voltage_dc*e

    def pseudo_potential(self, x):
        e, = self.potential(x, 1)
        return self.voltage_rf**2*(e**2).sum(axis=0)

    def pseudo_gradient(self, x):
        e, g = self.potential(x, 1, 2)
        return self.voltage_rf**2*2*(e[:, None]*g).sum(axis=0)

    def pseudo_curvature(self, x):
        e, g, c = self.potential(x, 1, 2, 3)
        return self.voltage_rf**2*2*(
                g[:, :, None]*g[:, None, :]+e[:, None, None]*c
                ).sum(axis=0)

    def optimize(self, constraints, verbose=True):
        """optimize this electrode's pixel voltages with respect to
        constraints"""
        p = cvxopt.modeling.variable(len(self.pixel_factors))
        obj = []
        ctrs = []
        for ci in constraints:
            obj.extend(ci.objective(self, p))
            ctrs.extend(ci.constraints(self, p))
        B = np.matrix([i[0] for i in obj])
        b = np.matrix([i[1] for i in obj])
        # the inhomogeneous solution
        g = b*np.linalg.pinv(B).T
        # maximize this
        obj = cvxopt.matrix(g)*p
        # B*g_perp
        B1 = B - b.T*g/(g*g.T)
        #u, l, v = np.linalg.svd(B1)
        #li = np.argmin(l)
        #print li, l[li], v[li], B1*v[li].T
        #self.pixel_factors = np.array(v)[li]
        #return 0.
        #FIXME: there is one singular value, drop one constraint
        B1 = B1[:-1]
        # B*g_perp*p == 0
        ctrs.append(cvxopt.matrix(B1)*p == 0.)
        solver = cvxopt.modeling.op(-obj, ctrs)
        if not verbose:
            cvxopt.solvers.options["show_progress"] = False
        else:
            print "variables:", sum(v._size
                    for v in solver.variables())
            print "inequalities", sum(v.multiplier._size
                    for v in solver.inequalities())
            print "equalities", sum(v.multiplier._size
                    for v in solver.equalities())
        solver.solve("sparse")
        c = float(np.matrix(p.value).T*g.T/(g*g.T))
        p = np.array(p.value).ravel()
        return p, c


class PointPixelElectrode(PixelElectrode):
    points = Array(dtype=np.float64, shape=(None, 3))
    areas = Array(dtype=np.float64, shape=(None,))

    def _areas_default(self):
        return np.ones((len(self.points)))

    def _pixel_factors_default(self):
        return np.ones(self.areas.shape)

    def orientations(self):
        return np.ones(self.areas.shape)

    def plot(self, ax, text=None, alpha=1., *a, **kw):
        # color="red"?
        p = self.points
        a = (self.areas/np.pi)**.5*2
        col = mpl.collections.EllipseCollection(
                edgecolors="none", cmap=plt.cm.binary,
                norm=plt.Normalize(0, 1.),
                widths=a, heights=a, units="x", # xy in matplotlib>r8111
                angles=np.zeros(a.shape),
                offsets=p[:, (0, 1)], transOffset=ax.transData)
        col.set_array(alpha*self.pixel_factors)
        ax.add_collection(col)
        if text is None:
            text = self.name
        if text:
            ax.text(p[:,0].mean(), p[:,1].mean(), text)

    def value_no_cover(self, x, *d):
        return [v.transpose((2, 0, 1)) if v.ndim==3 else v
                for v in point_value(x, self.areas, self.points, *d)]


class PolygonPixelElectrode(PixelElectrode):
    paths = List(Array(dtype=np.float64, shape=(None, 3)))

    def _pixel_factors_default(self):
        return np.ones(len(self.paths))

    def orientations(self):
        p, = self.value_no_cover(np.array([[0, 0, 1.]]), 0)
        return np.sign(p[:, 0])

    def plot(self, ax, text=None, alpha=1., edgecolor="none", *a, **kw):
        if text is None:
            text = self.name
        for vi, p in zip(self.pixel_factors, self.paths):
            ax.fill(p[:,0], p[:,1], edgecolor=edgecolor,
                    alpha=alpha*vi, *a, **kw)
            if text:
                ax.text(p[:,0].mean(), p[:,1].mean(), text)

    def to_points(self):
        a, c = zip(*(area_centroid(p) for p in self.paths))
        return PointPixelElectrode(name=self.name,
                pixel_factors=self.pixel_factors, nmax=self.nmax,
                cover_height=self.cover_height, areas=a, points=c)

    def value_no_cover(self, x, *d):
        return [v.transpose((2, 0, 1)) if v.ndim==3 else v
                for v in polygon_value(x, list(self.paths), *d)]

try:
    #raise ImportError
    from .speedups import point_value, polygon_value
except ImportError:
    def point_value(x, p, *d):
        return [v.transpose((1, 2, 0)) if v.ndim==3 else v
                for v in _point_value(x, p, *d)]

    def _point_value(x, a, p, *d):
        a = a[:, None]
        p1 = x[None, :] - p[:, None]
        r = norm(p1)
        x, y, z = p1.transpose((2, 0, 1))
        if 0 in d:
            yield a * z/(2*np.pi*r**3) #
        if 1 in d:
            yield a * np.array([-3*x*z, -3*y*z, x**2+y**2-2*z**2] # x y z
            )/(2*np.pi*r**5)
        if 2 in d:
            yield a * np.array([-3*z*(-4*x**2+y**2+z**2), 15*x*y*z, # xx xy
            -3*x*(x**2+y**2-4*z**2), -3*z*(x**2-4*y**2+z**2), # xz yy
            -3*y*(x**2+y**2-4*z**2)] # yz
            )/(2*np.pi*r**7)
        if 3 in d:
            yield a * np.array([15*y*z*(-6*x**2+y**2+z**2), # xxy
            3*(4*x**4-y**4+3*y**2*z**2+4*z**4+3*x**2*(y**2-9*z**2)), # xxz
            3*(-x**4+4*y**4-27*y**2*z**2+4*z**4+3*x**2*(y**2+z**2)), # yyz
            15*x*z*(x**2-6*y**2+z**2), # yyx
            45*x*(x**2+y**2)*z-60*x*z**3, # zzx
            45*y*(x**2+y**2)*z-60*y*z**3, # xxy
            15*x*y*(x**2+y**2-6*z**2)] # xyz
            )/(2*np.pi*r**9)

    def polygon_value(x, p, *d):
        v = [_polygon_value(x, pi, *d) for pi in p]
        for vi in zip(*v):
            vi = np.array(vi)
            if vi.ndim == 3:
                vi = vi.transpose((0, 2, 1))
            yield vi

    def _polygon_value(x, p, *d):
        p1 = x[None, :] - p[:, None]
        x1, y1, z = p1.transpose((2, 0, 1))
        r1 = norm(p1)
        x2 = np.roll(x1, -1, axis=0)
        y2 = np.roll(y1, -1, axis=0)
        r2 = np.roll(r1, -1, axis=0)
        l2 = (x1-x2)**2+(y1-y2)**2
        if 0 in d:
            zs = np.abs(z)
            yield np.arctan2(z*(x1*y2-y1*x2),
                    zs*(r1*r2+x1*x2+y1*y2+zs*(zs+r1+r2))).sum(axis=0)/np.pi
        if 1 in d:
            yield (np.array([-(y1-y2)*z, (x1-x2)*z, x2*y1-x1*y2]
                )*(r1+r2)/(r1*r2*((r1+r2)**2-l2))).sum(axis=1)/np.pi
        if 2 in d:
            yield (np.array([(l2*(r2**3*x1+r1**3*x2)-
                (r1+r2)**2*(r2**2*(2*r1+r2)*x1+r1**2*(r1+2*r2)*x2))*(-y1+y2)*z,
                  (-y1+y2)*(l2*(r2**3*y1+r1**3*y2)-
                  (r1+r2)**2*(r2**2*(2*r1+r2)*y1+r1**2*(r1+2*r2)*y2))*z,
                (r1+r2)*(-y1+y2)*(-(l2*r1**2*r2**2)+
                  l2*(r1**2-r1*r2+r2**2)*z**2+(r1+r2)**2*(r1**2*r2**2-
                  (r1**2+r1*r2+r2**2)*z**2)),
                (x1-x2)*(l2*(r2**3*y1+r1**3*y2)-
                  (r1+r2)**2*(r2**2*(2*r1+r2)*y1+r1**2*(r1+2*r2)*y2))*z,
                (r1+r2)*(-x1+x2)*(l2*r1**2*r2**2-l2*(r1**2-r1*r2+r2**2)*z**2+
                  (r1+r2)**2*(-(r1**2*r2**2)+(r1**2+r1*r2+r2**2)*z**2))
                ])/((r1*r2)**3*((r1+r2)**2-l2)**2)).sum(axis=1)/np.pi
        if 3 in d:
            yield (np.array([(-y1+y2)*(3*l2**2*(r2**5*x1*y1+r1**5*x2*y2)+
                  (r1+r2)**3*(9*r1*r2**5*x1*y1+3*r2**6*x1*y1+3*r1**6*x2*y2+
                  9*r1**5*r2*x2*y2+6*r1**3*r2**3*(x2*y1+x1*y2)+
                  2*r1**2*r2**4*(x2*y1+x1*(4*y1+y2))+
                  2*r1**4*r2**2*(x1*y2+x2*(y1+4*y2)))-
                  2*l2*(9*r1*r2**6*x1*y1+3*r2**7*x1*y1+3*r1**7*x2*y2+
                  9*r1**6*r2*x2*y2+r1**2*r2**5*(x2*y1+x1*(6*y1+y2))+
                  r1**5*r2**2*(x1*y2+x2*(y1+6*y2))))*z,
                (-y1+y2)*(-(l2**2*(r1**2*r2**5*x1-3*r2**5*x1*z**2+
                  r1**5*x2*(r2**2-3*z**2)))+2*l2*(r1+r2)**3*(-3*r2**4*x1*z**2+
                  r1**4*x2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*x1-(x1+x2)*z**2))-
                  (r1+r2)**2*(-12*r1*r2**6*x1*z**2-3*r2**7*x1*z**2+
                  r1**7*x2*(r2**2-3*z**2)+4*r1**6*r2*x2*(r2**2-3*z**2)+
                  r1**4*r2**3*(r2**2*(5*x1+2*x2)-8*(x1+2*x2)*z**2)+
                  r1**5*r2**2*(r2**2*(2*x1+5*x2)-(2*x1+19*x2)*z**2)+
                  r1**2*r2**3*(r2**4*x1-r2**2*(19*x1+2*x2)*z**2-
                  6*x1*((x1-x2)**2+(y1-y2)**2)*z**2)+
                  2*r1**3*r2**2*(2*r2**4*x1-4*r2**2*(2*x1+x2)*z**2-3*x2*(
                  (x1-x2)**2+(y1-y2)**2)*z**2))),
                (-x1+x2)*(l2**2*(r1**2*r2**5*y1-3*r2**5*y1*z**2+
                  r1**5*y2*(r2**2-3*z**2))-2*l2*(r1+r2)**3*(-3*r2**4*y1*z**2+
                  r1**4*y2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*y1-(y1+y2)*z**2))+
                  (r1+r2)**2*(-12*r1*r2**6*y1*z**2-3*r2**7*y1*z**2+
                  r1**7*y2*(r2**2-3*z**2)+4*r1**6*r2*y2*(r2**2-3*z**2)+
                  2*r1**3*r2**2*(2*r2**4*y1-3*((x1-x2)**2+(y1-y2)**2)*y2*z**2-
                  4*r2**2*(2*y1+y2)*z**2)+r1**4*r2**3*(r2**2*(5*y1+2*y2)-
                  8*(y1+2*y2)*z**2)+r1**2*r2**3*(r2**4*y1-6*y1*((x1-x2)**2+
                  (y1-y2)**2)*z**2-r2**2*(19*y1+2*y2)*z**2)+
                  r1**5*r2**2*(r2**2*(2*y1+5*y2)-(2*y1+19*y2)*z**2))),
                (-y1+y2)*(2*l2*(r1**2*r2**2*(r1+r2)**3*(r1**2+r2**2)-
                  3*r2**5*(r1+r2)*(2*r1+r2)*y1**2-2*r1**2*r2**2*(r1**3+
                  r2**3)*y1*y2-3*r1**5*(r1+r2)*(r1+2*r2)*y2**2)-
                  l2**2*(r1**2*r2**5-3*r2**5*y1**2+r1**5*(r2**2-3*y2**2))-
                  (r1+r2)**3*(-9*r1*r2**5*y1**2-3*r2**6*y1**2+
                  3*r1**3*r2**3*(r2**2-4*y1*y2)+r1**6*(r2**2-3*y2**2)+
                  3*r1**5*r2*(r2**2-3*y2**2)+r1**2*r2**4*(r2**2-4*y1*(2*y1+y2))+
                  4*r1**4*r2**2*(r2**2-y2*(y1+2*y2))))*z,
                (-r1-r2)*(-y1+y2)*z*(-2*l2*(r1+r2)**2*(3*r1**2*r2**2*(r1**2+
                  r2**2)+(-3*r1**4+2*r1**3*r2+r1**2*r2**2+2*r1*r2**3-
                  3*r2**4)*z**2)+3*l2**2*(r1**2*r2**2*(r1**2-r1*r2+r2**2)-
                  (r1**4-r1**3*r2+r1**2*r2**2-r1*r2**3+r2**4)*z**2)+
                  (r1+r2)**2*(-3*r2**6*z**2+r1*r2**3*(-9*r2**2+4*((x1-x2)**2+
                  (y1-y2)**2))*z**2+3*r1**2*r2**4*(r2**2-4*z**2)+
                  3*r1**6*(r2**2-z**2)+9*r1**5*r2*(r2**2-z**2)+
                  12*r1**4*r2**2*(r2**2-z**2)+r1**3*r2*(9*r2**4-12*r2**2*z**2+
                  4*((x1-x2)**2+(y1-y2)**2)*z**2))),
                (r1+r2)*(-x1+x2)*z*(-2*l2*(r1+r2)**2*(3*r1**2*r2**2*(r1**2+
                  r2**2)+(-3*r1**4+2*r1**3*r2+r1**2*r2**2+2*r1*r2**3-
                  3*r2**4)*z**2)+3*l2**2*(r1**2*r2**2*(r1**2-r1*r2+r2**2)-
                  (r1**4-r1**3*r2+r1**2*r2**2-r1*r2**3+r2**4)*z**2)+
                  (r1+r2)**2*(-3*r2**6*z**2+r1*r2**3*(-9*r2**2+4*((x1-x2)**2+
                  (y1-y2)**2))*z**2+3*r1**2*r2**4*(r2**2-4*z**2)+
                  3*r1**6*(r2**2-z**2)+9*r1**5*r2*(r2**2-z**2)+
                  12*r1**4*r2**2*(r2**2-z**2)+r1**3*r2*(9*r2**4-12*r2**2*z**2+
                  4*((x1-x2)**2+(y1-y2)**2)*z**2))),
                (y1-y2)*(l2**2*(r1**2*r2**5*y1-3*r2**5*y1*z**2+
                  r1**5*y2*(r2**2-3*z**2))-2*l2*(r1+r2)**3*(-3*r2**4*y1*z**2+
                  r1**4*y2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*y1-(y1+y2)*z**2))+
                  (r1+r2)**2*(-12*r1*r2**6*y1*z**2-3*r2**7*y1*z**2+
                  r1**7*y2*(r2**2-3*z**2)+4*r1**6*r2*y2*(r2**2-3*z**2)+
                  2*r1**3*r2**2*(2*r2**4*y1-3*((x1-x2)**2+
                  (y1-y2)**2)*y2*z**2-4*r2**2*(2*y1+y2)*z**2)+
                  r1**4*r2**3*(r2**2*(5*y1+2*y2)-8*(y1+2*y2)*z**2)+
                  r1**2*r2**3*(r2**4*y1-6*y1*((x1-x2)**2+
                  (y1-y2)**2)*z**2-r2**2*(19*y1+2*y2)*z**2)+
                  r1**5*r2**2*(r2**2*(2*y1+5*y2)-(2*y1+19*y2)*z**2)))
                ])/((r1*r2)**5*((r1+r2)**2-l2)**3)).sum(axis=1)/np.pi
