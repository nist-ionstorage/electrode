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


import itertools, sys, warnings

import numpy as np
import pylab as pl
import matplotlib as mpl
from scipy import optimize, ndimage, constants as ct

from traits.api import (HasTraits, Array, Float, Int, Str,
        Instance, List, Bool, Property, Trait, Enum)

from transformations import euler_from_matrix
import saddle

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)

try:
    from qc.theory.gni import gni
except ImportError:
    warnings.warn("qc modules not found, some stuff will fail", ImportWarning)



class _DummyPool(object):
    def apply_async(self, func, args=(), kwargs={}):
        class _DummyRet(object):
            def get(self):
                return func(*args, **kwargs)
        return _DummyRet()

dummy_pool = _DummyPool()


def apply_method(s, name, *args, **kwargs):
    """small helper to work around non-picklable
    instance methods and allow them to be called by multiprocessing
    tools"""
    return getattr(s, name)(*args, **kwargs)


def norm(a, axis=-1):
    return np.sqrt(np.square(a).sum(axis=axis))

def dot(a, b, axis=-1):
    return np.multiply(a, b).sum(axis=axis)

def triple(a, b, c, axis=-1):
    return dot(a, np.cross(b, c, axis=axis), axis=axis)


def wrap_x_loop(f):
    """return a version of f which internally does not vectorize over
    the first dimension of the first argument
    for speed and memory testing"""
    def x_loop(x, *a):
        return np.array([f(xi[None, :], *a) for xi in x])
    return x_loop

#@wrap_x_loop
def field(X, P):
    """
    Biot-Savart type integral.
    Returns field (a N,3 array) at points X (a N,3 array
    of points) given the piecewise linear path P (a M,3 array of
    points on the path, subsequent points denote edges)

    For surface electrodes this is the field at X per volt on the
    electrode described by the path P (counterclockwise from above, the
    rest of the plane is grounded).

    N*M in time and memory

    Electrostatics of surface-electrode ion traps
    J. H. Wesenberg, Phys Rev A 78, 063410 2008

    (and others)

    Compact expressions for the Biot-Savart fields of a filamentary
    segment. Hanson, James D. Hirshman, Steven P.
    Physics of Plasmas, Volume 9, Issue 10, pp. 4410-4412 (2002).
    """
    D = np.roll(P, -1, axis=0) - P # line element vectors
    l = norm(D) # line element lengths
    E = D/l[:, None] # normalized line elements
    I = X-P[:, None] # from X to initial point on line elements
    A = np.cross(E[:, None], I) # e cross Ri
    i = norm(I) # lengths of Ri
    f = np.roll(i, -1, axis=0) # lengths of Rf
    eps = l[:, None]/(i+f) # eccentricity of Ri,Rf ellipse through X
    b = eps/(1-eps**2)/(i*f) # fprime(eps)/(ri+rf)/2
    return (A*b[:, :, None]).sum(axis=0)/np.pi

#@wrap_x_loop
def potential(X, P):
    """
    Returns potential (a N, array) at points X (a N,3 array
    of points) given the piecewise linear path P (a M,3 array of
    points on the path)

    For surface electrodes this is the electrostatic potential at X per
    volt on the electrode described by the path P (counterclockwise
    from above, the rest of the plane is grounded).

    N*M in time and memory

    Van Oosterom, A.; Strackee, J.; "The Solid Angle of a Plane
    Triangle", IEEE Transactions on Biomedical Engineering,
    vol. BME-30, no.2, pp.125-126, Feb. 1983
    """
    # P1, P2, P3 are vectors from X to the corners of the patch triangles
    # P1 is arbitrarily taken to be the origin
    # p1, p2, p3 is their respective length
    P1 = P[:, None, :] - X[None, :, :]
    p1 = norm(P1)
    P2 = np.roll(P1, -1, axis=0)
    p2 = np.roll(p1, -1, axis=0)
    P3 = -X[None, :, :]
    # P3 += P.mean(axis=0)[None, None, :] # mean of corners
    p3 = norm(P3)
    n = triple(P1, P2, P3)
    d = p1*p2*p3 + dot(P1, P2)*p3 + dot(P1, P3)*p2 + dot(P2, P3)*p1
    return np.arctan2(n, d).sum(axis=0)/np.pi


def expand_tensor(c):
    """from the minimal linearly independent entries of a derivative of
    a harmonic field c build the complete tensor using its symmtry
    and laplace

    inverse of select_tensor()"""
    c = np.atleast_1d(c)
    if len(c.shape) == 1: # scalar
        return c
    order = c.shape[0]
    if order == 3:
        return c
    elif order == 5: # xx xy xz yy yz
        return np.array([
            [c[0], c[1], c[2]],
            [c[1], c[3], c[4]],
            [c[2], c[4], -c[3]-c[0]]])
    elif order == 7: # xxy xxz yyz yyx zzx zzy xyz
        return np.array([
            [[-c[3]-c[4], c[0], c[1]],
             [c[0], c[3], c[6]],
             [c[1], c[6], c[4]]],
            [[c[0], c[3], c[6]],
             [c[3], -c[0]-c[5], c[2]],
             [c[6], c[2], c[5]]],
            [[c[1], c[6], c[4]],
             [c[6], c[2], c[5]],
             [c[4], c[5], -c[1]-c[2]]],
            ])


def select_tensor(c):
    """select only a linealy idependent subset from a derivative of a
    harmonic field

    inverse of expand_tensor()"""
    c = np.atleast_1d(c)
    if len(c.shape) == 1: # scalar
        return c
    c = c.reshape((-1, c.shape[-1]))
    order = c.shape[0]
    if order == 3:
        return c
    elif order == 9: # xx xy xz yy yz
        return c[(0, 1, 2, 4, 5), :]
    elif order == 27: # xxy xxz yyz yyx zzx zzy xyz
        return c[(1, 2, 14, 4, 8, 17, 5), :]


def rotate_tensor(c, r, order=None):
    """rotate a tensor c into the coordinate system r
    assumes that its order is len(c.shape)-1
    the last dimension is used for parallelizing"""
    c = np.atleast_1d(c)
    r = np.atleast_2d(r)
    if order is None:
        order = len(c.shape)-1
    for i in range(order):
        c = np.dot(c.swapaxes(i, -1), r).swapaxes(i, -1)
    return c


def area_centroid(p1):
    """return the centroid and the area of the polygon p1
    (list of points)"""
    p2 = np.roll(p1, -1, axis=0)
    r = p1[:, 0]*p2[:, 1]-p2[:, 0]*p1[:, 1]
    a = r.sum(0)/2
    c = ((p1+p2)*r[:, None]).sum(0)/(6*a)
    return a, c


def mathieu(r, *a):
    """solve the mathieu/floquet equation 
        x'' + (a_0 + 2 a_1 cos(2 t) + 2 a_2 cos(4 t) ... ) x = 0
    in n dimensions and with a frequency cutoff at +- r
    a.shape == (n, n)
    returns mu eigenvalues and b eigenvectors
    mu.shape == (2*n*(2*r+1),) # duplicates for initial phase freedom
    b.shape == (2*n*(2*r+1), 2*n*(2*r+1))
        # b[:, i] the eigenvector to the eigenvalue mu[i]
    the lowest energy component is centered
    """
    n = a[0].shape[0]
    m = np.zeros((2*(2*r+1), 2*(2*r+1), n, n), dtype=np.complex)
    for l in range(2*r+1):
        m[2*l, 2*l] = np.identity(n)*2j*(l-r)
        m[2*l, 2*l+1] = np.identity(n)
        m[2*l+1, 2*l+1] = np.identity(n)*2j*(l-r)
        for i, ai in enumerate(a):
            if l <= 2*r+1-2*i:
                m[2*l+1, 2*l+2*i] = -ai
                m[2*l+1+2*i, 2*l] = -ai
    m = m.transpose((0, 2, 1, 3)).reshape((2*r+1)*2*n, (2*r+1)*2*n)
    mu, b = np.linalg.eig(m)
    return mu, b


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
        x = np.atleast_2d(x)
        r = [0.] * len(d)
        for n in range(-self.nmax, self.nmax+1):
            xx = x + [[0., 0., 2.*n*self.cover_height]]
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
                edgecolors="none", cmap=pl.cm.binary,
                norm=pl.Normalize(0, 1.),
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
        a = self.areas[:, None]
        p1 = x[None, :] - self.points[:, None]
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


class PolygonPixelElectrode(PixelElectrode):
    paths = List(Array(dtype=np.float64, shape=(None, 3)))

    def _pixel_factors_default(self):
        return np.ones(len(self.paths))

    def orientations(self):
        p, = self.value_no_cover(np.array([[0, 0, 1.]]), 0)
        return np.sign(p[:, 0])

    def plot(self, ax, text=None, alpha=1., *a, **kw):
        if text is None:
            text = self.name
        for vi, p in zip(self.pixel_factors, self.paths):
            ax.fill(p[:,0], p[:,1], edgecolor='none',
                    alpha=alpha*vi, *a, **kw)
            if text:
                ax.text(p[:,0].mean(), p[:,1].mean(), text)

    def to_points(self):
        a, c = zip(*(area_centroid(p) for p in self.paths))
        return PointPixelElectrode(name=self.name,
                pixel_factors=self.pixel_factors, nmax=self.nmax,
                cover_height=self.cover_height, areas=a, points=c)

    def value_no_cover(self, x, *d):
        v = [list(self.polygon_value(x, p, *d)) for p in self.paths]
        for vi in zip(*v):
            vi = np.array(vi)
            if len(vi.shape) > 2:
                vi = vi.transpose((1, 0, 2))
            yield vi

    def polygon_value(self, x, p, *d):
        p1 = x[None, :] - p[:, None]
        x1, y1, z = p1.transpose((2, 0, 1))
        r1 = norm(p1)
        x2 = np.roll(x1, -1, axis=0)
        y2 = np.roll(y1, -1, axis=0)
        r2 = np.roll(r1, -1, axis=0)
        l2 = (x1-x2)**2+(y1-y2)**2
        if 0 in d:
            #yield np.angle((x1-1j*y1)*(l2-(x1+1j*y1)*((x1-x2)-1j*(y1-y2)))*
            #    ((x1-x2)+1j*(y1-y2))*(1j*r1*(x1*y2-x2*y1)+
            #    (x1*(x1-x2)+y1*(y1-y2))*z)*(1j*r2*(x1*y2-x2*y1)+
            #    (l2-x1*(x1-x2)-y1*(y1-y2))*z)).sum(axis=0)/(2*np.pi)
            zs = np.abs(z)
            yield np.arctan2(z*(x1*y2-y1*x2),
                    zs*(r1*r2+x1*x2+y1*y2+zs*(zs+r1+r2))).sum(axis=0)/np.pi
        if 1 in d:
            yield (np.array([-(y1-y2)*z, (x1-x2)*z, x2*y1-x1*y2]
                )*(r1+r2)/(np.pi*r1*r2*((r1+r2)**2-l2))).sum(axis=1)
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
                ])/(np.pi*(r1*r2)**3*((r1+r2)**2-l2)**2)).sum(axis=1)
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
                ])/(np.pi*(r1*r2)**5*((r1+r2)**2-l2)**3)).sum(axis=1)


class PatternConstraint(HasTraits):
    def objective(self, electrode, variables):
        return
        yield

    def constraints(self, electrode, variables):
        return
        yield


class PatternValueConstraint(PatternConstraint):
    x = Array(dtype=np.float64, shape=(3,))
    d = Int
    v = Array(dtype=np.float64)
    r = Array(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    def objective(self, electrode, variables):
        v = select_tensor(self.v[..., None]).ravel()
        c, = electrode.value(self.x, self.d)
        c = expand_tensor(c[..., 0])
        c = np.array(rotate_tensor(c, self.r))
        c = select_tensor(c).reshape((v.shape[0], -1))
        return zip(c, v)


class PatternRangeConstraint(PatternConstraint):
    min = Float
    max = Float

    def constraints(self, electrode, variables):
        if self.min is not None:
            yield variables >= self.min
        if self.max is not None:
            yield variables <= self.max


class System(HasTraits):
    electrodes = List(Instance(Electrode))
    names = Property()
    voltages = Property()

    def _get_names(self):
        return [el.name for el in self.electrodes]

    def _get_voltages(self):
        return [(el.voltage_dc, el.voltage_rf)
                for el in self.electrodes]

    def _set_voltages(self, voltages):
        for el, (v_dc, v_rf) in zip(self.electrodes, voltages):
            el.voltage_dc = v_dc
            el.voltage_rf = v_rf

    def electrode(self, name):
        """return the first electrode named name or None if not found"""
        return self.electrodes[self.names.index(name)]

    def with_voltages(self, voltages, function, *args, **kwargs):
        """execute function(*args, **kwargs) with voltages set to voltages"""
        voltages, self.voltages = self.voltages, voltages
        ret = function(*args, **kwargs)
        self.voltages = voltages
        return ret

    def get_potentials(self, x, typ, *d):
        x = np.atleast_2d(x)
        n = x.shape[0]
        e = [np.zeros((n)), np.zeros((3,n)), np.zeros((3,3,n)),
                np.zeros((3,3,3,n))]
        for el in self.electrodes:
            v = getattr(el, "voltage_"+typ)
            if v != 0:
                for di, pi in zip(d, el.potential(x, *d)):
                    e[di] += v*pi
        return [e[di] for di in d]

    def field_dc(self, x):
        return self.get_potentials(x, "dc", 1)[0]

    def field_rf(self, x, t=0.):
        return np.cos(t)*self.get_potentials(x, "rf", 1)[0]

    def field(self, x, t=0.):
        """electrical field at an instant in time t (physical units:
        1/omega_rf), no adiabatic approximation here"""
        return self.field_dc(x) + self.field_rf(x, t)

    def potential_dc(self, x):
        return self.get_potentials(x, "dc", 0)[0]

    def potential_rf(self, x):
        e_rf, = self.get_potentials(x, "rf", 1)
        return (e_rf**2).sum(axis=0)

    def potential(self, x):
        """combined electrical and ponderomotive potential"""
        return self.potential_dc(x) + self.potential_rf(x)

    def gradient_dc(self, x):
        return self.get_potentials(x, "dc", 1)[0]

    def gradient_rf(self, x):
        e_rf, c_rf = self.get_potentials(x, "rf", 1, 2)
        return 2*(e_rf[:, None]*c_rf).sum(axis=0)

    def gradient(self, x):
        """gradient of the combined electric and ponderomotive
        potential"""
        return self.gradient_dc(x) + self.gradient_rf(x)

    def curvature_dc(self, x):
        return self.get_potentials(x, "dc", 2)[0]

    def curvature_rf(self, x):
        e_rf, c_rf, d_rf = self.get_potentials(x, "rf", 1, 2, 3)
        return 2*(c_rf[:, :, None]*c_rf[:, None, :]+
                e_rf[:, None, None]*d_rf).sum(axis=0)

    def curvature(self, x):
        """curvature of the combined electric and ponderomotive
        potential"""
        return self.curvature_dc(x) + self.curvature_rf(x)

    def parallel(self, pool, x, y, z, fn="potential"):
        """paralelize the calculation of method fn over the indices of
        the x axis"""
        r = [pool.apply_async(apply_method, (self, fn,
                np.array([x[i], y[i], z[i]]).reshape(3, -1).T))
            for i in range(x.shape[0])]
        r = np.array([ri.get().reshape((x.shape[1], x.shape[2]))
            for ri in r])
        return r

    def plot(self, ax, *a, **k):
        """plot electrodes with random colors"""
        for e, c in zip(self.electrodes, itertools.cycle("bgrcmy")):
            e.plot(ax, color=c, alpha=.5, *a, **k)

    def plot_voltages(self, ax, u=None, els=None):
        """plot electrodes with alpha proportional to voltage (scaled to
        max abs voltage being opaque), red for positive, blue for
        negative"""
        if els is None:
            els = self.electrodes
        if u is None:
            u = np.array([el.voltage_dc for el in els])
        um = abs(u).max() or 1.
        for el, ui in zip(els, u):
            el.plot(ax, color=(ui > 0 and "red" or "blue"),
                    alpha=abs(ui)/um, text="")

    def minimum(self, x0, axis=(0, 1, 2), coord=np.identity(3)):
        """find a potential minimum near x0 searching along the
        specified axes in the orthonormal matrix coord"""
        x = np.array(x0)
        def p(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return float(self.potential(np.dot(coord, x)))
        # downhill simplex
        xs = optimize.fmin(p, np.array(x0)[:, axis], disp=False)
        for i, ai in enumerate(axis):
            x[ai] = xs[i]
        return x

    def saddle(self, x0, axis=(0, 1, 2), coord=np.identity(3), **kw):
        """find a saddle point close to x0 along the specified axes in
        the coordinate system coord"""
        kwargs = dict(dx_max=.1, xtol=1e-5, ftol=1e-5)
        kwargs.update(kw)
        x = np.array(x0)
        def f(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return float(self.potential(np.dot(coord, x)))
        def g(xi):
            for i, ai in enumerate(axis):
                x[ai] = xi[i]
            return rotate_tensor(self.gradient(np.dot(coord, x)),
                    coord.T)[axis, 0]
        h = rotate_tensor(self.curvature(np.dot(coord, x)),
                coord.T)[axis, :][:, axis, 0]
        # rational function optimization
        xs, p, ret = saddle.rfo(f, g, np.array(x0)[:, axis], h=h, **kwargs)
        if not ret in ("ftol", "xtol"):
            raise ValueError, (x0, axis, x, xs, p, ret)
        # f(xs) # update x
        return x, p

    def modes(self, x, sorted=True):
        """returns curvatures and eigenmode vectors at x
        physical units of the trap frequenzies (Hz):
        scale = (q*u/(omega*scale))**2/(4*m)
        (scale*ew/scale**2/m)**.5/(2*pi)
        """
        ew, ev = np.linalg.eigh(self.curvature(x)[:, :, 0])
        if sorted:
            i = ew.argsort()
            ew, ev = ew[i], ev[:, i]
        return ew, ev

    def trajectory(self, x0, v0, axis=(0, 1, 2),
            t0=0, dt=.0063*2*np.pi, t1=1e4, nsteps=1,
            integ="gni_irk2", methc=2, field=None, *args, **kwargs):
        """integrate the trajectory with initial position and speed x0,
        v0 along the specified axes (use symmetry to eliminate one),
        time step dt, from time t0 to time t1, yielding current position
        and speed every nsteps time steps"""
        integ = getattr(gni, integ)
        if field is None:
            field = self.field
        t, p, q = t0, v0[:, axis], x0[:, axis]
        x0 = np.array(x0)
        def ddx(t, q, f):
            for i, ai in enumerate(axis):
                x0[ai] = q[i]
            f[:len(axis)] = field(x0, t)[axis, 0]
        while t < t1:
            integ(ddx, nsteps, t, p, q, t+dt, methc, *args, **kwargs)
            t += dt
            yield t, q.copy(), p.copy()

    def stability(self, maxx, *a, **k):
        """integrate the trajectory (see :meth:trajectory()) as long as
        the position remains within a maxx ball or t>t1"""
        t, q, p = [], [], []
        for ti, qi, pi in self.trajectory(*a, **k):
            t.append(ti), q.append(qi), p.append(pi)
            if (qi**2).sum() > maxx**2:
                break
        return map(np.array, (t, q, p))

    def effects(self, x, electrodes=None, pool=dummy_pool):
        """
        return potential, gradient and curvature for the system at x and
        contribution of each of the specified electrodes per volt

        O(len(electrodes)*len(x))
        """
        if electrodes is None:
            electrodes = self.electrodes
        x = np.atleast_2d(x) # n,3 positions
        r = [pool.apply_async(apply_method,
            (el, "potential", x, 0, 1, 2)) for el in electrodes]
        p, f, c = zip(*map(lambda i: i.get(), r))
        # m,n = len(electrodes), len(x)
        # m,n potentials for electrodes at x0
        p = np.array(p)
        # 3,m,n forces for electrodes at x0
        f = np.array(f).transpose((1, 0, 2))
        # 3,3,m,n curvatures for electrodes at x0
        c = np.array(c).transpose((1, 2, 0, 3))
        return p, f, c

    def shims(self, x, electrodes=None, forces=None, curvatures=None,
            coords=None, rcond=1e-3):
        """
        solve the shim equations simultaneously at all points x0,
        return an array of voltages (x, y, z, xx, xy, yy, xz, yz) *
        len(x0). -zz=xx+yy is dropped.
        """
        x = np.atleast_2d(x)
        if electrodes is None:
            electrodes = self.electrodes
        p, f, c = self.effects(x, electrodes)
        f = f.transpose((-1, 0, 1)) # x,fi,el
        c = c.transpose((-1, 0, 1, 2)) # x,ci,cj,el
        dc = []
        for i, (fi, ci) in enumerate(zip(f, c)): # over x
            if coords is not None:
                fi = rotate_tensor(fi, coords[i], 1)
                ci = rotate_tensor(ci, coords[i], 2)
            ci = select_tensor(ci)
            if forces is not None:
                fi = fi[forces[i], ...]
            if curvatures is not None:
                ci = ci[curvatures[i], ...]
            dc.extend(fi)
            dc.extend(ci)
        ev = np.identity(len(dc))
        u, res, rank, sing = np.linalg.lstsq(dc, ev, rcond=rcond)
        return u, (res, rank, sing)

    def solve(self, x, constraints, 
            electrodes=None, verbose=True, pool=dummy_pool):
        """
        optimize dc voltages at positions x to satisfy constraints.

        O(len(constraints)*len(x)*len(electrodes)) if sparse (most of the time)
        """
        if electrodes is None:
            electrodes = self.electrodes
        v0 = [el.voltage_dc for el in electrodes]
        for el in electrodes:
            el.voltage_dc = 0.
        p0, f0, c0 = self.potential(x), self.gradient(x), self.curvature(x)
        for el, vi in zip(electrodes, v0):
            el.voltage_dc = vi
        p, f, c = self.effects(x, electrodes, pool)

        variables = []
        pots = []
        for i, xi in enumerate(x):
            v = cvxopt.modeling.variable(len(electrodes), "u%i" % i)
            v.value = cvxopt.matrix(
                    [float(el.voltage_dc) for el in electrodes])
            variables.append(v)
            pots.append((p0[i], f0[:, i], c0[:, :, i],
                p[:, i], f[:, :, i], c[:, :, :, i]))

        # setup constraint equations
        obj = 0.
        ctrs = []
        for ci in constraints:
            obj += sum(ci.objective(self, electrodes, x, variables, pots))
            ctrs.extend(ci.constraints(self, electrodes, x, variables, pots))
        solver = cvxopt.modeling.op(obj, ctrs)

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

        u = np.array([np.array(v.value).ravel() for v in variables])
        p = np.array([p0i+np.dot(pi, ui)
            for p0i, ui, pi in zip(p0,
                u, p.transpose(-1, -2))])
        f = np.array([f0i+np.dot(fi, ui)
            for f0i, ui, fi in zip(f0.transpose(-1, 0),
                u, f.transpose(-1, 0, -2))])
        c = np.array([c0i+np.dot(ci, ui)
            for c0i, ui, ci in zip(c0.transpose(-1, 0, 1),
                u, c.transpose(-1, 0, 1, -2))])
        return u, p, f, c

    def mathieu(self, x, u_dc, u_rf, r=2, sorted=True):
        """return characteristic exponents (mode frequencies) and
        fourier components"""
        c_rf, = self.get_potentials(x, "rf", 2)
        c_dc, = self.get_potentials(x, "dc", 2)
        a = 4*u_dc*c_dc[..., 0]
        q = 2*u_rf*c_rf[..., 0]
        mu, b = mathieu(r, a, q)
        if sorted:
            i = mu.imag >= 0
            mu, b = mu[i], b[:, i]
            i = mu.imag.argsort()
            mu, b = mu[i], b[:, i]
        return mu/2, b
    
    def analyze_static(self, x, axis=(0, 1, 2), do_ions=False,
            m=1., q=1., u=1., l=1., o=1.):
        scale = (u*q/l/o)**2/(4*m) # rf pseudopotential energy scale
        dc_scale = scale/q # dc energy scale
        yield "u = %.3g V, f = %.3g MHz, m = %.3g amu, "\
                 "q = %.3g qe, l = %.3g µm, axis=%s" % (
                u, o/(2e6*np.pi), m/ct.atomic_mass,
                q/ct.elementary_charge, l/1e-6, axis)
        yield "analyze point: %s (%s µm)" % (x, x*l/1e-6)
        trap = self.minimum(x, axis=axis)
        yield " minimum is at offset: %s" % (trap - x)
        p_rf, p_dc = self.potential_rf(x), self.potential_dc(x)
        yield " rf, dc potentials: %.2g, %.2g (%.2g eV, %.2g eV)" % (
            p_rf, p_dc, p_rf*scale/q, p_dc*dc_scale)
        try:
            xs, xsp = self.saddle(x+1e-2, axis=axis)
            yield " saddle offset, height: %s, %.2g (%.2g eV)" % (
                xs - x, xsp - p_rf - p_dc, (xsp - p_rf - p_dc)*scale/q)
        except:
            yield " saddle not found"
        curves, modes_pp = self.modes(x)
        freqs_pp = (scale*curves/l**2/m)**.5/(2*np.pi)
        q_o2l2m = q/((l*o)**2*m)
        mu, b = self.mathieu(x, u_dc=dc_scale*q_o2l2m, u_rf=u*q_o2l2m,
                r=3, sorted=True)
        freqs = mu[:3].imag*o/(2*np.pi)
        modes = b[len(b)/2-3:len(b)/2, :3].real
        yield " pp+dc normal curvatures: %s" % curves
        yield " motion is bounded: %s" % np.allclose(0, mu.real)
        for nj, fj, mj in (("pseudopotential", freqs_pp, modes_pp),
                ("mathieu", freqs, modes)):
            yield " %s modes:" % nj
            for fi, mi in zip(fj, mj.T):
                yield "  %.4g MHz, %s" % (fi/1e6, mi)
            yield "  euler angles: %s" % (
                    np.array(euler_from_matrix(mj, "rxyz"))*180/np.pi)
        se = sum(list(el.potential(x, 1))[0][:, 0]**2
                for el in self.electrodes)/l**2
        ee = sum(list(el.potential(x, 0))[0][0]**2
                for el in self.electrodes)
        yield " heating for 1 nV²/Hz white on each electrode:"
        yield "  field-noise psd: %s V²/(m² Hz)" % (se*1e-9**2)
        yield "  pot-noise psd: %s V²/Hz" % (ee*1e-9**2)
        for fi, mi in zip(freqs_pp, modes_pp.T):
            sej = (np.abs(mi)*se**.5).sum()**2
            ndot = sej*q**2/(4*m*ct.h*fi)
            yield "  %.4g MHz: ndot=%.4g/s, S_E*f=%.4g (V² Hz)/(m² Hz)" % (
                fi/1e6, ndot*1e-9**2, sej*fi*1e-9**2)
        if do_ions:
            xi = x+np.random.randn(2)[:, None]*1e-3
            qi = np.ones(2)*q/(scale*l*4*np.pi*ct.epsilon_0)**.5
            xis, cis, mis = self.ions(xi, qi)
            freqs_ppi = (scale*cis/l**2/m)**.5/(1e6*np.pi)
            r2 = norm(xis[1]-xis[0])
            r2a = ((q*l)**2/(2*np.pi*ct.epsilon_0*scale*curves[0]))**(1/3.)
            yield " two ion modes:"
            yield "  separation: %.3g (%.3g µm, %.3g µm analytic)" % (
                r2, r2*l/1e-6, r2a/1e-6)
            for fi, mi in zip(freqs_ppi, mis.transpose(2, 0, 1)):
                yield "  %.4g MHz, %s/%s" % (fi/1e6, mi[0], mi[1])

    def analyze_shims(self, x, electrodes=None,
            forces=None, curvatures=None, use_modes=True):
        x = np.atleast_2d(x)
        if electrodes is None:
            electrodes = [e.name for e in self.electrodes
                    if e.voltage_rf == 0.]
        els = [self.electrode(n) for n in electrodes]
        if use_modes:
            coords = [self.modes(xi)[1] for xi in x]
        else:
            coords = None
        fn = "x y z".split()
        cn = "xx xy xz yy yz".split()
        if forces is None:
            forces = [fn] * len(x)
        if curvatures is None:
            curvatures = [cn] * len(x)
        fx = [[fn.index(fi) for fi in fj] for fj in forces]
        cx = [[cn.index(ci) for ci in cj] for cj in curvatures]
        us, (res, rank, sing) = self.shims(x, els, fx, cx, coords)
        yield "shim analysis for points: %s" % x
        yield " forces: %s" % forces
        yield " curvatures: %s" % curvatures
        yield " matrix shape: %s, rank: %i" % (us.shape, rank)
        yield " electrodes: %s" % np.array(electrodes)
        n = 0
        for i in range(len(x)):
            for ni in forces[i]+curvatures[i]:
                yield " sh_%i%-2s: %s" % (i, ni, us[:, n])
                n += 1
        yield forces, curvatures, us

    def ions(self, x0, q):
        """find the minimum energy configuration of several ions with
        normalized charges q and starting positions x0, return their
        equilibrium positions and the mode frequencies and vectors"""
        n = len(x0)
        qs = q[:, None]*q[None, :]

        def f(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = .5*qs/np.ma.array(
                    x**2+y**2+z**2)**(1/2.)
            return (p0+pi.sum(-1)).sum()

        def g(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.gradient(x0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = qs*[x, y, z]/np.ma.array(
                    x**2+y**2+z**2)**(3/2.)
            return (p0+pi.sum(-1)).T.ravel()

        def h(x0):
            x0 = x0.reshape(-1, 3)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            i, j = np.indices(x.shape)
            p0 = self.curvature(x0)
            p = expand_tensor(
                -qs*[2*x**2-y**2-z**2, 3*x*y, 3*x*z,
                    2*y**2-x**2-z**2, 3*y*z]/np.ma.array(
                    x**2+y**2+z**2)**(5/2.))
            p = p.transpose(2, 0, 3, 1)
            for i, (p0i, pii) in enumerate(
                    zip(p0.transpose(2, 0, 1), p.sum(2))):
                p[i, :, i, :] += p0i-pii
            return p.reshape(p.shape[0]*p.shape[1], -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            x, e0, itf, itg, ith, warn = optimize.fmin_ncg(
                f=f, fprime=g, fhess=h, x0=x0.ravel(), full_output=1,
                disp=0)
            #print warn
            #x1, e0, e1, e2, itf, itg, warn = optimize.fmin_bfgs(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x1)-np.sort(x))/np.sort(x)
            #x2, e0, itf, itg, warn = optimize.fmin_cg(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x2)-np.sort(x))/np.sort(x)
            c = h(x)
        ew, ev = np.linalg.eigh(c)
        i = np.argsort(ew)
        ew, ev = ew[i], ev[i, :].reshape(n, 3, -1)
        return x.reshape(-1, 3), ew, ev


class GridElectrode(Electrode):
    origin = Array(dtype=np.float64, shape=(3, ), value=(0, 0, 0))
    spacing = Array(dtype=np.float64, shape=(3, ), value=(1, 1, 1))
    data = Array(dtype=np.float64, shape=(None, None, None))
    voltage = Float(1.)
    delta = Int(1)

    def orientations(self):
        """dummy"""
        return np.array([1])

    def plot(self, ax, text=None, *a, **kw):
        """dummy, TODO: use geometry"""
        pass

    def electrical_potential(self, x):
        """return linearly interpolated potential"""
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        #x, y, z = np.floor(x).astype(np.int).T
        #p = self.data[x, y, z]
        p = ndimage.map_coordinates(self.data, x.T,
                order=1, mode="nearest")
        return self.voltage*p

    def electrical_gradient(self, x):
        """finite differences gradient around x"""
        k = self.delta
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        x, y, z = np.floor(x).astype(np.int).T
        dx = self.data[x+k, y, z] - self.data[x, y, z]
        dy = self.data[x, y+k, z] - self.data[x, y, z]
        dz = self.data[x, y, z+k] - self.data[x, y, z]
        dp = np.array([dx, dy, dz])/self.spacing/k
        return self.voltage*dp.T

    def electrical_curvature(self, x):
        """finite differences curvature around x"""
        k = self.delta
        x = (np.atleast_2d(x) - self.origin)/self.spacing
        x, y, z = np.round(x).astype(np.int).T
        c = self.data[x, y, z]
        cxx = self.data[x+k, y, z] + self.data[x-k, y, z] - 2*c
        cyy = self.data[x, y+k, z] + self.data[x, y-k, z] - 2*c
        czz = self.data[x, y, z+k] + self.data[x, y, z-k] - 2*c
        #assert np.allclose(cxx+.5*cyy, -czz-.5*cyy)
        cxy = self.data[x+k, y+k, z] + self.data[x-k, y-k, z] - (
            self.data[x-k, y+k, z] + self.data[x+k, y-k, z])
        cxz = self.data[x+k, y, z+k] + self.data[x-k, y, z-k] - (
            self.data[x-k, y, z+k] + self.data[x+k, y, z-k])
        cyz = self.data[x, y+k, z+k] + self.data[x, y-k, z-k] - (
            self.data[x, y-k, z+k] + self.data[x, y+k, z-k])
        cv = np.array([
            [cxx, cxy/4, cxz/4],
            [cxy/4, cyy, cyz/4],
            [cxz/4, cyz/4, czz]])/np.outer(self.spacing,
                    self.spacing)/k**2
        return self.voltage*cv.transpose(2, 0, 1)


class GridSystem(System):
    rf = List(Instance(GridElectrode))
    dc = List(Instance(GridElectrode))

    def potential(self, x):
        p = sum(e.electrical_potential(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return p

    def gradient(self, x):
        f = sum(e.electrical_gradient(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return f

    def curvature(self, x):
        c = sum(e.electrical_curvature(x) for e in self.dc + self.rf
                if e.voltage != 0)
        return c

    @classmethod
    def from_vtk(cls, geom, data, scale=1e-6):
        """load grid potential data from vti file "data" and geometry
        from "geom" and return a GridSystem instance, scale length
        units to scale"""
        from enthought.tvtk.api import tvtk
        o = cls()
        sgr = tvtk.XMLImageDataReader(file_name=data)
        sgr.update()
        sg = sgr.output
        for i in range(sg.point_data.number_of_arrays):
            name = sg.point_data.get_array_name(i)
            sp = sg.point_data.get_array(i)
            data = sp.to_array()
            spacing = sg.spacing
            origin = sg.origin
            dimensions = sg.dimensions
            if sp.number_of_components == 1:
                data = data.reshape(dimensions[::-1]).transpose(2, 1, 0)
            else:
                continue # ignore fields for now
                data = data.reshape(tuple(dimensions) +
                    (sp.number_of_components, ))
            if "_pondpot_1V1MHz1amu" in name:
                # convert to DC electrode equivalent potential
                data /= ct.elementary_charge**2/(4*ct.atomic_mass*(1e6*2*np.pi)**2
                        )/scale**2/ct.elementary_charge
                name = name[:-len("_pondpot_1V1MHz1amu")]
            else:
                data /= 1.
            el = GridElectrode(name=name, origin=origin/scale,
                    spacing=spacing/scale, data=data)
            o.electrodes.append(el)
        return o


class Constraint(HasTraits):
    weight = Float(1.) # if ==0, this becomes a constraint, else objective
    equal = Bool(True) # if True this constraint is == 0, else <= 0
    only_variable = Bool(False) # only take the variable parts into
                                # account, not the offset

    def values(self, system, electrodes, x, variables, pots):
        return
        yield

    def constraints(self, system, electrodes, x, variables, pots):
        if self.weight == 0:
            for v in self.values(system, electrodes, x, variables, pots):
                if self.equal:
                    yield v <= 0
                    yield v >= 0
                else:
                    yield v <= 0

    def objective(self, system, electrodes, x, variables, pots):
        if self.weight > 0:
            for v in self.values(system, electrodes, x, variables, pots):
                yield self.weight*cvxopt.modeling.sum(v)


class SingleIndexConstraint(Constraint):
    index = Trait(None, None, List(Int))

    def values(self, system, electrodes, x, variables, pots):
        if self.index is not None:
            idx = self.index
        else:
            idx = range(len(variables))
        for i in idx:
            for v in self.single_value(system, electrodes, x[i],
                    variables[i], pots[i]):
                yield v

    def single_value(self, system, electrodes, xi, variables, pots):
        return
        yield


class VoltageConstraint(Constraint):
    norm = Enum("none", "one", "inf") # norm to use across x
    delta = Int(1) # finite differences step size
    order = Int(0) # finite differences derivative order
    smooth = Bool(False) # flatten odd derivatives at the beginning and
                         # the end by mirror-extending them
    weight = 1.
    equal = False
    range = Trait(None, None, Float) # offset (use with weight=0,
                                         # equal=False, norm="none")
    variable = Trait(None, None, Int) # apply only to the given
                                      # electrode index

    def values(self, system, electrodes, x, variables, pots):
        if self.variable is not None:
            variables = [v[self.variable] for v in variables]
        if self.order == -1:
            obj = variables
        elif self.order == 0:
            obj = (abs(v) for v in variables)
        elif self.order == 1:
            obj = (abs(va-vb) for va, vb in
                    zip(variables[self.delta:],
                        variables[:-self.delta]))
        elif self.order == 2:
            obj = (abs(va-2*vb+vc) for va, vb, vc in
                    zip(variables[:-2*self.delta],
                        variables[self.delta:-self.delta],
                        variables[2*self.delta:]))
            if self.smooth:
                obj = tuple(obj) + (
                        abs(2*variables[0]-2*variables[self.delta]),
                        abs(2*variables[-1]-2*variables[-self.delta-1]))
        elif self.order == 3:
            obj = (abs(va-3*vb+3*vc-vd) for va, vb, vc, vd in
                    zip(variables[3*self.delta:],
                        variables[2*self.delta:-self.delta],
                        variables[1*self.delta:-2*self.delta],
                        variables[:-3*self.delta]))
        elif self.order == 4:
            obj = (abs(va-4*vb+6*vc-4*vd+ve) for va, vb, vc, vd, ve in
                    zip(variables[4*self.delta:],
                        variables[3*self.delta:-self.delta],
                        variables[2*self.delta:-2*self.delta],
                        variables[self.delta:-3*self.delta],
                        variables[:-4*self.delta]))
            if self.smooth:
                obj = tuple(obj) + (
                        abs(6*variables[0]-8*variables[self.delta]
                            +2*variables[2*self.delta]),
                        abs(6*variables[-1]-8*variables[-self.delta-1]
                            +2*variables[-2*self.delta-1]))
        for v in obj:
            if self.norm == "inf":
                v = cvxopt.modeling.max(v)
            elif self.norm == "one":
                v = cvxopt.modeling.sum(v)
            if self.range is not None:
                v = v - self.range
            yield v


class SymmetryConstraint(SingleIndexConstraint):
    symmetry = Array(dtype=np.int, shape=(None, 2)) # variable mapping
    weight = 0.
    equal = True

    def single_value(self, system, electrodes, x, variables, pots):
        for a, b in self.symmetry:
            va = a >= 0 and variables[int(a)] or 0
            vb = b >= 0 and variables[int(b)] or 0
            yield va - vb


class PotentialConstraint(SingleIndexConstraint):
    pmin = Trait(None, None, Float)
    pmax = Trait(None, None, Float)
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert p0.shape == (), p0.shape
        assert p.shape == (m,), p.shape
        if self.only_variable:
            p0 *= 0
        vi = p0 + cvxopt.modeling.dot(cvxopt.matrix(p.copy()), variables)
        if self.pmax is not None:
            yield vi - self.pmax
        if self.pmin is not None:
            yield self.pmin - vi


class ForceConstraint(SingleIndexConstraint):
    fmax = Trait(None, None, Array(shape=(3,), dtype=np.float64))
    coord = Trait(None, None, Array(shape=(3, 3), dtype=np.float64))
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert f0.shape == (3,), f0.shape
        assert f.shape == (3, m), f.shape
        if self.only_variable:
            f0 *= 0
        if self.coord is not None:
            f0 = rotate_tensor(f0, self.coord, order=1)
            f = rotate_tensor(f, self.coord, order=1)
        v = cvxopt.matrix(f0.copy()) + cvxopt.matrix(f.copy())*variables
        if self.fmax is not None:
            yield abs(v) - cvxopt.matrix(self.fmax)
        else:
            yield v


class CurvatureConstraint(SingleIndexConstraint):
    cmin = Trait(None, None, Array(shape=(6,), dtype=np.float64))
    cmax = Trait(None, None, Array(shape=(6,), dtype=np.float64))
    coord = Trait(None, None, Array(shape=(3, 3), dtype=np.float64))
    weight = 0.
    equal = False

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert c0.shape == (3, 3), c0.shape
        assert c.shape == (3, 3, m), c.shape
        if self.only_variable:
            c0 *= 0
        if self.coord is not None:
            c0 = rotate_tensor(c0, self.coord, order=2)
            c = rotate_tensor(c, self.coord)
        a, b = np.triu_indices(3)
        c0 = c0[a, b]
        c = c[a, b, :]
        lhs = cvxopt.matrix(c0.copy()) + cvxopt.matrix(c.copy())*variables
        if self.cmin is not None:
            yield cvxopt.matrix(self.cmin) - lhs
        if self.cmax is not None:
            yield lhs - cvxopt.matrix(self.cmax)


class OffsetPotentialConstraint(SingleIndexConstraint):
    pmin = Trait(None, None, Float)
    pmax = Trait(None, None, Float)
    weight = 0.
    equal = False
    reference = Array(shape=(3,), dtype=np.float64)

    def single_value(self, system, electrodes, x, variables, pots):
        m = variables._size
        p0, f0, c0, p, f, c = pots
        assert p0.shape == (), p0.shape
        assert p.shape == (m,), p.shape
        pref0 = system.potential(self.reference)[0]
        pref = []
        for el in electrodes:
            v0, el.voltage_dc = el.voltage_dc, 1.
            pref.append(el.electrical_potential(self.reference)[0])
            el.voltage_dc = v0
        pref = np.array(pref)
        if self.only_variable:
            p0 *= 0
            pref0 *= 0
        v = pref0 - p0 + cvxopt.modeling.dot(cvxopt.matrix(pref-p), variables)
        if self.pmin is not None:
            yield self.pmin - v
        if self.pmax is not None:
            yield v - self.pmax



