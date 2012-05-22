# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
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


def apply_method(s, name, *args, **kwargs):
    """small helper to work around non-picklable
    instance methods and allow them to be called by multiprocessing
    tools"""
    return getattr(s, name)(*args, **kwargs)


def norm(a, axis=-1):
    """special version of np.linalg.norm() that only covers the
    specified axis"""
    return np.sqrt(np.square(a).sum(axis=axis))


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
        x'' + (2 a_0 + 2 a_1 cos(2 t) + 2 a_2 cos(4 t) ... ) x = 0
    in n dimensions and with a frequency cutoff at +- r
    a.shape == (n, n)
    returns mu eigenvalues and b eigenvectors
    mu.shape == (2*n*(2*r+1),) # duplicates for initial phase freedom
    b.shape == (2*n*(2*r+1), 2*n*(2*r+1))
    b[:, i] the eigenvector to the eigenvalue mu[i]
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


class DummyPool(object):
    def apply_async(self, func, args=(), kwargs={}):
        class _DummyRet(object):
            def get(self):
                return func(*args, **kwargs)
        return _DummyRet()
