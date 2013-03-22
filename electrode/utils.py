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
            [c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], -c[3]-c[0]]
            ])
    elif order == 7: # xxy xxz yyz yyx zzx zzy xyz
        return np.array([
            [[-c[3]-c[4], c[0], c[1]], [c[0], c[3], c[6]], [c[1], c[6], c[4]]],
            [[c[0], c[3], c[6]], [c[3], -c[0]-c[5], c[2]], [c[6], c[2], c[5]]],
            [[c[1], c[6], c[4]], [c[6], c[2], c[5]], [c[4], c[5], -c[1]-c[2]]],
            ])
    elif order == 9: # xxxy xxxz xxyy xxzz xyyy xzzz yyyz yyzz yzzz
        return np.array([
            [[[-c[2]-c[3], c[0], c[1]], [c[0], c[2], -c[6]-c[8]], [c[1], -c[6]-c[8], c[3]]], 
              [[c[0], c[2], -c[6]-c[8]], [c[2], c[4], -c[1]-c[5]], [-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]]], 
              [[c[1], -c[6]-c[8], c[3]], [-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]], [c[3], -c[0]-c[4], c[5]]]], 
            [[[c[0], c[2], -c[6]-c[8]], [c[2], c[4], -c[1]-c[5]], [-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]]], 
              [[c[2], c[4], -c[1]-c[5]], [c[4], -c[2]-c[7], c[6]], [-c[1]-c[5], c[6], c[7]]], 
              [[-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]], [-c[1]-c[5], c[6], c[7]], [-c[0]-c[4], c[7], c[8]]]], 
            [[[c[1], -c[6]-c[8], c[3]], [-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]], [c[3], -c[0]-c[4], c[5]]], 
              [[-c[6]-c[8], -c[1]-c[5], -c[0]-c[4]], [-c[1]-c[5], c[6], c[7]], [-c[0]-c[4], c[7], c[8]]], 
              [[c[3], -c[0]-c[4], c[5]], [-c[0]-c[4], c[7], c[8]], [c[5], c[8], -c[3]-c[7]]]]
            ])
    elif order == 11: # xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyz xyyzz yyyzz yyzzz
        return np.array([
            [[[[-c[0]-c[2], -c[3]-c[5], -c[4]-c[6]], [-c[3] -c[5], c[0], c[1]], [-c[4]-c[6], c[1], c[2]]],
              [[-c[3]-c[5], c[0], c[1]], [c[0], c[3], c[4]], [c[1], c[4], c[5]]],
              [[-c[4]-c[6], c[1], c[2]], [c[1], c[4], c[5]], [c[2], c[5], c[6]]]],
             [[[-c[3]-c[5], c[0], c[1]], [c[0], c[3], c[4]], [c[1], c[4], c[5]]],
              [[c[0], c[3], c[4]], [c[3], -c[0]-c[8], c[7]], [c[4], c[7], c[8]]],
              [[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]]],
             [[[-c[4]-c[6], c[1], c[2]], [c[1], c[4], c[5]], [c[2], c[5], c[6]]],
              [[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]],
              [[c[2], c[5], c[6]], [c[5], c[8], -c[1]-c[7]], [c[6], -c[1]-c[7], -c[2]-c[8]]]]],
            [[[[-c[3]-c[5], c[0], c[1]], [c[0], c[3], c[4]], [c[1], c[4], c[5]]],
              [[c[0], c[3], c[4]], [c[3], -c[0]-c[8], c[7]], [c[4], c[7], c[8]]],
              [[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]]],
             [[[c[0], c[3], c[4]], [c[3], -c[0]-c[8], c[7]], [c[4], c[7], c[8]]],
              [[c[3], -c[0]-c[8], c[7]], [-c[0]-c[8], -c[3]-c[9], -c[4]-c[10]], [c[7], -c[4]-c[10], c[9]]],
              [[c[4], c[7], c[8]], [c[7], -c[4]-c[10], c[9]], [c[8], c[9], c[10]]]],
             [[[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]],
              [[c[4], c[7], c[8]], [c[7], -c[4]-c[10], c[9]], [c[8], c[9], c[10]]],
              [[c[5], c[8], -c[1]-c[7]], [c[8], c[9], c[10]], [-c[1]-c[7], c[10], -c[5]-c[9]]]]],
            [[[[-c[4]-c[6], c[1], c[2]], [c[1], c[4], c[5]], [c[2], c[5], c[6]]],
              [[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]],
              [[c[2], c[5], c[6]], [c[5], c[8], -c[1]-c[7]], [c[6], -c[1]-c[7], -c[2]-c[8]]]],
             [[[c[1], c[4], c[5]], [c[4], c[7], c[8]], [c[5], c[8], -c[1]-c[7]]],
              [[c[4], c[7], c[8]], [c[7], -c[4]-c[10], c[9]], [c[8], c[9], c[10]]],
              [[c[5], c[8], -c[1]-c[7]], [c[8], c[9], c[10]], [-c[1]-c[7], c[10], -c[5]-c[9]]]],
             [[[c[2], c[5], c[6]], [c[5], c[8], -c[1]-c[7]], [c[6], -c[1]-c[7], -c[2]-c[8]]],
              [[c[5], c[8], -c[1]-c[7]], [c[8], c[9], c[10]], [-c[1]-c[7], c[10], -c[5]-c[9]]],
              [[c[6], -c[1]-c[7], -c[2]-c[8]], [-c[1]-c[7], c[10], -c[5]-c[9]], [-c[2]-c[8], -c[5]-c[9], -c[6]-c[10]]]]]
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
    elif order == 3**2: # xx xy xz yy yz
        return c[(0, 1, 2, 4, 5), :]
    elif order == 3**3: # xxy xxz yyz yyx zzx zzy xyz
        return c[(1, 2, 14, 4, 8, 17, 5), :]
    elif order == 3**4: # xxxy xxxz xxyy xxzz xyyy xzzz yyyz yyzz yzzz
        return c[(1, 2, 4, 8, 13, 26, 41, 44, 53), :]
    elif order == 3**5: # xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyz xyyzz yyyzz yyzzz
        return c[(4, 5, 8, 13, 14, 17, 26, 41, 44, 125, 134), :]


def cartesian_to_spherical_harmonics(c):
    """given a cartesian derivative of a harmonic potential where the
    derivative index is the first dimension (reduced as per
    select_tensor, expand_tensor), rewrite it in terms of real 
    spherical harmonics where m (-l...l) is the first dimension. l is
    inferred from the input shape. 
    Convention and conversion to complex spherical harmonics as per
    http://theoretical-physics.net/dev/src/math/operators.html#real-spherical-harmonics
    """
    c = np.atleast_2d(c)
    l = (c.shape[0] - 1)/2
    if l == 0:
        return np.sqrt(1/(4*np.pi))*c
    elif l == 1:
        x, y, z = c
        c = np.array([-x, z, -y])
        return np.sqrt(3/(4*np.pi))*c
    elif l == 2:
        xx, xy, xz, yy, yz = c
        c = np.array([])
        return np.sqrt(5/(16*np.pi))*c
    elif l == 3:
        xxy, xxz, yyz, yyx, zzx, zzy, xyz = c
        c = np.array([])
        return np.sqrt(7/(16*np.pi))*c
    elif l == 4:
        xxxy, xxxz, xxyy, xxzz, xyyy, xzzz, yyyz, yyzz, yzzz = c
        c = np.array([])
        return np.sqrt(5/(8*np.pi))*c
    elif l == 5:
        xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz, xyyyz, xyyzz, \
            yyyzz, yyzzz = c
        c = np.array([])
        return np.sqrt(5/(8*np.pi))*c


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
    b.shape == (2*r+1, 2, n, 2*n*(2*r+1))
    with indices: frequency component (-r...r), derivative, dimension, eigenvalue
    mu.shape == (2*n*(2*r+1),)
    b[..., i] the eigenvector to the eigenvalue mu[i]
    the eigenvalues and eigenvectors are not necessarily ordered
    (see numpy.linalg.eig())
    """
    n = a[0].shape[0]
    m = np.zeros((2*r+1, 2*r+1, 2, 2, n, n), dtype=np.complex)
    for l in range(2*r+1):
        # derivative on the diagonal
        m[l, l, 0, 0] = m[l, l, 1, 1] = np.identity(n)*2j*(l-r)
        # the off-diagonal 1st-1st derivative link
        m[l, l, 0, 1] = np.identity(n)
        # a_0, a_1... on the 2nd-0th component
        # fill to the right and below instead of left and right and left
        for i, ai in enumerate(a):
            if l+i < 2*r+1: # cutoff
                # i=0 (a_0) written twice (no factor of two in diff eq)
                m[l, l+i, 1, 0] = -ai
                m[l+i, l, 1, 0] = -ai
    # fold frequency components, derivative index and dimensions into
    # one axis each
    m = m.transpose((0, 2, 4, 1, 3, 5)).reshape((2*r+1)*2*n, -1)
    mu, b = np.linalg.eig(m)
    # b = b.reshape((2*r+1, 2, n, -1))
    return mu, b


class DummyPool(object):
    def apply_async(self, func, args=(), kwargs={}):
        class _DummyRet(object):
            def get(self):
                return func(*args, **kwargs)
        return _DummyRet()
