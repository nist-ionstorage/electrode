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


from math import factorial
from itertools import product

import numpy as np


def shaper(func, x, *args, **kwargs):
    y = x.reshape(-1, 3)
    v = func(y, *args, **kwargs)
    w = v.reshape(x.shape[:-1]+(v.shape[-1],))
    return w


def apply_method(s, name, *args, **kwargs):
    """small helper to work around non-picklable
    instance methods and allow them to be called by multiprocessing
    tools"""
    return getattr(s, name)(*args, **kwargs)


def norm(a, axis=-1):
    """special version of np.linalg.norm() that only covers the
    specified axis"""
    # return np.sqrt(np.einsum("...j,...j->...", a, a))
    return np.sqrt(np.square(a).sum(axis=axis))


def rotate_tensor(c, r, order=None):
    """rotate a tensor c into the coordinate system r
    assumes that its order is len(c.shape)-1
    the first dimension is used for parallelizing"""
    c = np.atleast_1d(c)
    r = np.atleast_2d(r)
    if order is None:
        order = len(c.shape)-1
    #slower: O(n**order)
    #ops = [c, range(order) + [Ellipsis]]
    #for i in range(order):
    #    ops.extend([r, [i, i+order]])
    #ops.append(range(order, 2*order) + [Ellipsis])
    #c = np.einsum(*ops)
    for i in range(order):
        #O(n*order):
        c = np.dot(c.swapaxes(i+1, -1), r).swapaxes(i+1, -1)
        #incorrect and probably not faster:
        #c = np.tensordot(c, r, axes=[i, 0])
    return c


derivative_names = [[""]] + [s.split() for s in [
    "x y z",
    "xx xy xz yy yz",
    "xxy xxz yyz xyy xzz yzz xyz",
    "xxxy xxxz xxyy xxzz xyyy xzzz yyyz yyzz yzzz",
    "xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyz xyyzz yyyzz yyzzz",
    ]]

laplace_map = {}
derivatives_map = {}
name_map = {}
expand_map = {}
select_map = {}

def name_to_idx(name):
    return tuple("xyz".index(n) for n in name)

def idx_to_name(idx):
    return "".join(sorted("xyz"[i] for i in idx))

def idx_to_nidx(idx):
    return sum(j*3**(len(idx)-i-1) for i, j in enumerate(idx))

def find_laplace(c):
    """given derivative name c returns the two derivatives a and b
    such that a+b+c=0 for a harmonic tensor"""
    name = sorted(list(c))
    letters = list("xyz")
    for i in letters:
        if name.count(i) >= 2:
            break
    k = name.index(i)
    del name[k:k+2], letters[letters.index(i)]
    a, b = ("".join(sorted(name+[j]*2)) for j in letters)
    return a, b

# populate the maps
for deriv, names in enumerate(derivative_names):
    for idx, name in enumerate(names):
        derivatives_map[name] = (deriv, idx)
        name_map[(deriv, idx)] = name
    idx = tuple(idx_to_nidx(name_to_idx(name)) for name in names)
    select_map[deriv] =  idx
    expand_map[deriv] = [None] * 3**deriv
    for idx in product(range(3), repeat=deriv):
        nidx = idx_to_nidx(idx)
        name = idx_to_name(idx)
        if name in names:
            expand_map[deriv][nidx] = (names.index(name),)
            laplace_map[name] = name,
        else:
            a, b = find_laplace(name)
            laplace_map[name] = a, b
            expand_map[deriv][nidx] = (names.index(a), names.index(b))

expand_map[0] = None

def name_to_deriv(name):
    return derivatives_map[name]

def deriv_to_name(deriv, idx):
    return name_map[(deriv, idx)]


def expand_tensor(c, order=None):
    """from the minimal linearly independent entries of a derivative of
    a harmonic field c build the complete tensor using its symmtry
    and laplace

    inverse of select_tensor()"""
    c = np.atleast_2d(c)
    if order is None:
        order = (c.shape[-1]-1)/2
    if order == 0:
        c = c[..., 0]
    elif order == 1:
        c = c
    else:
        d = np.array([
            c[..., i[0]] if len(i) == 1 else -c[..., i[0]]-c[..., i[1]]
            for i in expand_map[order]
            ])
        c = d.swapaxes(0, -1).reshape(c.shape[:-1] + (3,)*order)
    return c


def select_tensor(c, order=None):
    """select only a linealy idependent subset from a derivative of a
    harmonic field

    inverse of expand_tensor()"""
    c = np.atleast_2d(c)
    if order is None:
        order = len(c.shape) - 1 # nx, 3, ..., 3
    c = c.reshape(c.shape[:len(c.shape)-order]+(-1,))
    if order < 2:
        return c # fastpath
    else:
        return c[:, select_map[order]]


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
    #n = 1/(factorial(l)*2**l*np.sqrt(np.pi/(2*l+1)))
    #n *= 4*np.pi*2**(l+1)/(l+1)*factorial(l+1)**2/factorial(2*l+2)
    n = 8*np.sqrt(np.pi*(2*l+1))*factorial(l+1)/factorial(2*l+2)
    if l == 0:
        c = c/2
    elif l == 1:
        x, y, z = c
        c = np.array([
            -y,
            z,
            -x
        ])
    elif l == 2:
        xx, xy, xz, yy, yz = c
        zz = -xx-yy
        c = np.array([
            2*np.sqrt(3)*xy,
            -2*np.sqrt(3)*yz,
            -xx-yy+2*zz,
            -2*np.sqrt(3)*xz,
            np.sqrt(3)*(xx-yy),
        ])
    elif l == 3:
        xxy, xxz, yyz, xyy, xzz, yzz, xyz = c
        yyy = -xxy-yzz
        xxx = -xyy-xzz
        zzz = -xxz-yyz
        c = np.array([
            np.sqrt(10)*(-3*xxy+yyy),
            4*np.sqrt(15)*xyz,
            np.sqrt(6)*(xxy+yyy-4*yzz),
            2*(-3*xxz-3*yyz+2*zzz),
            np.sqrt(6)*(xxx+xyy-4*xzz),
            2*np.sqrt(15)*(xxz-yyz),
            np.sqrt(10)*(-xxx+3*xyy),
        ])
    elif l == 4:
        xxxy, xxxz, xxyy, xxzz, xyyy, xzzz, yyyz, yyzz, yzzz = c
        xxyz = -yyyz-yzzz
        xyzz = -xyyy-xxxy
        xxxx = -xxyy-xxzz
        yyyy = -xxyy-yyzz
        zzzz = -yyzz-xxzz
        xyyz = -xxxz-xzzz
        c = np.array([
            4*np.sqrt(35)*(xxxy-xyyy),
            2*np.sqrt(70)*(-3*xxyz+yyyz),
            4*np.sqrt(5)*(-xxxy-xyyy+6*xyzz),
            2*np.sqrt(10)*(3*xxyz+3*yyyz-4*yzzz),
            3*xxxx+6*xxyy-24*xxzz+3*yyyy-24*yyzz+8*zzzz,
            2*np.sqrt(10)*(3*xxxz+3*xyyz-4*xzzz),
            2*np.sqrt(5)*(-xxxx+6*xxzz+yyyy-6*yyzz),
            2*np.sqrt(70)*(-xxxz+3*xyyz),
            np.sqrt(35)*(xxxx-6*xxyy+yyyy),
        ])
    elif l == 5:
        xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz, xyyyz, xyyzz, \
            yyyzz, yyzzz = c
        xxxxy = -xxyyy-xxyzz
        yyyyy = -xxyyy-yyyzz
        xyzzz = -xxxyz-xyyyz
        yzzzz = -xxyzz-yyyzz
        xxxxz = -xxyyz-xxzzz
        yyyyz = -xxyyz-yyzzz
        zzzzz = -yyzzz-xxzzz
        xxxxx = -xxxyy-xxxzz
        xyyyy = -xxxyy-xyyzz
        xzzzz = -xyyzz-xxxzz
        c = np.array([
            3*np.sqrt(14)*(-5*xxxxy+10*xxyyy-yyyyy),
            24*np.sqrt(35)*(xxxyz-xyyyz),
            np.sqrt(70)*(3*xxxxy+2*xxyyy-24*xxyzz-yyyyy+8*yyyzz),
            8*np.sqrt(105)*(-xxxyz-xyyyz+2*xyzzz),
            2*np.sqrt(15)*(-xxxxy-2*xxyyy+12*xxyzz-yyyyy+12*yyyzz-8*yzzzz),
            2*(15*xxxxz+30*xxyyz-40*xxzzz+15*yyyyz-40*yyzzz+8*zzzzz),
            2*np.sqrt(15)*(-xxxxx-2*xxxyy+12*xxxzz-xyyyy+12*xyyzz-8*xzzzz),
            4*np.sqrt(105)*(-xxxxz+2*xxzzz+yyyyz-2*yyzzz),
            np.sqrt(70)*(xxxxx-2*xxxyy-8*xxxzz-3*xyyyy+24*xyyzz),
            6*np.sqrt(35)*(xxxxz-6*xxyyz+yyyyz),
            3*np.sqrt(14)*(-xxxxx+10*xxxyy-5*xyyyy),
        ])
    return n*c


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
