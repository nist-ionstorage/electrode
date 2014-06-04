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

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

from math import factorial
from itertools import product

import numpy as np


def shaped(func):
    """Wraps the given function so that it can be given a (n, m, ..., k)
    array. The function to be wrapped is is called with a (n, l) array. 
    The wrapper then returns a (m, ..., k, l) array.

    .. note:: deprecated, unused
    """
    def shape_wrapper(xyz, *args, **kwargs):
        xyz = np.atleast_2d(xyz)
        s = xyz.shape
        xyz = xyz.reshape(s[0], -1).T
        v = func(xyz, *args, **kwargs)
        v = v.reshape(s[1:]+v.shape[1:])
        return v
    return shape_wrapper


def shaper(func, xyz, *args, **kwargs):
    """Builds a `shaped()` wrapper for `func` on the fly and calls it.

    .. note:: deprecated, unused
    """
    return shaped(func)(xyz, *args, **kwargs)


def apply_method(s, name, *args, **kwargs):
    """Small helper to work around non-picklable
    instance methods and allow them to be called by multiprocessing
    tools.
    
    """
    return getattr(s, name)(*args, **kwargs)


def norm(a, axis=-1):
    """Special version of np.linalg.norm() that only covers the
    specified axis.
    
    .. note:: unused

    Parameters
    ----------
    a : array_like
    axis : int
        Axis to calculate the norm over.
    """
    # apparently faster
    return np.sqrt(np.einsum("...j,...j->...", a, a))
    # return np.sqrt(np.square(a).sum(axis=axis))


def rotate_tensor(c, r, order=None):
    """Rotate a tensor into another coordinate system.
    
    The first dimension(s) are used for parallelizing (arrays of
    tensors). The last dimensions are the indices of the tensor.

    Parameters
    ----------
    c : array_like, shape (3, 3)
        Coordinate system
    r : array_like
        Tensor to be rotated.
    order : int or None
        Tensor degree. If given, the last `order` dimenstions are rotated,
        else the degree degree is `r.ndim - 1`.
    """
    #c = np.atleast_1d(c)
    #r = np.atleast_2d(r)
    n = c.ndim - 1
    if order is None:
        order = n
    #slower: O(n**order)
    #ops = [c, range(order) + [Ellipsis]]
    #for i in range(order):
    #    ops.extend([r, [i, i+order]])
    #ops.append(range(order, 2*order) + [Ellipsis])
    #c = np.einsum(*ops)
    for i in range(order):
        #O(n*order):
        c = np.dot(c.swapaxes(n-i, -1), r).swapaxes(n-i, -1)
        #incorrect and probably not faster:
        #c = np.tensordot(c, r, axes=[i, 0])
    return c


_derivative_names = [[""]] + [s.split() for s in [
    "x y z",
    "xx xy xz yy yz",
    "xxy xxz yyz xyy xzz yzz xyz",
    "xxxy xxxz xxyy xxzz xyyy xzzz yyyz yyzz yzzz",
    "xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyz xyyzz yyyzz yyzzz",
    ]]

_derivatives_map = {} # sorted name: (derivative order, derivative index)
_name_map = {} # reverse derivatives_map
_expand_map = [] # derivative order: 3**order list of selected index
# or laplace pair
_select_map = [] # derivative order: 2*order+1 list of indices into
# 3**order expanded
_derive_map = {} # (derivative order, derivative index): ((lower
# derivative order, lower derivative index), axis to derive)

def name_to_idx(name):
    """Return a tuple of axis indices for given derivative
    
    Parameters
    ----------
    name : str
        A derivative name, e.g. `"xxz."`

    Returns
    -------
    idx : tuple of int
        Axis tuple, e.g. `(0, 0, 2)`.

    See also
    --------
    idx_to_name : Inverse
    """
    return tuple("xyz".index(n) for n in name)

def idx_to_name(idx):
    """Return sorted derivative name for axis tuple
    
    Parameters
    ----------
    idx : tuple of int
        An axis tuple.
    
    Returns
    -------
    name : str
        Derivative name.

    See also
    --------
    name_to_idx : Inverse
    """
    return "".join("xyz"[i] for i in sorted(idx))

def idx_to_nidx(idx):
    """Return index into flattened 3**order array for given order-tuple.
    
    Parameters
    ----------
    idx : tuple of int
        Axis tuple.
        
    Returns
    -------
    i : int
        Derivative order.
    j : int
        Index into flattened derivative tensor.
    """
    return sum(j*3**(len(idx)-i-1) for i, j in enumerate(idx))

def find_laplace(c):
    """Finds the two partial derivatives `a` and `b` such that the
    triple `a, b, c` is traceless, `a + b + c == 0`.

    Parameters
    ----------
        c : axis tuple

    Returns
    -------
    generator
        Generator of tuples `(a, b)` such that `a + b + c == 0` for any
        harmonic tensor of any order.
    """
    name = sorted(c)
    letters = list(range(3))
    found = None
    for i in letters:
        if name.count(i) >= 2:
            keep = name[:]
            keep.remove(i)
            keep.remove(i)
            take = letters[:]
            take.remove(i)
            a, b = (tuple(sorted(keep+[j,j])) for j in take)
            yield a, b

def _populate_maps():
    for deriv, names in enumerate(_derivative_names):
        #assert len(names) == 2*deriv+1, names
        for idx, name in enumerate(names):
            assert len(name) == deriv, name
            _derivatives_map[name] = (deriv, idx)
            _name_map[(deriv, idx)] = name
            if deriv > 0:
                for i, n in enumerate(_derivative_names[deriv-1]):
                    for j, m in enumerate("xyz"):
                        if name == "".join(sorted(n+m)):
                            _derive_map[(deriv, idx)] = (deriv-1, i), j
                            break
                assert (deriv, idx) in _derive_map, name
            for lap in find_laplace(name_to_idx(name)):
                a, b = map(idx_to_name, lap)
                assert (a not in names) or (b not in names), (name, a, b)
        idx = tuple(idx_to_nidx(name_to_idx(name)) for name in names)
        _select_map.append(idx)
        _expand_map.append([])
        for idx in product(range(3), repeat=deriv):
            name = idx_to_name(idx)
            if name in names:
                _expand_map[deriv].append(names.index(name))
            else:
                for a, b in find_laplace(idx):
                    a, b = map(idx_to_name, (a, b))
                    if a in names and b in names:
                        ia, ib = (names.index(i) for i in (a, b))
                        _expand_map[deriv].append((ia, ib))
        assert len(_expand_map[deriv]) == 3**deriv

_populate_maps()


def name_to_deriv(name):
    """Return (derivtive order, derivative index) for a given derivative
    name.

    Parameters
    ----------
    name : str
        Derivative name

    Returns
    -------
    order : int
    index : int

    See also
    --------
    deriv_to_name : inverse
    """
    return _derivatives_map[name]

def deriv_to_name(deriv, idx):
    """Return name for given (derivative order, index).

    Parameters
    ----------
    deriv : int
    idx : int

    Returns
    -------
    name : str

    See also
    --------
    name_to_deriv ; inverse
    """
    return _name_map[(deriv, idx)]

def construct_derivative(deriv, idx):
    """Return lower deriv and axis to derive.
    When constructing a higher order derivative, take the value of the
    lower order derivative and evaluate its derivative along the axis
    returned by this function.
    
    Parameters
    ----------
    deriv : int
    idx : int
    
    Returns
    -------
    i : tuple (int, int)
        Lower derivative (derivative order, derivative index)
    j : int
        Axis to derive along
    """
    return _derive_map[(deriv, idx)]


def expand_tensor(c, order=None):
    """From the minimal linearly independent entries of a derivative of
    a harmonic field build the complete tensor using its symmtry
    and Laplace.

    See also
    --------
    select_tensor : The inverse to this function.
    
    Parameters
    ----------
    c : array_like, shape (n, m)
    order : int or None
    
    Returns
    -------
    d : array_like, shape (n, 3, ..., 3)
    """
    #c = np.atleast_2d(c)
    if order is None:
        order = (c.shape[-1]-1)//2
    if order == 0:
        return c[..., 0]
    elif order == 1:
        return c
    else:
        shape = c.shape[:-1]
        d = np.empty(shape + (3**order,), c.dtype)
        for i, j in enumerate(_expand_map[order]):
            if type(j) is int:
                d[..., i] = c[..., j]
            else:
                d[..., i] = -c[..., j[0]]-c[..., j[1]] # laplace
        return d.reshape(shape + (3,)*order)


def deriv_to_reduced_idx(d):
    """Return the index or indices into a reduced tensor for a given
    derivative name

    Parameters
    ----------
    d : str
        Derivative name

    Returns
    r : int or tuple
        Either the index into the reduced tensor or a pair of indices
        `a, b` such that `-c[a]-c[b]` yields the desired derivative.
    """
    order = len(d)
    idx = name_to_idx(d)
    nidx = idx_to_nidx(idx)
    r = _expand_map[order][nidx]
    return r


def select_tensor(c, order=None):
    """Select only a linealy idependent subset from a derivative of a
    harmonic field.

    See also
    --------
    expand_tensor : The inverse to this function.
 
    Parameters
    ----------
    c : array_like, shape (n, 3, ..., 3)
        Input array to select from. `n` is the point index to select
        multiple values in parallel. The remaining axis are the
        respective derivatives. The order of the tensor is `l = (c.ndim
        - 1)/2`
    order : int
        Overrides the value inferred from c.ndim. The length of the
        first axis of the output array.

    Returns
    -------
    d : array_like, shape (n, m)
        Selected lineraly independent tensor. `l = (m - 1)/2` is the
        order of the tensor.
    """
    #c = np.atleast_1d(c)
    n = c.ndim
    if order is None:
        order = n - 1 # nx, 3, ..., 3
    c = c.reshape(c.shape[:n-order]+(-1,))
    if order < 2:
        return c # fastpath
    else:
        return c[..., _select_map[order]]


def cartesian_to_spherical_harmonics(c):
    """Converts basis cartesian derivative set to spherical harmonics.
    
    Given a cartesian derivative of a harmonic potential
    rewrite it in terms of real spherical harmonics.

    Convention and conversion to complex spherical harmonics as per
    http://theoretical-physics.net/dev/src/math/operators.html#real-spherical-harmonics

    Parameters
    ----------
    c : array_like, shape (m, n)
        Cartesian derivative values. `m` determines the derivative order
        `l = (m - 1)/2`. `n` is the point index (to evaluate mutliple
        values in parallel). Reshape to (n, -1) where necessary.
        This tensor should have either been created in reduced form or 
        reduced using to `select_tensor()`.

    Returns
    -------
    d : array_like, shape (m, n)
        Spherical harmonics values. The first axis is the azimuthal part
        and ranges from `(-l...l)`.
    """
    #c = np.atleast_1d(c)
    l = (c.shape[0] - 1)//2
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
    """Area and centroid of 2d polygon.
    
    Parameters
    ----------
    p1 : array_like, shape (n, 2)
        polygon boundary
    
    Returns
    -------
    a : float
        Polygon area, positive if polygon boundary is CCW if viewed from
        above.
    c : array, shape (2,)
        Centroid of the polygon area. Not necessarily inside the
        polygon.
    """
    p2 = np.roll(p1, -1, axis=0)
    r = p1[:, 0]*p2[:, 1] - p2[:, 0]*p1[:, 1]
    a = r.sum(0)/2.
    c = ((p1 + p2)*r[:, None]).sum(0)/(6*a)
    return a, c


def mathieu(r, *a):
    """Solve the exteded Mathieu/Floquet equation::

        x'' + (a_0 + 2 a_1 cos(2 t) + 2 a_2 cos(4 t) ... ) x = 0

    .. math:: \\frac{\\partial^2x}{\\partial t^2} + \\left(a_0 + \\sum_{i=1}^k
        2 a_i \\cos(2 i t)\\right)x = 0

    in n dimensions.

    Parameters
    ----------
    r : int
        frequency cutoff at `+- r`
    *a : tuple of array_like, all shape (n, n)
        `a[0]` is usually called `q`.
        `a[1]` is often called `-a`.
        `a[i]` is the prefactor of the `2 cos(2 i t)` term.
        Each `a[i]` can be an (n, n) matrix. In this case the `x` is an
        (n,) vector.

    Returns
    -------
    mu : array, shape (2*n*(2*r + 1),)
        eigenvalues
    b : array, shape (2*r + 1, 2, n, 2*n*(2*r + 1))
        eigenvectors with the following indices:
        (frequency component (-r...r), derivative, dimension,
        eigenvalue). b[..., i] the eigenvector to the eigenvalue mu[i].

    Notes
    -----
    * the eigenvalues and eigenvectors are not necessarily ordered
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
    """Trivial dummy class that offers the `multiprocessing.Pool`
    `apply_async()` method. But in a synchronous way.

    """
    def apply_async(self, func, args=(), kwargs={}):
        class _DummyRet(object):
            def get(self):
                return func(*args, **kwargs)
        return _DummyRet()
