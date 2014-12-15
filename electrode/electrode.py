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

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .utils import area_centroid, construct_derivative

try:
    if False: # test slow python only or fast numba expressions
        raise ImportError
    from .cexpressions import (point_potential, polygon_potential,
            mesh_potential)
except ImportError:
    from .expressions import (point_potential, polygon_potential,
            mesh_potential)



class Electrode(object):
    """An electrode of a Paul trap.

    Encapsulates the name, the dc and rf voltages and the electrical
    potential contribution of an electrode.

    Parameters
    ----------
    name : str
    dc : float
        DC potential associated with the constituents of this electrode.
        The electrodes's electrical potential is proportional to the DC
        potential. Does not influence the pseudopotential contribution.
    rf : float
        RF potential of this electrode. The pseudopotential controbution
        of this electrode is proportional to the square of its RF
        potential.
    """
    __slots__ = "name dc rf".split()

    def __init__(self, name="", dc=0., rf=0.):
        self.name = name
        self.dc = dc
        self.rf = rf

    def potential(self, x, derivative=0, potential=1., out=None):
        """Electrical potential contribution.

        Return the specified derivative of the eletrical potential
        contribution of this electrode assuming all other electrodes in
        the system are grounded.

        Parameters
        ----------
        x : array_like, shape (n, 3)
            Position to evaluate the electrical potential at. The first
            dimension is used to evaluate at several points in parallel.
        derivative : int
            Derivative order of the potential. `derivative=0` returns
            the potential, `derivative=1` the field/force,
            `derivative=2` the curvature/hessian.
        potential : float
            Scaling of the potential. Could be set to `self.rf` or
            `self.dc`. Since this method is used to determine both
            potentials (electrical and pseudo). Scaling with `self.dc`
            and `self.rf` is done in the respective methods of the `System`
            instance that contains this electrode.
        out : None or array_like, shape (n, 2*derivative + 1), double
            Array to add the potential contribution to. Needs to be
            zeroed before. If None, an array is created and returned.

        Returns
        -------
        potential : array, shape(n, 2*derivative + 1), double
            Output potential or `out` if given. The first dimension is
            the point index (same as `x`) the second is the derivative
            index. There are only `2*derivative + 1` values as the
            others are lineraly dependent. See `utils.expand_tensor` and
            `utils.select_tensor` for details and utility methods.

        See Also
        --------
        utils.expand_tensor
            Expand this reduced tensor to full form.
        utils.cartesian_to_spherical_harmonics
            Convert the reduced tensor to spherical harmonics.
        utils.find_laplace
            Find partial derivates that can be used to construct others
            using the vanishing trace of the Laplacian of.
        """
        raise NotImplementedError

    def orientations(self):
        """Return the orientation of the electrode surfaces with respect
        to the `z > 0` half space.

        Positive orientation yields positive potential for
        positive voltage and z>0.

        .. note:: Only fully implemented for `PolygonPixelElectrode`.
        """
        return np.array([])

    def plot(self, ax, label=None, color=None, **kw):
        """Plot this electrode in the supplied axes.

        Visualize a 2D projection of the electrode in the plot.

        .. note:: Only fully implemented in `PolyGonPixelElectrode` and
            `PointPixelElectrode`.
        """
        pass


class CoverElectrode(Electrode):
    """
    Continuous infinite conducting cover or mesh electrode.

    Parameters
    ----------
    height : float
        height above `z = 0`

    Notes
    -----
    * Only valid as part of a `System` consisting purely of
      `SurfaceElectrode`.
    * The other electrodes in the `System` all need to have their
      `cover_height` adjusted and set to the same value as `height`
      here. Otherwise their contributions are calculated wrong.
    """
    __slots__ = "height".split()

    def __init__(self, height=50., **kwargs):
        super(CoverElectrode, self).__init__(**kwargs)
        self.height = height

    def potential(self, x, derivative=0, potential=1., out=None):
        if out is None:
            out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        if derivative == 0:
            out[:, 0] += potential*x[:, 2]/self.height
        elif derivative == 1:
            out[:, 2] += potential/self.height
        else:
            pass
        return out


class SurfaceElectrode(Electrode):
    """A patch set embedded in a gapless infinite grounded conducting
    plane at `z = 0`.

    Subclasses of this class can make use of the cover electrode
    potential expansion [1]_ and play together with a `CoverElectrode`
    instance in a `System`.

    Parameters
    ----------
    cover_height : float
        The height of the CoverElectrode plane above the `z=0` plane.
    cover_nmax : int
        Expansion order of the effect of the cover plane onto this
        electrode's potential contribution.

    See Also
    --------
    Electrode
        `name`, `dc`, `rf` attributes/parameters
    CoverElectrode

    References
    ----------
    .. [1] Roman Schmied et al. 2011 New J. Phys. 13 115011
        http://dx.doi.org/10.1088/1367-2630/13/11/115011
    """
    __slots__ = "cover_height cover_nmax".split()

    def __init__(self, cover_height=50., cover_nmax=0, **kwargs):
        super(SurfaceElectrode, self).__init__(**kwargs)
        # cover plane height
        self.cover_height = cover_height
        # max components in cover plane potential expansion
        self.cover_nmax = cover_nmax


class PointPixelElectrode(SurfaceElectrode):
    """Surface electrode comprising several small pixels.

    The pixels are approximated as potential points. Their potential
    contribution is scaled by `areas`.

    Parameters
    ----------
    points : array_like, shape (n, s)
        Point pixel positions
    areas : array_like, shape (n,)
        Point pixel areas

    See Also
    --------
    Electrode
        `name`, `dc`, `rf` attributes/parameters
    SurfaceElectrode
        `cover_nmax` and `cover_height` attributes/constructor parameters
    """
    __slots__ = "points areas".split()

    def __init__(self, points=[], areas=[], **kwargs):
        super(PointPixelElectrode, self).__init__(**kwargs)
        self.points = np.asanyarray(points, np.double)
        self.areas = np.asanyarray(areas, np.double)

    def orientations(self):
        return np.ones_like(self.areas)

    def plot(self, ax, label=None, color=None, **kw):
        import matplotlib as mpl
        # color="red"?
        p = self.points
        a = (self.areas/np.pi)**.5*2
        col = mpl.collections.EllipseCollection(
                edgecolors="none",
                #cmap=plt.cm.binary, norm=plt.Normalize(0, 1.),
                facecolor=color,
                # FIXME/workaround: x in matplotlib<r8111
                widths=a, heights=a, units="xy",
                angles=np.zeros(a.shape),
                offsets=p[:, (0, 1)], transOffset=ax.transData)
        ax.add_collection(col)
        if label is None:
            label = self.name
        if label:
            ax.text(p[:,0].mean(), p[:,1].mean(), label,
                    horizontalalignment="center",
                    verticalalignment="center")

    def potential(self, x, derivative=0, potential=1., out=None):
        return point_potential(x, self.points, self.areas, potential,
                derivative, self.cover_nmax, self.cover_height, out)


class PolygonPixelElectrode(SurfaceElectrode):
    """Surface electrode comprising several polygonal patches.

    Parameters
    ----------
    paths : list of array_like, shape (n, 2)
        Polygon boundaries as lists of points. Polygons with positive
        orientation (counter clock wise) contribute with poisitive sign.
        THose with negative orientation contribute with negative sign.

    See Also
    --------
    Electrode
        `name`, `dc`, `rf` attributes/parameters
    SurfaceElectrode
        `cover_nmax` and `cover_height` attributes/constructor parameters
    """
    __slots__ = "paths".split()

    def __init__(self, paths=[], **kwargs):
        super(PolygonPixelElectrode, self).__init__(**kwargs)
        self.paths = [np.asanyarray(i, np.double) for i in paths]

    def orientations(self):
        return np.sign([area_centroid(pi)[0] for pi in self.paths])

    def plot(self, ax, label=None, color=None, **kw):
        import matplotlib as mpl
        # we already store the right order for interior/exterior
        vertices = np.concatenate([np.r_[p, [p[0]]]  for p in self.paths])
        codes = np.concatenate([np.r_[
            mpl.path.Path.MOVETO, np.ones(len(p))*mpl.path.Path.LINETO
            ].astype(mpl.path.Path.code_type)
            for p in self.paths])
        path = mpl.path.Path(vertices, codes)
        patch = mpl.patches.PathPatch(path, facecolor=color,
            edgecolor=kw.pop("edgecolor", "none"), **kw)
        ax.add_patch(patch)
        if label is None:
            label = self.name
        if label:
            for p in self.paths:
                ax.text(p[:,0].mean(), p[:,1].mean(), label,
                        horizontalalignment="center",
                        verticalalignment="center")

    def to_points(self):
        """Convert all polygons to points at their centroids with the
        appropriate area.

        Returns
        -------
        PointPixelElectrode
        """
        a, c = [], []
        for p in self.paths:
            ai, ci = area_centroid(p)
            a.append(ai)
            c.append(ci)
        e = PointPixelElectrode(name=self.name, dc=self.dc, rf=self.rf,
                cover_nmax=self.cover_nmax, cover_height=self.cover_height,
                areas=a, points=c)
        return e

    def potential(self, x, derivative=0, potential=1., out=None):
        return polygon_potential(x, self.paths, potential, derivative,
                self.cover_nmax, self.cover_height, out)


class MeshPixelElectrode(SurfaceElectrode):
    """A surface electrode consisting of a polygonal mesh with
    different potential for each polygon.

    .. note:: untested, unused

    Parameters
    ----------
    points : array_like, shape (n, 2)
        Vertex coordinates
    edges : array_like, shape (m, 2)
        The two vertex indices comprising each edge.
        Each value is an index into the first axis of `points`.
    polygons : array_like, shape (m,)
        Polygon associations of each edge. Each value is an index into
        `potentials`.
    potentials : array_like, shape (k,)
        Polygon potential prefactors.

    See Also
    --------
    Electrode
        `name`, `dc`, `rf` attributes/parameters
    SurfaceElectrode
        `cover_nmax` and `cover_height` attributes/constructor parameters
    """
    __slots__ = "points edges polygons potentials".split()

    def __init__(self, points=[], edges=[], polygons=[], potentials=[],
            **kwargs):
        super(MeshPixelElectrode, self).__init__(**kwargs)
        self.points = np.asanyarray(points, np.double)
        self.edges = np.asanyarray(edges, np.intc)
        self.polygons = np.asanyarray(polygons, np.intc)
        self.potentials = np.asanyarray(potentials, np.double)

    @classmethod
    def from_polygon_system(cls, s):
        points = []
        edges = []
        polygons = []
        potentials = []
        for p in s:
            assert isinstance(p, PolygonPixelElectrode), p
            for i in p.paths:
                ei = len(points)+np.arange(len(i))
                points.extend(i)
                edges.extend(np.c_[np.roll(ei, 1, 0), ei])
                polygons.extend(len(potentials)*np.ones(len(ei)))
            potentials.append(p.dc)
        return cls(dc=1, points=points, edges=edges, polygons=polygons,
                potentials=potentials)

    def potential(self, x, derivative=0, potential=1., out=None):
        return mesh_potential(x, self.points, self.edges, self.polygons,
                self.potentials*potential, derivative,
                self.cover_nmax, self.cover_height, out)


class GridElectrode(Electrode):
    """Electrode based on a precalculated grid of electrical potentials.

    Parameters
    ----------
    data : list of array_like, shape (n, m, k, l)
        List of potential derivatives. The ith data entry is of order
        (l - 1)/2. Each entry is shaped as a (n, m, k) grid.
    origin : array_like, shape (3,)
        Position of the (n, m, k) = (0, 0, 0) voxel.
    spacing : array_like, shape (3,)
        Voxel pitch.

    See Also
    --------
    Electrode
        `name`, `dc`, `rf` attributes/parameters
    """
    __slots__ = "data origin spacing".split()

    def __init__(self, data=[], origin=(0, 0, 0), spacing=(1, 1, 1),
            **kwargs):
        super(GridElectrode, self).__init__(**kwargs)
        self.data = [np.asanyarray(i, np.double) for i in data]
        self.origin = np.asanyarray(origin, np.double)
        self.spacing = np.asanyarray(spacing, np.double)

    @classmethod
    def from_result(cls, result, maxderiv=4):
        """Create a `GridElectrode` from a `bem.result.Result` instance.

        Parameters
        ----------
        result : bem.result.Result
        maxderiv : int
            Maximum derivative order to precompute based on the
            available data.

        Returns
        -------
        GridElectrode
        """
        origin = result.grid.get_origin()
        spacing = result.grid.step
        data = [result.potential[:, :, :, None]]
        if result.field is not None:
            data.append(result.field.transpose(1, 2, 3, 0))
        obj = cls(origin=origin, spacing=spacing, data=data)
        obj.generate(maxderiv)
        return obj

    @classmethod
    def from_vtk(cls, fil, maxderiv=4):
        """Load grid potential data from vtk StructuredPoints.

        .. note:: needs `tvtk`

        Parameters
        ----------
        fil : str
            File name of the VTK StructuredPoints file containing the
            gridded data.
        maxderiv : int
            Maximum derivative order to precompute.

        Returns
        -------
        GridElectrode
        """
        from tvtk.api import tvtk
        #sgr = tvtk.XMLImageDataReader(file_name=fil)
        sgr = tvtk.StructuredPointsReader(file_name=fil)
        sgr.update()
        sg = sgr.output
        pot = [None, None]
        for i in range(sg.point_data.number_of_arrays):
            name = sg.point_data.get_array_name(i)
            if "_pondpot" in name:
                continue # not harmonic, do not use it
            elif name not in ("potential", "field"):
                continue
            sp = sg.point_data.get_array(i)
            data = sp.to_array()
            spacing = sg.spacing
            origin = sg.origin
            dimensions = tuple(sg.dimensions)
            dim = sp.number_of_components
            data = data.reshape(dimensions[::-1]+(dim,)).transpose(2, 1, 0, 3)
            pot[int((dim-1)/2)] = data
        obj = cls(origin=origin, spacing=spacing, data=pot)
        obj.generate(maxderiv)
        return obj

    def generate(self, maxderiv=4):
        """Generate missing derivative orders by successive finite
        differences from the already present derivative orders.

        .. note:: Finite differences amplify noise and discontinuities
            in the original data.

        Parameters
        ----------
        maxderiv : int
            Maximum derivative order to precompute if not already
            present.
        """
        for deriv in range(maxderiv):
            if len(self.data) < deriv+1:
                self.data.append(self.derive(deriv))
            ddata = self.data[deriv]
            assert ddata.ndim == 4, ddata.ndim
            assert ddata.shape[-1] == 2*deriv+1, ddata.shape
            if deriv > 0:
                assert ddata.shape[:-1] == self.data[deriv-1].shape[:-1]

    def derive(self, deriv):
        """Take finite differences along each axis.

        Parameters
        ----------
        deriv : derivative order to generate

        Returns
        -------
        data : array, shape (n, m, k, l)
            New derivative data, l = 2*deriv + 1
        """
        odata = self.data[deriv-1]
        ddata = np.empty(odata.shape[:-1] + (2*deriv+1,), np.double)
        for i in range(2*deriv+1):
            (e, j), k = construct_derivative(deriv, i)
            # TODO triple work
            grad = np.gradient(odata[..., j], *self.spacing)[k]
            ddata[..., i] = grad
        return ddata

    def potential(self, x, derivative=0, potential=1., out=None):
        x = (x - self.origin[None, :])/self.spacing[None, :]
        if out is None:
            out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        dat = self.data[derivative]
        for i in range(2*derivative+1):
            out[:, i] += potential*map_coordinates(dat[..., i], x.T,
                    order=1, mode="nearest")
        return out
