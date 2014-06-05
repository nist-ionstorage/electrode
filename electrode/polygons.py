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

import logging, operator

import numpy as np
from scipy.interpolate import splprep, splev
from shapely import geometry, ops
from gdsii import library, structure, elements

from .system import System
from .electrode import PolygonPixelElectrode
from .utils import area_centroid

logger = logging.getLogger("electrode")


"""Polygons class, methods for polygons manipulation, and tools for
loading, saving GDS files.

.. note::
    Needs GDSII and Shapely
"""


class Polygons(list):
    """A simple representations of polygonal electrode data (two
    dimensional).
    
    A named list of shapely `MultiPolygons`::

        [
            ("electrode_name", MultiPolygon(...)),
            ...
        ]
    """

    @classmethod
    def from_system(cls, system):
        """Convert a `System` instance to a `Polygons` instance.
        
        Parameters
        ----------
        system : System
            
        Returns
        -------
        Polygons
        """
        obj = cls()
        for e in system:
            if not hasattr(e, "paths"):
                continue
            # assert isinstance(e, PolygonPixelElectrode), (e, e.name)
            exts, ints = [], []
            for pi in e.paths:
                # shapely ignores f-contiguous arrays so copy
                # https://github.com/sgillies/shapely/issues/26
                ei = area_centroid(pi)[0]
                pi = geometry.LinearRing(pi.copy("C"))
                if ei < 0:
                    ints.append((abs(ei), pi))
                elif ei > 0:
                    exts.append((abs(ei), pi))
            if not exts:
                continue
            ints.sort(key=operator.itemgetter(0))
            exts.sort(key=operator.itemgetter(0))
            # the following needs to be complicated to cover
            # onion-like "ext in int in ext" cases.
            groups = []
            for exta, exterior in exts:
                ep = geometry.Polygon(exterior)
                gint = []
                done = set()
                for i, (inta, interior) in enumerate(ints):
                    if inta >= exta:
                        break
                    if ep.contains(interior):
                        gint.append(interior)
                        done.add(i)
                    ints = [i for j, i in enumerate(ints) if j not in done]
                pi = geometry.Polygon(exterior, gint)
                if pi.is_valid and pi.area > 0:
                    groups.append(pi)
            # remaining interiors must be top level or "free"
            #assert not ints, ints
            #mp = geometry.MultiPolygon(groups)
            #mp = mp.union(geometry.Point())
            mp = ops.unary_union(groups)
            obj.append((e.name, mp))
        return obj

    def to_system(self):
        """Convert self to a `System` instance.

        Returns
        -------
        System
        """
        s = System()
        for n, p in self:
            e = PolygonPixelElectrode(name=n, paths=[])
            s.append(e)
            if isinstance(p, geometry.Polygon):
                p = [p]
            for pi in p:
                if not pi.is_valid or not pi.area:
                    continue
                ext = np.array(pi.exterior.coords)
                if not pi.exterior.is_ccw:
                    ext = ext[::-1]
                e.paths.append(ext[:-1, :2])
                for ii in pi.interiors:
                    int = np.array(ii.coords)
                    if ii.is_ccw:
                        int = int[::-1]
                    e.paths.append(int[:-1, :2])
        return s

    # attribute namespaces anyone?
    _attr_base = sum(ord(i) for i in "electrode") # 951
    _attr_name = _attr_base + 0

    @classmethod
    def from_gds(cls, fil, scale=1., name=None, poly_layers=None,
            gap_layers=None, route_layers=[], bridge_layers=[], **kwargs):
        """Opens a GDS Library and converts a Structure to a `Polygons`
        instance.

        Parameters
        ----------
        fil : file or string
            GDS file to be opened
        scale : float
            Natural length scale to rescale the data to. Usually the ion
            height. Defaults to 1 (one meter).
        name : str
            Name of the Structure (aka Cell) in the GDS file to be used.
            The first if `name` is `None`.
        poly_layers : list of tuples (layer number, data type number)
            Layers and datatypes where polygons are to be extraced and
            used. E.g. [(13, 6), (14, 0)].
        gap_layers : list of tuples (layer number, data type number)
            Layers/datatypes where paths are to be used as gaps cutting
            into the polygons.
        route_layers : list of tuples (layer number, data type number)
            Layers/datatypes where paths are to be used to route through
            polygons. Polygons defined by the union of the paths in
            `route_layers` and `bridge_layers` together define new
            polygons. Paths in `route_layers` separate these from
            existing polygons. Paths in `bridge_layers` connect these
            new polygons to existing ones.
        **kwargs : passed to `cls.from_data()`

        Returns
        -------
        System

        See also
        --------
        from_data()
        """
        lib = library.Library.load(fil)
        polys = []
        gaps = []
        routes = []
        bridges = []
        for stru in lib:
            assert isinstance(stru, structure.Structure)
            if name is None or name == stru.name:
                break
        for e in stru:
            path = np.array(e.xy)*lib.physical_unit/scale
            props = dict(e.properties)
            name = props.get(cls._attr_name, "")
            if isinstance(e, elements.Boundary):
                ij = e.layer, e.data_type
                if poly_layers is None or ij in poly_layers:
                    polys.append((name, path))
            elif isinstance(e, elements.Path):
                if path.shape[0] < 2:
                    continue
                ij = e.layer, e.data_type
                if gap_layers is None or ij in gap_layers:
                    gaps.append(path)
                elif ij in route_layers:
                    routes.append(path)
                elif ij in bridge_layers:
                    bridges.append(path)
                else:
                    logger.debug("%s skipped", e)
            else:
                logger.debug("%s skipped", e)
        return cls.from_data(polys, gaps, routes, bridges, **kwargs)

    @classmethod
    def from_data(cls, polys=[], gaps=[], routes=[],
            bridges=[], edge=40., buffer=1e-10):
        """Process polygonal, gaps, route and bridge data into a
        `Polygons` instance.

        Start with a `edge` by `edge` square centered at (0, 0), and cut
        it according to gaps. Then undo the fragmentation that is fully
        encircled by routes and bridges. Then undo fragmentation within
        polys and then fragment along the poly boundaries. Tinally
        fragment along routes (again).

        The result is a `Polygons` that contains the fragmented area of
        the original square, with polys as some of the fragments and the
        rest fragmented along routes and those gaps that are
        not encircled in routes and bridges.

        Parameters
        ----------
        polys : list of array_like
            Polygons (arrays of points with (n, 2) shape).
        gaps : list of array_like
            Gap paths (arrays of points with (n, 2) shape).
        routes : list of array_like
        bridges : list of array_like
        edge : float
            Edge length of the bounding square. All Polygons and gaps
            are intersected with this. There will be nothing outside
            this square and all area inside the square will be segmented
            and output in one of the resulting polygons.
        buffer : float
            A small size that is used to associate a finite width
            to gaps and routes. there should be no features smaller than
            10*buffer.

        Returns
        -------
        Polygons
        """
        fragments = cls()
        field = geometry.Polygon([[edge/2, edge/2], [-edge/2, edge/2],
                                  [-edge/2, -edge/2], [edge/2, -edge/2]])
        field0 = field
        if gaps:
            gaps = geometry.MultiLineString(gaps)
            field = field.difference(gaps.buffer(buffer, 1))
        if routes:
            routes = geometry.MultiLineString(routes)
        if routes and bridges:
            bridges = geometry.MultiLineString(bridges)
            for poly in ops.polygonize(routes.union(bridges)):
                field = field.union(poly.intersection(field0))
        # field = field.buffer(0, 1)
        for name, poly in polys:
            poly = geometry.Polygon(poly)
            assert poly.is_valid, poly
            poly = geometry.polygon.orient(poly, 1)
            poly = poly.intersection(field)
            fragments.append((name, poly))
            field = field.difference(poly)
        if routes:
            field = field.difference(routes.buffer(buffer, 1))
        assert field.is_valid, field
        if isinstance(field, geometry.Polygon):
            field = [field]
        for i in field:
            # assume that buffer is much smaller than any relevant
            # distance and round coordinates to 10*buffer, then simplify
            i = [np.around(j.coords, int(-np.log10(buffer)-1)) for j in
                    [i.exterior] + list(i.interiors)]
            i = geometry.Polygon(i[0], i[1:])
            fragments.append(("", i))
        return fragments

    def to_gds(self, scale=1., poly_layer=(0, 0), gap_layer=(1, 0),
            text_layer=(0, 0), phys_unit=1e-9, name="trap_electrodes",
            edge=None, gap_width=0.):
        """Convert this `Polygons` instance to a GDS library with one
        structure.

        Parameters
        ----------
        scale : float
            Length scale to convert to. One Polygons length unit will be
            one physical scale unit in the GDS file (irrespective of
            `phys_unit`).
        poly_layer : tuple (int, int)
        gap_layer : tuple (int, int)
        text_layer : tuple (int, int)
            Layers to put the various pieces of data into.
        phys_unit : float
            Physical length scale in the GDS (see the GDS file format
            specifications).
        name : str
            Name of the Structure (aka Cell) to write the data to.
        edge : float
            Edge length of the clipping square to clip all `Polygons`
            data to.
        gap_width : float
            Gap paths are to be exported with a width of `gap_width`

        Returns
        -------
        Library
            `gdsii.library.Library` with one Structure/Cell named `name`
            containing the given data.
        """
        lib = library.Library(version=5, name=bytes(name),
                physical_unit=phys_unit, logical_unit=1e-3)
        stru = structure.Structure(name=bytes(name))
        lib.append(stru)
        if edge:
            field = geometry.Polygon([[edge/2, edge/2], [-edge/2, edge/2],
                                      [-edge/2, -edge/2], [edge/2, -edge/2]])
        #stru.append(elements.Node(layer=layer, node_type=0, xy=[(0, 0)]))
        gaps = []
        for name, polys in self:
            props = {}
            if name:
                props[self._attr_name] = bytes(name)
            if not hasattr(polys, "geoms"):
                polys = [polys]
            for poly in polys:
                assert poly.is_valid, poly
                if poly.is_empty:
                    continue
                if edge is not None:
                    poly = poly.intersection(field)
                xy = np.array(poly.exterior.coords.xy).copy()
                xy = xy.T[:, :2]*scale/phys_unit
                if text_layer is not None and name:
                    p = elements.Text(layer=text_layer[0],
                            text_type=text_layer[1], xy=xy[:1],
                            string=bytes(name))
                    p.properties = props.items()
                    stru.append(p)
                if poly_layer is not None:
                    p = elements.Boundary(layer=poly_layer[0],
                            data_type=poly_layer[1], xy=xy)
                    p.properties = props.items()
                    stru.append(p)
                gaps.append(poly.exterior)
                gaps.extend(poly.interiors)
        if gap_layer is not None:
            ##g = ops.cascaded_union(gaps) # segfaults
            #g = geometry.MultiLineString(gaps)
            # this breaks it up badly
            g = geometry.LineString()
            for i in gaps:
                g = g.union(i)
            if edge is not None:
                g = g.intersection(field)
                g = g.difference(field.boundary)
            if not hasattr(g, "geoms"):
                g = [g]
            for loop in g:
                xy = np.array(loop.coords.xy).copy()
                xy = xy.T[:, :2]*scale/phys_unit
                xyb = np.r_[xy, xy[:1]]
                p = elements.Path(layer=gap_layer[0],
                        data_type=gap_layer[1], xy=xyb)
                p.width = int(gap_width*scale/phys_unit)
                stru.append(p)
        return lib

    def validate(self):
        """Asserts geometric validity of all electrodes.

        Raises
        ------
        ValueError
           If the polygon is geometrically invalid (see shapely
           documentation).
        """
        for ni, pi in self:
            if not pi.is_valid:
                raise ValueError("%s %s" % (ni, pi))

    def remove_overlaps(self):
        """Successively removes overlaps with preceeding electrodes.

        Returns
        -------
        Polygons
            Output with overlaps removed
        """
        p = Polygons()
        acc = geometry.Point()
        for ni, pi in self:
            pa = acc.intersection(pi)
            if pa.is_valid and pa.area > np.finfo(np.float32).eps:
                pc = pi.difference(pa)
                if pc.is_valid:
                    pi = pc
            acca = acc.union(pi)
            if acca.is_valid:
                acc = acca
            p.append((ni, pi))
        return p

    def add_gaps(self, gapsize=0):
        """Shrinks each electrode by adding a buffer around it.

        Gaps between previously touching electrodes will be gapsize wide
        electrodes must not be overlapping.

        Parameters
        ----------
        gapsize : float
           Size of the gap. Each polygon will be buffered by
           `-gapsize/2`.

        Returns
        -------
        Polygons
            Output with gaps added
        """
        p = Polygons()
        for ni, pi in self:
            pb = pi.buffer(-gapsize/2., 1)
            if pb.is_valid:
                pi = pb
            p.append((ni, pi))
        return p

    def unify(self):
        """Form unary unions of polygons in each electrode

        Returns
        -------
        Polygons
            Output containing unions of electrode polygons.
        """
        p = Polygons()
        for ni, pi in self:
            if not hasattr(pi, "geoms"):
                pi = [pi]
            p.append((ni, ops.unary_union(list(pi))))
        return p

    def simplify(self, buffer=0, preserve_topology=False):
        """Simplify the polygons.

        See the shapely method for details

        Parameters
        ----------
        buffer : float
            Determines the simplification tolerance (see shapely). 
            In the output polygons, no point will be further than
            `buffer` away from the original geometry.
        preserve_topology : bool
            See shapely documentation.

        Returns
        -------
        Polygons
            Simplified output
        """
        if buffer == 0:
            return self.add_gaps(buffer)
        p = Polygons()
        for ni, pi in self:
            p.append((ni, pi.simplify(buffer,
                preserve_topology=preserve_topology)))
        return p

    def filter(self, test=None, deep=True):
        """Drops all patches that fail the test function.
       
        Parameters
        ----------
        test : callable
            Function to be called as `test(name, poly_or_boundary)` that
            returns `True` if the Polygon or boundary (LinearRing) is to
            be kept. If `None`, defaults to dropping polygons smaller than
            1e-2 in area.
        deep : bool
            If `True`, apply test to every boundary (exterior and
            interiors), else every polygon.

        Returns
        -------
        Polygons
            Filtered output
        """
        if test is None and deep:
            def test(name, boundary):
                return abs(geometry.Polygon(boundary).area) > 1e-2

        p = Polygons()
        for ni, pi in self:
            if not hasattr(pi, "geoms"):
                pi = [pi]
            if deep:
                pe = []
                for ppi in pi:
                    if test(ni, ppi.exterior):
                        pe.append(geometry.Polygon(ppi.exterior,
                            [_ for _ in ppi.interiors if test(ni, _)]))
            else:
                pe = [_ for _ in pi if test(ni, _)]
            if pe:
                p.append((ni, geometry.MultiPolygon(pe)))
        return p

    def smooth(self, smoothing=1, straight=1e-9, clip_len=(1e-2, 1e2)):
        """Smoothes the polygons.

        Parameters
        ----------
        smoothing : float
            Gets passed down to splprep() and is the average
            deviation in units of the local segment length.
        straight : float
            Enables straight path detection and forces the spline
            to pass through straight points. Also the straightness
            threshold in dx and dy.
        clip_len : tuple (float, float)
            The local segment length to within the given interval. Only
            used for weighting the deviation of the path relative to the
            local edge length.

        Returns
        -------
        Polygons
            Smoothed output
        """
        p = Polygons()
        for name, mpoly in self:
            smoothed = []
            if not hasattr(mpoly, "geoms"):
                mpoly = [mpoly]
            for poly in mpoly:
                loops = []
                b = poly.boundary
                try:
                    len(b)
                except TypeError:
                    b = [b]
                for line in b:
                    x, y = np.array(line.coords.xy)
                    assert x[0] == x[-1] and y[0] == y[-1], "not periodic"
                    # periodic central differences
                    dx = np.roll(x, 1) - np.roll(x, -1)
                    dy = np.roll(y, 1) - np.roll(y, -1)
                    du = np.sqrt(dx**2 + dy**2)/2
                    if clip_len is not None:
                        np.clip(du, clip_len[0], clip_len[1], du)
                    if straight is not None:
                        ver = np.fabs(dx) < straight
                        hor = np.fabs(dy) < straight
                        ta = np.roll(ver, -1) | np.roll(ver, 1)
                        tb = np.roll(hor, -1) | np.roll(hor, 1)
                        du = np.where(ta | tb, straight, du)
                    u = np.cumsum(du)
                    s = len(x)*smoothing
                    tckp, u = splprep([x, y], w=1/du, u=u/u[-1], s=s,
                                      k=1, per=1, nest=len(x) + 2)
                    xn, yn = splev(np.r_[tckp[0][1+1:-1-1], 1], tckp)
                    loops.append(np.c_[xn, yn])
                # TODO: does not ensure that all interiors are within
                # exterior and that the interiors do not overlap
                exterior, interior = loops[0], loops[1:]
                if len(exterior) >= 3:
                    interior = [_ for _ in interior if len(_) >= 3]
                    smoothed.append(geometry.Polygon(exterior, interior))
            p.append((name, geometry.MultiPolygon(smoothed)))
        return p

    def assign_to_pad(self, pads):
        """Finds polygons intersecting the points given.
        
        Parameters
        ----------
        pads : list of tuples (float, float)
            (x, y) tuples of the pad coordinates

        Returns
        -------
        iterator
            Iterator over matching tuples `(pad number, polygon index,
            polygon)`
        """
        polys = range(len(self))
        for pad, (x, y) in enumerate(pads):
            p = geometry.Point(x, y)
            for i in polys:
                name, poly = self[i]
                if p.intersects(poly):
                    yield pad, i
                    polys.remove(i)
                    break
            if not polys:
                break
        # assert not polys, polys

    def gaps_union(self):
        """Union of the boundaries of the polygons.

        If the boundaries of adjacent polygons coincide, this returns
        only the gap paths.

        Returns
        -------
        LineString
            Union of all gaps.
        """
        gaps = []
        for name, multipoly in self:
            if isinstance(multipoly, geometry.Polygon):
                multipoly = [multipoly]
            for poly in multipoly:
                gaps.append(poly.boundary)
        #return ops.cascaded_union(gaps) # segfaults
        g = geometry.LineString()
        for i in gaps:
            g = g.union(i)
        return g

    def restrict(self, geometry):
        """Intersect each constituent with the given geometry.

        Returns
        -------
        Polygons
        """
        p = Polygons()
        for name, multipoly in self:
            p.append((name, multipoly.intersection(geometry)))
        return p

    def within(self, edge=50):
        e = edge/2.
        g = geometry.Polygon([(e, e), (-e, e), (-e, -e), (e, -e)])
        return self.restrict(g)


def square_pads(step=10., edge=200., odd=False, start_corner=0):
    """Generatex XY coordinates for equally spaced pads around a square.
    
    Pads are along each edge of the square, placed inwards by `step/2`.

    Parameters
    ----------
    step : float
        Pad pitch
    edge : float
        Edge length of the square
    odd : bool
        If odd=True, there is a pad the center of an edge.
    start_corner : int
        Corner to start in. 0 is top left (-x, +y). Counter
        clockwise from there.

    Returns
    -------
    array_like
        (n, 2) array of xy coordinates of pad centers.
    """
    n = int(edge/step)
    if odd: n += (n % 2) + 1
    p = np.arange(-n/2.+.5, n/2.+.5)*step
    assert len(p) == n, (p, n)
    # top left as origin is common for packages
    q = (edge/2.-step/2.)*np.ones_like(p)
    edges = [(-q, -p), (p, -q), (q, p), (-p, q)]
    xy = np.concatenate(edges[start_corner:] + edges[:start_corner], axis=1)
    assert xy.shape == (2, 4*n), xy.shape
    return xy.T


if __name__ == "__main__":
    import cPickle as pickle
    s = pickle.load(open("rfjunction.pickle", "rb"))
    p = Polygons.from_system(s)
    p = p.remove_overlaps()
    p = p.add_gaps(.05)
    s1 = p.to_system()
    for si in s, s1:
        for ei in si:
            if not hasattr(ei, "paths"):
                continue
            for pi, oi in zip(ei.paths, ei.orientations()):
                print(ei.name, pi, oi)
        print()
    pickle.dump(s1, open("rfjunction1.pickle", "wb"))

    import sys
    from matplotlib import pyplot as plt
    with open(sys.argv[1], "rb") as fil:
        s = from_gds(fil)
    fig, ax = plt.subplots()
    s.plot(ax)
    fig.savefig("gds_to_system.pdf")
    l = to_gds(s)
    with open("system_to_gds.gds", "wb") as fil:
        l.save(fil)
