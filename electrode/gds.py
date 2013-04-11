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

import logging

import numpy as np
from gdsii import library, structure, elements
from shapely import geometry, ops

from .system import System
from .electrode import PolygonPixelElectrode
from .polygons import Polygons

logger = logging.getLogger()


class GdsPolygons(Polygons):
    # attribute namespaces anyone?
    attr_base = sum(ord(i) for i in "electrode") # 951
    attr_name = attr_base + 0

    @classmethod
    def from_gds_simple(cls, fil, scale=1., layers=None, name=None):
        lib = library.Library.load(fil)
        polys = {}
        for stru in lib:
            assert type(stru) is structure.Structure
            if name is not None and name != stru.name:
                continue
            for e in stru:
                if not type(e) is elements.Boundary:
                    logger.debug("%s skipped", e)
                    continue
                if (layers is not None and
                        (e.layer, e.data_type) not in layers):
                    logger.debug("%s skipped", e)
                    continue
                props = dict(e.properties)
                name = props.get(cls.attr_name, "")
                path = np.array(e.xy)*lib.physical_unit/scale
                # a gds boundary is a full loop, shapely is ok with that
                #path = path[:-1]
                polys.setdefault(name, []).append(path)
        obj = cls()
        # there are only outer loops
        for name, poly in polys.items():
            poly = [geometry.polygon.orient(geometry.Polygon(i), 1) for i in poly]
            if name is None:
                obj.extend([(name, geometry.MultiPolygon(i)) for i in poly])
            else:
                obj.append((name, geometry.MultiPolygon(poly)))
        return obj

    @classmethod
    def from_gds(cls, fil, scale=1., name=None, poly_layers=None,
            gap_layers=None, route_layers=[], bridge_layers=[], **kwargs):
        lib = library.Library.load(fil)
        polys = []
        gaps = []
        routes = []
        bridges = []
        for stru in lib:
            assert type(stru) is structure.Structure
            for e in stru:
                ij = e.layer, e.data_type
                path = np.array(e.xy)*lib.physical_unit/scale
                if type(e) is elements.Boundary:
                    if poly_layers is None or ij in poly_layers:
                        polys.append(path)
                elif type(e) is elements.Path:
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
            bridges=[], edge=40., buffer=1e-12):
        """
        start with a edge by edge square,
        subtract boundaries and put them in polygon
        electrodes named by their layer/datatype then pattern
        the rest using the routes (taken to be near-zero-width) and
        add one polygon electrode for each resulting fragment
        """
        fragments = []
        field = geometry.Polygon([[edge/2, edge/2], [-edge/2, edge/2],
                                  [-edge/2, -edge/2], [edge/2, -edge/2]])
        gaps = geometry.MultiLineString(gaps)
        #gaps = reduce(lambda a, b: a.union(b), gaps)
        #gaps = ops.cascaded_union(gaps).intersection(field) # segfaults
        field = field.difference(gaps.buffer(buffer, 1))
        if routes and bridges:
            routes = geometry.MultiLineString(routes)
            bridges = geometry.MultiLineString(bridges)
            for poly in ops.polygonize(routes.union(bridges)):
                field = field.union(poly)
        field = field.buffer(0, 1)
        polys = map(geometry.Polygon, polys)
        for poly in polys:
            assert poly.is_valid, poly
            assert field.is_valid, field
            poly = poly.intersection(field)
            field = field.difference(poly)
            fragments.append(poly)
        if routes:
            routes = geometry.MultiLineString(routes)
            field = field.difference(routes.buffer(buffer, 1))
        fragments.append(field)

        p = cls()
        for fragment in fragments:
            if type(fragment) is geometry.Polygon:
                fragment = [fragment]
            for i in fragment:
                # assume that buffer is much smaller than any relevant
                # distance and round coordinates to 10*buffer, then simplify
                i = np.array(i.exterior.coords)
                i = np.around(i, int(-np.log10(buffer)-1))
                i = geometry.Polygon(i)
                p.append(("", geometry.MultiPolygon([i])))
        return p

    def to_gds(self, scale=1., poly_layer=(0, 0), gap_layer=None,
            text_layer=0, phys_unit=1e-9):
        lib = library.Library(version=5, name=b"trap_electrodes",
                physical_unit=phys_unit, logical_unit=1e-3)
        stru = structure.Structure(name=b"trap_electrodes")
        lib.append(stru)
        #stru.append(elements.Node(layer=layer, node_type=0, xy=[(0, 0)]))
        for name, polys in self:
            props = {cls.attr_name: bytes(name)}
            for p in e.paths:
                xy = p*scale/phys_unit
                xyb = np.r_[xy, xy[:1]]
                if poly_layer is not None:
                    p = elements.Boundary(layer=poly_layer[0],
                            data_type=poly_layer[1], xy=xy)
                    p.properties = props.items()
                    stru.append(p)
                if gap_layer is not None:
                    p = elements.Path(layer=gap_layer[0],
                            data_type=gap_layer[1], xy=xyb)
                    p.properties = props.items()
                    stru.append(p)
                if text_layer is not None:
                    p = elements.Text(layer=text_layer[0],
                            text_type=text_layer[1], xy=xy[:1],
                            string=bytes(name))
                    p.properties = props.items()
                    stru.append(p)
        return lib


if __name__ == "__main__":
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
