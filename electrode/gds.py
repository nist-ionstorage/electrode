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

from gdsii import library, structure, elements

from .system import System
from .electrode import PolygonPixelElectrode


# attribute namespaces anyone?
attr_base = sum(ord(i) for i in "electrode") # 951
attr_info = attr_base + 5
attr_name = attr_base + 10
attr_vdc = attr_base + 11
attr_vrf = attr_base + 12

def from_gds(fil, scale=1., layer=None):
    lib = library.Library.load(fil)
    s = System()
    for stru in lib:
        if not type(stru) is structure.Structure:
            print "%s skipped" % stru
            continue
        for e in stru:
            if not type(e) is elements.Boundary:
                print "%s skipped" % e
                continue
            if layer is not None and not e.layer == layer:
                print "%s skipped" % e
                continue
            props = dict(e.properties)
            name = props.get(attr_name, None)
            if name is None:
                ele = PolygonPixelElectrode()
                s.electrodes.append(ele)
            elif name in s.names:
                ele = s.electrode(name)
            else:
                ele = PolygonPixelElectrode(name=name)
                s.electrodes.append(ele)
            ele.dc = float(props.get(attr_vdc, 0.))
            ele.rf = float(props.get(attr_vrf, 0.))
            path = np.array(e.xy)*lib.physical_unit/scale
            path = path[:-1] # a gds boundary is a full loop
            ele.paths.append(path)
    # there are only outer loops
    for e in s.electrodes:
        e.paths = [p[::o] for p, o in zip(e.paths, e.orientations())]
    return s

def to_gds(sys, scale=1., layer=0, phys_unit=1e-9, gap_layer=1):
    lib = library.Library(version=5, name=b"trap", physical_unit=phys_unit,
            logical_unit=.001)
    eles = structure.Structure(name=b"electrodes")
    lib.append(eles)
    gaps = structure.Structure(name=b"gaps")
    lib.append(gaps)

    #stru.append(elements.Node(layer=layer, node_type=0, xy=[(0, 0)]))
    for e in sys.electrodes:
        if not type(e) is PolygonPixelElectrode:
            print "%s skipped" % e
            continue
        for p in e.paths:
            xy = p[:, :2]*scale/phys_unit
            xyb = np.r_[xy, xy[:1]]
            b = elements.Boundary(layer=layer, data_type=0, xy=xy)
            p = elements.Path(layer=gap_layer, data_type=0, xy=xyb)
            for i in p, b:
                i.properties = []
                if e.name:
                    i.properties.append((attr_name, e.name))
                i.properties.append((attr_vdc, str(e.dc)))
                i.properties.append((attr_vrf, str(e.rf)))
            eles.append(b)
            gaps.append(p)
            for l in eles, gaps:
                l.append(elements.Text(layer=layer, text_type=0,
                    xy=xy[:1], string=bytes(e.name)))
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
