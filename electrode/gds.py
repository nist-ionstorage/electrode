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
attr_base = sum(ord(i) for i in "electrode")
attr_name = attr_base + 10
attr_vdc = attr_base + 11
attr_vrf = attr_base + 12

def from_gds(fil, layer=None):
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
            ele.voltage_dc = float(props.get(attr_vdc, 0.))
            ele.voltage_rf = float(props.get(attr_vrf, 0.))
            path = np.array(e.xy)*lib.physical_unit
            path = np.c_[path, np.zeros((path.shape[0],))]
            ele.paths.append(path)
    return s

def to_gds(sys, layer=0, phys_unit=1e-9):
    lib = library.Library(version=5, name=b"trap", physical_unit=phys_unit,
            logical_unit=.001)
    stru = structure.Structure(name=b"electrodes")
    lib.append(stru)
    #stru.append(elements.Node(layer=layer, node_type=0, xy=[(0, 0)]))
    for e in sys.electrodes:
        if not type(e) is PolygonPixelElectrode:
            print "%s skipped" % e
            continue
        for p in e.paths:
            b = elements.Boundary(layer=layer, data_type=0,
                    xy=p[:, :2]/phys_unit)
            b.properties = []
            if e.name:
                b.properties.append((attr_name, e.name))
            b.properties.append((attr_vdc, str(e.voltage_dc)))
            b.properties.append((attr_vrf, str(e.voltage_rf)))
            stru.append(b)
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
