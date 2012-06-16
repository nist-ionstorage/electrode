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

from .system import System
from .electrode import PolygonPixelElectrode


def xic_to_list(xic):
    es = []
    for line in xic:
        line = line.rstrip()
        if line[0] == "P":
            p = [line.split()[1:]]
        elif line[0] == " ":
            if not line.endswith(";"):
                p.append(line.split())
            else:
                p.append(line[:-1].split())
                es.append(np.array([map(float, i) for i in p])*1e-8)
        elif not line.strip():
            pass
        else:
            pass # print line
    return es
