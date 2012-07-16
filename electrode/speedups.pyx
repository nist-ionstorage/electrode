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
cimport numpy as np

np.import_array()

from .utils import norm

def polygon_value(x, p, *d):
    p1 = x[None, :] - p[:, None]
    x1, y1, z = p1.transpose((2, 0, 1))
    r1 = norm(p1)
    x2 = np.roll(x1, -1, axis=0)
    y2 = np.roll(y1, -1, axis=0)
    r2 = np.roll(r1, -1, axis=0)
    l2 = (x1-x2)**2+(y1-y2)**2
    if 0 in d:
        zs = np.abs(z)
        yield np.arctan2(z*(x1*y2-y1*x2),
                zs*(r1*r2+x1*x2+y1*y2+zs*(zs+r1+r2))).sum(axis=0)/np.pi
    if 1 in d:
        yield (np.array([-(y1-y2)*z, (x1-x2)*z, x2*y1-x1*y2]
            )*(r1+r2)/(np.pi*r1*r2*((r1+r2)**2-l2))).sum(axis=1)
    if 2 in d:
        yield (np.array([(l2*(r2**3*x1+r1**3*x2)-
            (r1+r2)**2*(r2**2*(2*r1+r2)*x1+r1**2*(r1+2*r2)*x2))*(-y1+y2)*z,
              (-y1+y2)*(l2*(r2**3*y1+r1**3*y2)-
              (r1+r2)**2*(r2**2*(2*r1+r2)*y1+r1**2*(r1+2*r2)*y2))*z,
            (r1+r2)*(-y1+y2)*(-(l2*r1**2*r2**2)+
              l2*(r1**2-r1*r2+r2**2)*z**2+(r1+r2)**2*(r1**2*r2**2-
              (r1**2+r1*r2+r2**2)*z**2)),
            (x1-x2)*(l2*(r2**3*y1+r1**3*y2)-
              (r1+r2)**2*(r2**2*(2*r1+r2)*y1+r1**2*(r1+2*r2)*y2))*z,
            (r1+r2)*(-x1+x2)*(l2*r1**2*r2**2-l2*(r1**2-r1*r2+r2**2)*z**2+
              (r1+r2)**2*(-(r1**2*r2**2)+(r1**2+r1*r2+r2**2)*z**2))
            ])/(np.pi*(r1*r2)**3*((r1+r2)**2-l2)**2)).sum(axis=1)
    if 3 in d:
        yield (np.array([(-y1+y2)*(3*l2**2*(r2**5*x1*y1+r1**5*x2*y2)+
              (r1+r2)**3*(9*r1*r2**5*x1*y1+3*r2**6*x1*y1+3*r1**6*x2*y2+
              9*r1**5*r2*x2*y2+6*r1**3*r2**3*(x2*y1+x1*y2)+
              2*r1**2*r2**4*(x2*y1+x1*(4*y1+y2))+
              2*r1**4*r2**2*(x1*y2+x2*(y1+4*y2)))-
              2*l2*(9*r1*r2**6*x1*y1+3*r2**7*x1*y1+3*r1**7*x2*y2+
              9*r1**6*r2*x2*y2+r1**2*r2**5*(x2*y1+x1*(6*y1+y2))+
              r1**5*r2**2*(x1*y2+x2*(y1+6*y2))))*z,
            (-y1+y2)*(-(l2**2*(r1**2*r2**5*x1-3*r2**5*x1*z**2+
              r1**5*x2*(r2**2-3*z**2)))+2*l2*(r1+r2)**3*(-3*r2**4*x1*z**2+
              r1**4*x2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*x1-(x1+x2)*z**2))-
              (r1+r2)**2*(-12*r1*r2**6*x1*z**2-3*r2**7*x1*z**2+
              r1**7*x2*(r2**2-3*z**2)+4*r1**6*r2*x2*(r2**2-3*z**2)+
              r1**4*r2**3*(r2**2*(5*x1+2*x2)-8*(x1+2*x2)*z**2)+
              r1**5*r2**2*(r2**2*(2*x1+5*x2)-(2*x1+19*x2)*z**2)+
              r1**2*r2**3*(r2**4*x1-r2**2*(19*x1+2*x2)*z**2-
              6*x1*((x1-x2)**2+(y1-y2)**2)*z**2)+
              2*r1**3*r2**2*(2*r2**4*x1-4*r2**2*(2*x1+x2)*z**2-3*x2*(
              (x1-x2)**2+(y1-y2)**2)*z**2))),
            (-x1+x2)*(l2**2*(r1**2*r2**5*y1-3*r2**5*y1*z**2+
              r1**5*y2*(r2**2-3*z**2))-2*l2*(r1+r2)**3*(-3*r2**4*y1*z**2+
              r1**4*y2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*y1-(y1+y2)*z**2))+
              (r1+r2)**2*(-12*r1*r2**6*y1*z**2-3*r2**7*y1*z**2+
              r1**7*y2*(r2**2-3*z**2)+4*r1**6*r2*y2*(r2**2-3*z**2)+
              2*r1**3*r2**2*(2*r2**4*y1-3*((x1-x2)**2+(y1-y2)**2)*y2*z**2-
              4*r2**2*(2*y1+y2)*z**2)+r1**4*r2**3*(r2**2*(5*y1+2*y2)-
              8*(y1+2*y2)*z**2)+r1**2*r2**3*(r2**4*y1-6*y1*((x1-x2)**2+
              (y1-y2)**2)*z**2-r2**2*(19*y1+2*y2)*z**2)+
              r1**5*r2**2*(r2**2*(2*y1+5*y2)-(2*y1+19*y2)*z**2))),
            (-y1+y2)*(2*l2*(r1**2*r2**2*(r1+r2)**3*(r1**2+r2**2)-
              3*r2**5*(r1+r2)*(2*r1+r2)*y1**2-2*r1**2*r2**2*(r1**3+
              r2**3)*y1*y2-3*r1**5*(r1+r2)*(r1+2*r2)*y2**2)-
              l2**2*(r1**2*r2**5-3*r2**5*y1**2+r1**5*(r2**2-3*y2**2))-
              (r1+r2)**3*(-9*r1*r2**5*y1**2-3*r2**6*y1**2+
              3*r1**3*r2**3*(r2**2-4*y1*y2)+r1**6*(r2**2-3*y2**2)+
              3*r1**5*r2*(r2**2-3*y2**2)+r1**2*r2**4*(r2**2-4*y1*(2*y1+y2))+
              4*r1**4*r2**2*(r2**2-y2*(y1+2*y2))))*z,
            (-r1-r2)*(-y1+y2)*z*(-2*l2*(r1+r2)**2*(3*r1**2*r2**2*(r1**2+
              r2**2)+(-3*r1**4+2*r1**3*r2+r1**2*r2**2+2*r1*r2**3-
              3*r2**4)*z**2)+3*l2**2*(r1**2*r2**2*(r1**2-r1*r2+r2**2)-
              (r1**4-r1**3*r2+r1**2*r2**2-r1*r2**3+r2**4)*z**2)+
              (r1+r2)**2*(-3*r2**6*z**2+r1*r2**3*(-9*r2**2+4*((x1-x2)**2+
              (y1-y2)**2))*z**2+3*r1**2*r2**4*(r2**2-4*z**2)+
              3*r1**6*(r2**2-z**2)+9*r1**5*r2*(r2**2-z**2)+
              12*r1**4*r2**2*(r2**2-z**2)+r1**3*r2*(9*r2**4-12*r2**2*z**2+
              4*((x1-x2)**2+(y1-y2)**2)*z**2))),
            (r1+r2)*(-x1+x2)*z*(-2*l2*(r1+r2)**2*(3*r1**2*r2**2*(r1**2+
              r2**2)+(-3*r1**4+2*r1**3*r2+r1**2*r2**2+2*r1*r2**3-
              3*r2**4)*z**2)+3*l2**2*(r1**2*r2**2*(r1**2-r1*r2+r2**2)-
              (r1**4-r1**3*r2+r1**2*r2**2-r1*r2**3+r2**4)*z**2)+
              (r1+r2)**2*(-3*r2**6*z**2+r1*r2**3*(-9*r2**2+4*((x1-x2)**2+
              (y1-y2)**2))*z**2+3*r1**2*r2**4*(r2**2-4*z**2)+
              3*r1**6*(r2**2-z**2)+9*r1**5*r2*(r2**2-z**2)+
              12*r1**4*r2**2*(r2**2-z**2)+r1**3*r2*(9*r2**4-12*r2**2*z**2+
              4*((x1-x2)**2+(y1-y2)**2)*z**2))),
            (y1-y2)*(l2**2*(r1**2*r2**5*y1-3*r2**5*y1*z**2+
              r1**5*y2*(r2**2-3*z**2))-2*l2*(r1+r2)**3*(-3*r2**4*y1*z**2+
              r1**4*y2*(r2**2-3*z**2)+r1**2*r2**2*(r2**2*y1-(y1+y2)*z**2))+
              (r1+r2)**2*(-12*r1*r2**6*y1*z**2-3*r2**7*y1*z**2+
              r1**7*y2*(r2**2-3*z**2)+4*r1**6*r2*y2*(r2**2-3*z**2)+
              2*r1**3*r2**2*(2*r2**4*y1-3*((x1-x2)**2+
              (y1-y2)**2)*y2*z**2-4*r2**2*(2*y1+y2)*z**2)+
              r1**4*r2**3*(r2**2*(5*y1+2*y2)-8*(y1+2*y2)*z**2)+
              r1**2*r2**3*(r2**4*y1-6*y1*((x1-x2)**2+
              (y1-y2)**2)*z**2-r2**2*(19*y1+2*y2)*z**2)+
              r1**5*r2**2*(r2**2*(2*y1+5*y2)-(2*y1+19*y2)*z**2)))
            ])/(np.pi*(r1*r2)**5*((r1+r2)**2-l2)**3)).sum(axis=1)
