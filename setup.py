#!/usr/bin/python
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

#from __future__ import absolute_import, print_function, unicode_literals

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
from glob import glob
import numpy

setup(
        name = "electrode",
        version = "1.3+dev",
        author = "Robert Jordens",
        author_email = "jordens@gmail.com",
        url = "http://launchpad.net/electrode",
        description = "toolkit to develop and analyze rf surface ion traps",
        long_description = 
"""Electrode is a toolkit to develop and analyze RF ion traps. It can
optimize 2D surface electrode patterns to achieve desired trapping
properties and extract relevant parameters of the resulting geometry.
The software also treats precomputed 3D volumetric field and potential
data transparently.

See also:

[1] Roman Schmied <roman.schmied@unibas.ch>, SurfacePattern software
package.
http://atom.physik.unibas.ch/people/romanschmied/code/SurfacePattern.php

[2] Roman Schmied: Electrostatics of gapped and finite surface
electrodes. New Journal of Physics 12:023038 (2010).
http://dx.doi.org/10.1088/1367-2630/12/2/023038

[3] Roman Schmied, Janus H. Wesenberg, and Dietrich Leibfried: Optimal
Surface-Electrode Trap Lattices for Quantum Simulation with Trapped
Ions. Physical Review Letters 102:233002 (2009).
http://dx.doi.org/10.1103/PhysRevLett.102.233002

[4] A. van Oosterom and J. Strackee: The Solid Angle of a Plane
Triangle, IEEE Transactions on Biomedical Engineering, vol. BME-30, no.
2, pp. 125-126. (1983)
http://dx.doi.org/10.1109/TBME.1983.325207

[5] Mário H. Oliveira and José A. Miranda: Biot–Savart-like law in
electrostatics. European Journal of Physics 22:31 (2001).
http://dx.doi.org/10.1088/0143-0807/22/1/304
""",
        license = "GPLv3+",
        install_requires = [
            "numpy", "scipy", "matplotlib", "nose"],
        extras_require = {
            "notebooks": ["ipython>=0.12"],
            "integrate": ["qc"],
            "optimization": ["cvxopt>=1"],
            "visualization": ["mayavi>4"],
            "polygons": ["shapely>=1.2"],
            "gds": ["gdsii"],
            "speedups": ["cython"],
            },
        dependency_links = [],
        packages = find_packages(),
        namespace_packages = [],
        #test_suite = "electrode.tests.test_all",
        test_suite = "nose.collector",
        scripts = glob("notebooks/*.py"),
        include_package_data = True,
        #package_data = {"": ["notebooks/*.ipynb"]},
        ext_modules=[
                Extension("electrode._transformations",
                    sources=["electrode/transformations.c"],),
                Extension("electrode.cexpressions",
                    sources=["electrode/cexpressions.pyx",
                         #"electrode/cexpressions.c",
                         ],
                extra_compile_args=[
                        "-ffast-math", # improves expressions
                        #"-Wa,-adhlns=cexprssions.lst", # for amusement
                        ],
                include_dirs=[numpy.get_include()]),
            ],
        cmdclass = {"build_ext": build_ext},
        )
