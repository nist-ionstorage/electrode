#!/usr/bin/python

from setuptools import setup, find_packages
from glob import glob

setup(
        name = "electrode",
        version = "1.0",
        author = "Robert Jordens",
        author_email = "jordens@phys.ethz.ch",
        url = "http://launchpad.net/electrode",
        description = "toolkit to develop and analyze rf surface ion traps",
        license = "GPLv3+",
        install_requires = [
            "numpy", "scipy", "traits>=4", "matplotlib"],
        extras_require = {
            "notebooks": ["ipython>=0.12"],
            "integrate": ["qc"],
            "optimization": ["cvxopt>=1"],
            },
        dependency_links = [],
        packages = find_packages(),
        namespace_packages = [],
        test_suite = "electrode.tests.test_all",
        scripts = glob("notebooks/*.py"),
        include_package_data = True,
        #package_data = {"": ["notebooks/*.ipynb"]},
        )
