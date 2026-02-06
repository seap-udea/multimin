##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                               #
#                                                                #
# Authors: Jorge I. Zuluaga                                     #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)        #
##################################################################

"""
MultiMin: Multivariate Gaussian fitting

This package provides tools for fitting composed multivariate normal
distributions (CMND) and other statistical utilities.

Main Features:
--------------
- Multivariate fitting (CMND)
- Visualization tools (Corner plots)
- Statistical utilities

Usage:
------
    >>> import multimin as mn
    >>> # Your analysis code here

For more information, visit: https://github.com/seap-udea/multimin
"""

import os
import warnings

# Import version from version.py
from .version import __version__

# Set package metadata
__name__ = "multimin"
__author__ = "Jorge I. Zuluaga"
__email__ = "jorge.zuluaga@udea.edu.co"
__url__ = "https://github.com/seap-udea/multimin"
__license__ = "AGPL-3.0-only"
__description__ = "MultiMin: Multivariate Gaussian fitting"


# Print a nice welcome message
def welcome():
    print(f"Welcome to MultiMin v{__version__}")


if not os.getenv("MULTIMIN_NO_WELCOME"):
    welcome()

ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class MultiMinBase:
    """
    Base class for MultiMin package.

    All major classes in the package inherit from this base class,
    providing common functionality and attributes.
    """

    def __init__(self):
        pass

    def __str__(self):
        """String representation of the object."""
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __repr__(self):
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}({self.__dict__})"


# Import main classes and functions from modules
from .multimin import ComposedMultiVariateNormal, FitCMND
from .plot import CornerPlot
from .util import Util

# Aliases
CMND = ComposedMultiVariateNormal


# Package initialization message (optional, can be removed in production)
# Clean up namespace
del warnings
