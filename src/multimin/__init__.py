##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                        #
#                                                                #
# Authors: Jorge I. Zuluaga                                      #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)       #
##################################################################

"""
MultiMin: Multivariate Gaussian fitting

This package provides tools for fitting composed multivariate normal
distributions (CMND) and other statistical utilities.

Main Features
-------------
- Multivariate fitting (CMND)
- Visualization tools (Density plots)
- Statistical utilities

Usage
-----
    >>> import multimin as mn


For more information, visit: https://github.com/seap-udea/multimin
"""

import os

# Import version from version.py
from .version import __version__

# =============================================================================
# PACKAGE METADATA
# =============================================================================
# Set package metadata
__name__ = "multimin"
__author__ = "Jorge I. Zuluaga"
__email__ = "jorge.zuluaga@udea.edu.co"
__url__ = "https://github.com/seap-udea/multimin"
__license__ = "AGPL-3.0-only"
__description__ = "MultiMin: Multivariate Gaussian fitting"

# Global option: set to False to disable watermarks on all plots (DensityPlot, plot_sample, plot_fit).
show_watermark = os.getenv("MULTIMIN_NO_WATERMARK", "").lower() not in (
    "1",
    "true",
    "yes",
)

# Print a nice welcome message
def welcome():
    print(f"Welcome to MultiMin v{__version__}. ¡Al infinito y más allá!")

if not os.getenv("MULTIMIN_NO_WELCOME"):
    welcome()

# =============================================================================
# IMPORT ALL CLASSES AND FUNCTIONS FROM MODULES
# =============================================================================

# Base class and utilities
from .base import MultiMinBase, ROOTDIR

# Utility classes
from .util import Util, Stats

# Plotting and visualization
from .plotting import DensityPlot, multimin_watermark

# CMND class
from .cmnd import ComposedMultiVariateNormal

# Fitting classes
from .fitting import FitCMND, FitFunctionCMND

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    "__license__",
    "__description__",
    # Configuration
    "show_watermark",
    "welcome",
    "ROOTDIR",
    # Base
    "MultiMinBase",
    # Utilities (nmd, tnmd, docstring_summary, props_to_properties available as Util static methods)
    "Util",
    "Stats",
    # Plotting
    "DensityPlot",
    "multimin_watermark",
    # CMND
    "ComposedMultiVariateNormal",
    # Fitting
    "FitCMND",
    "FitFunctionCMND",
]
