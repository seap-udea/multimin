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
Base class and utilities for MultiMin package.

Contains:
- MultiMinBase: Base class for all major classes
- ROOTDIR: Package root directory
"""

import os
import inspect
import copy

# Get package root directory
ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class MultiMinBase:
    """
    Base class for MultiMin package.

    All major classes in the package inherit from this base class,
    providing common functionality and attributes.
    """

    def __init__(self):
        pass

    @classmethod
    def describe(cls):
        """
        Show the list of public methods for this instance's class with their
        short description taken from each method's docstring.

        Can be called on an instance (e.g. obj.describe()) or on the class
        (e.g. mn.MultiPlot.describe()). Intended for discovery of available
        functionality on any MultiMinBase subclass (e.g. MultiPlot, MoG).
        """
        # Local import to avoid circular dependency
        from .util import Util

        methods = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            obj = getattr(cls, name)
            if not callable(obj):
                continue
            methods.append((name, obj))
        methods.sort(key=lambda x: x[0])
        lines = [
            f"\nAvailable methods for this object/class",
            "=" * (30 + len(cls.__name__)),
        ]
        for name, meth in methods:
            if name == "describe":
                continue
            doc = inspect.getdoc(meth)
            summary = Util.docstring_summary(doc) if doc else "(sin descripción)"
            summary = summary.replace("\n", " ").strip()
            # if len(summary) > 70:
            #     summary = summary[:67] + "..."
            lines.append(f"  {name}()")
            lines.append(f"    {summary}")
            lines.append("")
        print("\n".join(lines))

    def copy(self):
        """Return a copy of the object."""
        return copy.deepcopy(self)

    def __str__(self):
        """String representation of the object."""
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __repr__(self):
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}({self.__dict__})"
