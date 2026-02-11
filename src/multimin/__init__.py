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

import inspect
import os
import math
import string
import itertools
import pickle
from hashlib import md5
from time import time
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal as multinorm
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

import numpy as np
import spiceypy as spy

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
# Can be set after import: import multimin as mn; mn.show_watermark = False
# Or via environment variable before import: MULTIMIN_NO_WATERMARK=1
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

ROOTDIR = os.path.dirname(os.path.abspath(__file__))


def _docstring_summary(doc):
    """
    Extract the first line or paragraph of a docstring as summary.

    Stops at common section headers (Parameters, Returns, etc.)
    so that the description does not include parameter lists.

    Parameters
    ----------
    doc : str or None
        The docstring.

    Returns
    -------
    str
        One-line summary or empty string if no docstring.
    """
    if not doc or not doc.strip():
        return ""
    doc = doc.strip()
    section_markers = (
        "Parameters",
        "Returns",
        "Examples",
        "Attributes",
        "Notes",
        "Methods",
        "Raises",
        "See Also",
        "Warnings",
    )
    lines = doc.splitlines()
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if any(
            stripped == marker
            or stripped.startswith(marker + " ")
            or stripped == marker + ":"
            for marker in section_markers
        ):
            break
        if stripped.endswith("---") or stripped.endswith("===="):
            break
        summary_lines.append(stripped)
    return " ".join(summary_lines).strip() if summary_lines else ""


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
        (e.g. mn.DensityPlot.describe()). Intended for discovery of available
        functionality on any MultiMinBase subclass (e.g. DensityPlot, CMND).
        """
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
            summary = _docstring_summary(doc) if doc else "(sin descripción)"
            summary = summary.replace("\n", " ").strip()
            # if len(summary) > 70:
            #     summary = summary[:67] + "..."
            lines.append(f"  {name}()")
            lines.append(f"    {summary}")
            lines.append("")
        print("\n".join(lines))

    def __str__(self):
        """String representation of the object."""
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __repr__(self):
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}({self.__dict__})"


# =============================================================================
# UTILITIES
# =============================================================================
class Util(MultiMinBase):
    """
    This abstract class contains useful methods for the package.


    """

    # Mathematical functions
    sin = np.sin
    cos = np.cos
    log = np.log
    exp = np.exp

    # Stores the time of start of the script when gravray is imported
    TIMESTART = time()
    # Stores the time of the last call of el_time
    TIME = time()
    # Stores the duration between el_time consecutive calls
    DTIME = -1
    DTIME = -1
    DUTIME = []

    @staticmethod
    def get_data(filename):
        """
        Get the full path of the `filename` which is one of the datafiles provided with the package.
        Ex. get_data("nea_extended.json.gz")

        Parameters
        ----------
        filename : str
            Name of the data file.

        Returns
        -------
        path : str
            Full path to package datafile.

        Examples
        --------
        >>> from multimin.util import Util
        >>> path=Util.get_data("nea_extended.json.gz")

        """
        return os.path.join(ROOTDIR, "data", filename)

    @staticmethod
    def f2u(x, s):
        """
        Convert from a finite interval [0,s] to an unbound one [-inf,inf].
        Ex. f2u(0.5, 1.0)

        Parameters
        ----------
        x : float or array_like
            Value in the interval [0,s].
        s : float
            Scale (upper limit of the interval).

        Returns
        -------
        u : float or array_like
            Unbound value.

        """
        return Util.log((x / s) / (1 - (x / s)))

    @staticmethod
    def u2f(t, s):
        """
        Convert from an unbound interval [-inf,inf] to a finite one [0,s].
        Ex. u2f(0.0, 1.0)

        Parameters
        ----------
        t : float or array_like
            Unbound value.
        s : float
            Scale (upper limit of the interval).

        Returns
        -------
        x : float or array_like
            Value in the interval [0,s].

        """
        return s / (1 + Util.exp(-t))

    @staticmethod
    def t_if(p, s, f):
        """
        Transform a set of parameters using a transformation function f and scales s.
        This routine allows the conversion from a finite interval [0,s] to an unbound one [-inf,inf]
        (using f=Util.f2u) or vice versa (using f=Util.u2f).
        Ex. Util.t_if(minparams, scales, Util.f2u)

        Parameters
        ----------
        p : list or array_like
            Parameters to transform.
        s : list or array_like
            Scales for each parameter. If s[i] > 0, the transformation is applied.
            If s[i] == 0, the parameter is unchanged.
        f : function
            Transformation function (e.g. Util.f2u or Util.u2f).

        Returns
        -------
        tp : list
            Transformed parameters.

        Examples
        --------
        >>> scales = [0, 0, 10, 10, 1]
        >>> minparams = [0.0, 0.0, 1, 1, 0.7]
        >>> uparams = Util.t_if(minparams, scales, Util.f2u)
        >>> print(uparams)
        [0.0, 0.0, -2.197224577336219, -2.197224577336219, 0.8472978603872034]

        """
        return [f(p[i], s[i]) if s[i] > 0 else p[i] for i in range(len(p))]

    @staticmethod
    def true_anomaly_to_mean_anomaly(e, f):
        """
        Convert true anomaly to mean anomaly.
        Ex. true_anomaly_to_mean_anomaly(0.1, 0.5)

        Parameters
        ----------
        e : float
            Eccentricity.
        f : float
            True anomaly.

        Returns
        -------
        M : float
            Mean anomaly.

        """
        # Calculate eccentric anomaly E from true anomaly f
        E = 2 * np.arctan(np.tan(f / 2) / np.sqrt((1 + e) / (1 - e)))
        M = E - e * np.sin(E)

        return M

    def error_msg(error, msg):
        """
        Add a custom message msg to an error handle.
        Ex. error_msg(error, 'custom message')

        Parameters
        ----------
        error : Exception
            Error handle (eg. except ValueError as error).
        msg : str
            Message to add to error.

        """
        error.args = (error.args if error.args else tuple()) + (msg,)

    def _t_unit(t):
        for unit, base in dict(
            d=86400, h=3600, min=60, s=1e0, ms=1e-3, us=1e-6, ns=1e-9
        ).items():
            tu = t / base
            if tu > 1:
                break
        return tu, unit, base

    def el_time(verbose=1, start=False):
        """
        Compute the time elapsed since last call of this routine.  The displayed time
        is preseneted in the more convenient unit, ns (nano seconds), us (micro seconds),
        ms (miliseconds), s (seconds), min (minutes), h (hours), d (days).
        Ex. el_time(), el_time(verbose=0), el_time(start=True)

        Parameters
        ----------
        verbose : int or bool, optional
            Show the time in screen (default 1).
        start : int or bool, optional
            Compute time from program start (default 0).

        Returns
        -------
        dt : float
            Elapsed time in seconds.
        dtu_unit : list
            List containing [time in units, unit string].

        Examples
        --------
        >>> Util.el_time() # basic usage (show output)
        >>> Util.el_time(verbose=0) # no output
        >>> Util.el_time(start=True) # measure elapsed time since program start
        >>> print(Util.DTIME, Util.DUTIME) # show values of elapsed time

        """
        t = time()
        dt = t - Util.TIME
        if start:
            dt = t - Util.TIMESTART
            msg = "since script start"
        else:
            msg = "since last call"
        dtu, unit, base = Util._t_unit(dt)
        if verbose:
            print("Elapsed time %s: %g %s" % (msg, dtu, unit))
        Util.DTIME = dt
        Util.DUTIME = [dtu, unit]
        Util.TIME = time()
        return dt, [dtu, unit]

    def mantisa_exp(x):
        """
        Calculate the mantisa and exponent of a number.
        Ex. m, e = Util.mantisa_exp(234.5)

        Parameters
        ----------
        x : float
            Number.

        Returns
        -------
        man : float
            Mantisa.
        exp : float
            Exponent.

        Examples
        --------
        >>> m, e = Util.mantisa_exp(234.5)
        # returns m=2.345, e=2
        >>> m, e = Util.mantisa_exp(-0.000023213)
        # return m=-2.3213, e=-5

        """
        xa = np.abs(x)
        s = np.sign(x)
        try:
            exp = int(np.floor(np.log10(xa)))
            man = s * xa / 10 ** (exp)
        except OverflowError as e:
            man = exp = 0
        return man, exp


class Stats(MultiMinBase):
    """
    Abstract class with useful routines


    """

    # Golden ratio: required for golden gaussian.
    phi = (1 + 5**0.5) / 2

    def gen_index(probs):
        """
        Given a set of (normalized) probabilities, randomly generate an index n following the
        probabilities. For instance if we have 3 events with probabilities 0.1, 0.7, 0.2, gen_index
        will generate a number in the set (0,1,2) having those probabilities, ie. 1 will have 70%.
        Ex. Stats.gen_index([0.1, 0.7, 0.2])

        Parameters
        ----------
        probs : numpy.ndarray
            Probabilities (N), adimensional.
            NOTE: It should be normalized, ie. sum(probs)=1

        Returns
        -------
        n : int
            Index in the set [0,1,2,... len(probs)-1].

        Examples
        --------
        >>> n = Stats.gen_index([0.1, 0.7, 0.2])

        """
        cums = np.cumsum(probs)
        if not math.isclose(cums[-1], 1, rel_tol=1e-5):
            raise ValueError("Probabilities must be normalized, ie. sum(probs) = 1")
        cond = (np.random.rand() - cums) < 0
        isort = np.arange(len(probs))
        n = isort[cond][0] if sum(cond) > 0 else isort[0]
        return n

    def set_matrix_off_diagonal(M, off):
        """
        Set a matrix with the terms of the off diagonal.
        Ex. Stats.set_matrix_off_diagonal(M, [0.1, 0.2, 0.3])

        Parameters
        ----------
        M : numpy.ndarray
            Matrix (n x n).
        off : list or numpy.ndarray
            Terms off diagonal (n x (n-1) / 2).

        Returns
        -------
        None
            Implicitly the matrix M has now the off diagonal terms.

        Examples
        --------
        >>> M = np.eye(3)
        >>> off = [0.1, 0.2, 0.3]
        >>> Stats.set_matrix_off_diagonal(M, off)
        >>> print(M)
        [[1. , 0.1, 0.2],
         [0.1, 1. , 0.3],
         [0.2, 0.3, 1. ]]

        """
        I, J = np.where(~np.eye(M.shape[0], dtype=bool))
        ffo = list(off[::-1])
        for i, j in zip(I, J):
            M[i, j] = ffo.pop() if j > i else 0
        M[:, :] = np.triu(M) + np.tril(M.T, -1)

    def calc_covariance_from_correlations(sigmas, rhos):
        """
        Compute covariance matrices from the standard deviations and correlations (rho).
        Ex. Stats.calc_covariance_from_correlations(sigmas, rhos)

        Parameters
        ----------
        sigmas : numpy.ndarray
            Array of values of standard deviation for variables (ngauss x nvars).
        rhos : numpy.ndarray
            Array with correlations (ngauss x nvars x (nvars-1)/2).

        Returns
        -------
        Sigmas : numpy.ndarray
            Array with covariance matrices corresponding to these sigmas and rhos (ngauss x nvars x nvars).

        Examples
        --------
        >>> import numpy as np
        >>> sigmas = np.array([[1, 2, 3]])
        >>> # rho_12, rho_13, rho_23
        >>> rhos = np.array([[0.1, 0.2, 0.3]])
        >>> S = Stats.calc_covariance_from_correlations(sigmas, rhos)
        >>> print(S)
        [[[1.  0.2 0.6]
          [0.2 4.  1.8]
          [0.6 1.8 9. ]]]

        This is equivalent to:

        >>> rho = rhos[0]
        >>> sigma = sigmas[0]
        >>> R = np.eye(3)
        >>> Stats.set_matrix_off_diagonal(R, rho)
        >>> M = np.zeros((3, 3))
        >>> for i in range(3):
        ...     for j in range(3):
        ...         M[i,j] = R[i,j] * sigma[i] * sigma[j]
        >>> print(M)
        [[1.  0.2 0.6]
         [0.2 4.  1.8]
         [0.6 1.8 9. ]]

        """
        try:
            nvars = len(sigmas[0])
        except:
            raise AssertionError("Array of sigmas must be an array of arrays")
        try:
            Nrhos = len(rhos[0])
        except:
            raise AssertionError("Array of rhos must be an array of arrays")

        Noff = int(nvars * (nvars - 1) / 2)
        if Nrhos != Noff:
            raise AssertionError(
                f"Size of rhos ({Nrhos}) are incompatible with nvars={nvars}.  It should be nvars(nvars-1)/2={Noff}."
            )

        Sigmas = np.array(len(sigmas) * [np.eye(nvars)])
        for Sigma, sigma, rho in zip(Sigmas, sigmas, rhos):
            Stats.set_matrix_off_diagonal(Sigma, rho)
            Sigma *= np.outer(sigma, sigma)
        return Sigmas

    def calc_correlations_from_covariances(Sigmas):
        """
        Compute the standard deviations and corresponding correlation coefficients given a set of
        covariance matrices.
        Ex. sigmas, rhos = Stats.calc_correlations_from_covariances(Sigmas)

        Parameters
        ----------
        Sigmas : numpy.ndarray
            Array of covariance matrices (ngauss x nvars x nvars).

        Returns
        -------
        sigmas : numpy.ndarray
            Array of standard deviations (ngauss x nvars).
        rhos : numpy.ndarray
            Array of correlation coefficients (ngauss x nvars * (nvars-1) / 2).

        Examples
        --------
        >>> Sigmas = [
        ...     [[1. , 0.2, 0.6],
        ...      [0.2, 4. , 1.8],
        ...      [0.6, 1.8, 9. ]]
        ... ]
        >>> sigmas, rhos = Stats.calc_correlations_from_covariances(Sigmas)
        >>> print(sigmas)
        [1. 2. 3.]
        >>> print(rhos)
        [[0.1 0.2 0.3]]


        """
        if len(np.array(Sigmas).shape) != 3:
            raise AssertionError(
                f"Array of Sigmas (shape {np.array(Sigmas).shape}) must be an array of matrices"
            )

        sigmas = []
        rhos = []
        for n, Sigma in enumerate(np.array(Sigmas)):
            sigmas += [(np.diag(Sigma)) ** 0.5]
            R = Sigma / np.outer(sigmas[n], sigmas[n])
            I, J = np.where(~np.eye(R.shape[0], dtype=bool))
            rhos += [[]]
            for i, j in zip(I, J):
                rhos[n] += [R[i, j]] if j > i else []
        return np.array(sigmas), np.array(rhos)

    def calc_covariance_from_rotation(sigmas, angles):
        """
        Compute covariance matrices from the stds and the angles.
        Ex. Stats.calc_covariance_from_rotation(sigmas, angles)

        Parameters
        ----------
        sigmas : numpy.ndarray
            Array of values of standard deviation for variables (ngauss x 3).
        angles : numpy.ndarray
            Euler angles expressing the directions of the principal axes of the distribution (ngauss x 3).

        Returns
        -------
        Sigmas : numpy.ndarray
            Array with covariance matrices corresponding to these sigmas and angles (ngauss x 3 x 3).

        """
        try:
            nvars = len(sigmas[0])
        except:
            raise AssertionError("Sigmas must be an array of arrays")
        if nvars == 1:
            # Univariate: covariance is just variance (sigma^2) per component
            return np.array([[[scale[0] ** 2]] for scale in sigmas])
        Sigmas = []
        for scale, angle in zip(sigmas, angles):
            L = np.identity(nvars) * np.outer(np.ones(nvars), scale)
            Rot = (
                spy.eul2m(-angle[0], -angle[1], -angle[2], 3, 1, 3)
                if nvars == 3
                else spy.rotate(-angle[0], 3)[:2, :2]
            )
            Sigmas += [np.matmul(np.matmul(Rot, np.matmul(L, L)), np.linalg.inv(Rot))]

        return np.array(Sigmas)

    def flatten_symmetric_matrix(M):
        """
        Given a symmetric matrix the routine returns the flatten version of the Matrix.
        Ex. Stats.flatten_symmetric_matrix(M)

        Parameters
        ----------
        M : numpy.ndarray
            Matrix (n x n).

        Returns
        -------
        F : numpy.ndarray
            Flatten array (nx(n+1)/2).

        Examples
        --------
        >>> M = np.array([[1, 0.2], [0.2, 3]])
        >>> F = Stats.flatten_symmetric_matrix(M)
        >>> print(F)
        [1.  0.2 3. ]

        """
        return M[np.triu_indices(M.shape[0], k=0)]

    def unflatten_symmetric_matrix(F, M):
        """
        Given a flatten version of a matrix, returns the symmetric matrix.
        Ex. Stats.unflatten_symmetric_matrix(F, M)

        Parameters
        ----------
        F : numpy.ndarray
            Flatten array (n x (n+1)/2).
        M : numpy.ndarray
            Matrix where the result will be stored (n x n).

        Returns
        -------
        None
            It return the results in matrix M.

        Examples
        --------
        >>> F = [1, 0.2, 3]
        >>> M = np.zeros((2, 2))
        >>> Stats.unflatten_symmetric_matrix(F, M)
        >>> print(M)
        [[1.  0.2]
         [0.2 3. ]]

        """
        M[np.triu_indices(M.shape[0], k=0)] = np.array(F)
        M[:, :] = np.triu(M) + np.tril(M.T, -1)


# =============================================================================
# SIMPLE MULTIVARIATE NORMAL PDF
# =============================================================================
def nmd(X, mu, Sigma):
    """PDF of a multivariate normal at x; mu=mean vector, Sigma=covariance matrix.
    Ex. nmd(X, mu, Sigma)

    Parameters
    ----------
    X : array-like
        Point(s) at which to evaluate the PDF. Shape (n,) or (N, n).
    mu : array-like
        Mean vector, shape (n,).
    Sigma : array-like
        Covariance matrix, shape (n, n). For univariate (n=1), a 1x1 matrix or scalar variance.

    Returns
    -------
    float or ndarray
        PDF value(s). Scalar if x is 1D, array if x is 2D.
    """
    if isinstance(X, float):
        value = norm.pdf(X, mu, Sigma)
    else:
        value = multinorm.pdf(X, mu, Sigma)

    return value


def tnmd(X, mu, Sigma, a, b, Z=None):
    """PDF of a truncated (multivariate) normal at X; single evaluation like nmd. Uses
    scipy.stats.truncnorm for 1D (one call). For nD, returns normal PDF in the box [a, b]
    divided by the normalization constant Z.
    Ex. tnmd(X, mu, Sigma, a, b)

    Parameters
    ----------
    X : array-like
        Point(s). Shape (n,) or (N, n).
    mu : array-like
        Mean vector, shape (n,).
    Sigma : array-like
        Covariance matrix (n, n). For n=1, scalar variance or 1x1.
    a, b : array-like or float
        Truncation bounds (lower, upper). For 1D, scalars; for nD, vectors of length n.
    Z : float, optional
        Normalization constant P(a < X < b). For nD, pass to avoid extra computation.

    Returns
    -------
    float or ndarray
        PDF value(s). Zero outside [a, b].
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    Sig = np.asarray(Sigma, dtype=float)
    n_dim = (
        1
        if mu.size == 1 and (Sig.size == 1 or (Sig.ndim == 2 and Sig.shape[0] == 1))
        else (Sig.shape[0] if Sig.ndim >= 2 else int(mu.size))
    )
    univariate = n_dim == 1
    X = np.asarray(X, dtype=float)
    if univariate:
        x_flat = np.atleast_1d(X).ravel()
        mu0 = float(mu.ravel()[0])
        var = float(Sig.ravel()[0])
        sigma = np.sqrt(var)
        a0, b0 = float(a), float(b)
        a_std = (a0 - mu0) / sigma
        b_std = (b0 - mu0) / sigma
        out = truncnorm.pdf(x_flat, a_std, b_std, loc=mu0, scale=sigma)
        return float(out[0]) if out.size == 1 else out
    # nD: one multinorm.pdf, then divide by Z
    X = np.atleast_2d(X)
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()[:n_dim]
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()[:n_dim]
    in_box = np.all((X >= a) & (X <= b), axis=1)
    raw = multinorm.pdf(X, mu, Sig)
    if Z is None:
        Z = _norm_const_box(mu, Sig, a, b)
    out = np.where(in_box, np.maximum(raw / Z, 1e-300), 0.0)
    return float(out[0]) if out.size == 1 else out


def _norm_const_box(mu, Sigma, a, b):
    """P(a < X < b) for X ~ N(mu, Sigma). Used when Z is None in tnmd (nD)."""
    n = len(mu)
    if n == 1:
        sig = np.sqrt(float(np.asarray(Sigma).ravel()[0]))
        return float(norm.cdf((b[0] - mu[0]) / sig) - norm.cdf((a[0] - mu[0]) / sig))
    draws = multinorm.rvs(mu, Sigma, size=50000, random_state=42)
    in_box = np.all((draws >= a) & (draws <= b), axis=1)
    return max(float(np.mean(in_box)), 1e-300)


# =============================================================================
# VISUALIZATION
# =============================================================================
def multimin_watermark(ax, frac=1 / 4, alpha=1):
    """Add a water mark to a 2d or 3d plot.

    Parameters:

        ax: Class axes:
            Axe where the pryngles mark will be placed.
    """
    if not show_watermark:
        return None
    # Get the height of axe
    axh = (
        ax.get_window_extent()
        .transformed(ax.get_figure().dpi_scale_trans.inverted())
        .height
    )
    fig_factor = frac * axh

    # Options of the water mark
    args = dict(
        rotation=270,
        ha="left",
        va="top",
        transform=ax.transAxes,
        color="pink",
        fontsize=8 * fig_factor,
        zorder=100,
        alpha=alpha,
    )

    # Text of the water mark
    mark = f"MultiMin {__version__}"

    # Choose the according to the fact it is a 2d or 3d plot
    try:
        ax.add_collection3d
        plt_text = ax.text2D
    except:
        plt_text = ax.text

    text = plt_text(1, 1, mark, **args)
    return text


def _props_to_properties(properties, nvars, ranges=None):
    """Convert properties (list or DensityPlot-style dict) to properties dict for DensityPlot.

    - If properties is a dict: use as-is (DensityPlot-style); each value must have 'label'
      and optionally 'range'. Keys define variable names; first nvars keys are used.
    - If properties is a list or sequence: use each element as variable name and as axis
      label, with range=None unless ranges is provided (backward compatible).
    - If properties is None: use ascii_letters and ranges from the ranges argument.
    """
    if properties is None:
        return {
            string.ascii_letters[i]: dict(
                label=f"${string.ascii_letters[i]}$",
                range=ranges[i] if ranges is not None and i < len(ranges) else None,
            )
            for i in range(nvars)
        }
    if hasattr(properties, "keys"):
        keys = list(properties.keys())[:nvars]
        out = {}
        for i, k in enumerate(keys):
            v = properties[k]
            if isinstance(v, dict):
                out[k] = dict(
                    label=v.get("label", str(k)),
                    range=v.get("range") if "range" in v else None,
                )
            else:
                out[k] = dict(label=str(v), range=None)
        return out
    # list or sequence: use elements as names and labels, range from ranges
    return {
        (str(properties[i]) if i < len(properties) else string.ascii_letters[i]): dict(
            label=str(properties[i])
            if i < len(properties)
            else f"${string.ascii_letters[i]}$",
            range=ranges[i] if ranges is not None and i < len(ranges) else None,
        )
        for i in range(nvars)
    }


class DensityPlot(MultiMinBase):
    """
    Create a grid of plots showing the projection of a N-dimensional data.

    Parameters
    ----------
    properties : dict
        List of properties to be shown, dictionary of dictionaries (N entries).
        Keys are label of attribute, ex. "q".
        Dictionary values:

        * label: label used in axis, string
        * range: range for property, tuple (2)
    figsize : int, optional
        Base size for panels (the size of figure will be M x figsize), default 3.
    fontsize : int, optional
        Base fontsize, default 10.
    direction : str, optional
        Direction of ticks in panels, default 'out'.

    Attributes
    ----------
    N : int
        Number of properties.
    M : int
        Size of grid matrix (M=N-1).
    fw : int
        Figsize.
    fs : int
        Fontsize.
    fig : matplotlib.figure.Figure
        Figure handle.
    axs : numpy.ndarray
        Matrix with subplots, axes handles (MxM).
    axp : dict
        Matrix with subplots, dictionary of dictionaries.
    properties : list
        List of properties labels, list of strings (N).

    Methods
    -------
    tight_layout()
        Tight layout if no constrained_layout was used.
    set_labels(**args)
        Set labels parameters.
    set_ranges()
        Set ranges in panels according to ranges defined in dparameters.
    set_tick_params(**args)
        Set tick parameters.
    plot_hist(data, colorbar=False, **args)
        Create a 2d-histograms of data on all panels of the DensityPlot.
    scatter_plot(data, **args)
        Scatter plot on all panels of the DensityPlot.

    """

    def __init__(self, properties, figsize=3, fontsize=10, direction="out"):

        # Basic attributes
        self.dproperties = properties
        self.properties = list(properties.keys())

        # Secondary attributes
        self.N = len(properties)
        self.M = max(1, self.N - 1)  # 1 when univariate so we have one panel
        self._univariate = self.N == 1

        # Optional properties
        self.fw = figsize
        self.fs = fontsize

        # Univariate: single 1D panel
        if self._univariate:
            from matplotlib import pyplot as plt

            self.fig, ax = plt.subplots(
                1, 1, constrained_layout=True, figsize=(self.fw * 1.5, self.fw)
            )
            self.axs = np.array([[ax]])
            self.constrained = True
            self.single = True
            self.axp = dict()
            prop0 = self.properties[0]
            self.axp[prop0] = {prop0: ax}
            ax.set_xlabel(self.dproperties[prop0]["label"], fontsize=fontsize)
            self.tight_layout()
            return

        # Create figure and axes: it works
        try:
            from matplotlib import pyplot as plt

            self.fig, self.axs = plt.subplots(
                self.M,
                self.M,
                constrained_layout=True,
                figsize=(self.M * self.fw, self.M * self.fw),
                sharex="col",
                sharey="row",
            )
            self.constrained = True
        except:
            self.fig, self.axs = plt.subplots(
                self.M,
                self.M,
                figsize=(self.M * self.fw, self.M * self.fw),
                sharex="col",
                sharey="row",
            )
            self.constrained = False

        if not isinstance(self.axs, np.ndarray):
            self.axs = np.array([[self.axs]])
            self.single = True
        else:
            self.single = False

        # Create named axis
        self.axp = dict()
        for j in range(self.N):
            propj = self.properties[j]
            if propj not in self.axp.keys():
                self.axp[propj] = dict()
            for i in range(self.N):
                propi = self.properties[i]
                if i == j:
                    continue
                if propi not in self.axp.keys():
                    self.axp[propi] = dict()
                if i < j:
                    self.axp[propj][propi] = self.axp[propi][propj]
                    continue
                self.axp[propj][propi] = self.axs[i - 1][j]

        # Deactivate unused panels
        for i in range(self.M):
            for j in range(i + 1, self.M):
                self.axs[i][j].axis("off")

        # Place ticks
        for i in range(self.M):
            for j in range(i + 1):
                if not self.single:
                    self.axs[i, j].tick_params(axis="both", direction=direction)
                else:
                    self.axs[i, i].tick_params(axis="both", direction=direction)
        for i in range(self.M):
            self.axs[i, 0].tick_params(axis="y", direction="out")
            self.axs[self.M - 1, i].tick_params(axis="x", direction="out")

        # Set properties of panels
        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()

    def tight_layout(self):
        """
        Tight layout if no constrained_layout was used.


        """
        if self.constrained == False:
            self.fig.subplots_adjust(wspace=self.fw / 100.0, hspace=self.fw / 100.0)
        self.fig.tight_layout()

    def set_tick_params(self, **args):
        """
        Set tick parameters.
        Ex. set_tick_params(labelsize=10)

        Parameters
        ----------
        **args : dict
            Same arguments as tick_params method.


        """
        opts = dict(axis="both", which="major", labelsize=0.8 * self.fs)
        opts.update(args)
        for i in range(self.M):
            for j in range(self.M):
                self.axs[i][j].tick_params(**opts)

    def set_ranges(self):
        """
        Set ranges in panels according to ranges defined in dparameters.


        """
        if getattr(self, "_univariate", False):
            prop = self.properties[0]
            if self.dproperties[prop]["range"] is not None:
                self.axs[0][0].set_xlim(self.dproperties[prop]["range"])
            return
        for i, propi in enumerate(self.properties):
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue
                if self.dproperties[propi]["range"] is not None:
                    self.axp[propi][propj].set_xlim(self.dproperties[propi]["range"])
                if self.dproperties[propj]["range"] is not None:
                    self.axp[propi][propj].set_ylim(self.dproperties[propj]["range"])

    def set_labels(self, **args):
        """
        Set labels parameters.
        Ex. set_labels(fontsize=12)

        Parameters
        ----------
        **args : dict
            Common arguments of set_xlabel, set_ylabel and text.


        """
        opts = dict(fontsize=self.fs)
        opts.update(args)
        for i, prop in enumerate(self.properties[:-1]):
            label = self.dproperties[prop]["label"]
            self.axs[self.M - 1][i].set_xlabel(label, **opts)
        for i, prop in enumerate(self.properties[1:]):
            label = self.dproperties[prop]["label"]
            self.axs[i][0].set_ylabel(label, rotation=90, labelpad=10, **opts)

        for i in range(1, self.M):
            label = self.dproperties[self.properties[i]]["label"]
            self.axs[i - 1][i].text(
                0.5,
                0.0,
                label,
                ha="center",
                transform=self.axs[i - 1][i].transAxes,
                **opts,
            )
            # 270 if you want rotation
            self.axs[i - 1][i].text(
                0.0,
                0.5,
                label,
                rotation=270,
                va="center",
                transform=self.axs[i - 1][i].transAxes,
                **opts,
            )

        label = self.dproperties[self.properties[0]]["label"]
        if not self.single:
            self.axs[0][1].text(
                0.0,
                1.0,
                label,
                rotation=0,
                ha="left",
                va="top",
                transform=self.axs[0][1].transAxes,
                **opts,
            )

        label = self.dproperties[self.properties[-1]]["label"]
        # 270 if you want rotation
        self.axs[-1][-1].text(
            1.05,
            0.5,
            label,
            rotation=270,
            ha="left",
            va="center",
            transform=self.axs[-1][-1].transAxes,
            **opts,
        )

        self.tight_layout()

    def plot_hist(self, data, colorbar=False, **args):
        """
        Create a 2d-histograms of data on all panels of the DensityPlot.
        Ex. G.plot_hist(data, bins=100, cmap='viridis')

        Parameters
        ----------
        data : numpy.ndarray
            Data to be histogramed (n=len(data)), numpy array (nxN).
        colorbar : bool, optional
            Include a colorbar? (default False).
        **args : dict
            All arguments of hist2d method.

        Returns
        -------
        hist : list
            List of histogram instances.

        Examples
        --------
        >>> properties = {
        ...     'Q': {'label': r"$Q$", 'range': None},
        ...     'E': {'label': r"$C$", 'range': None},
        ...     'I': {'label': r"$I$", 'range': None},
        ... }
        >>> G = mm.DensityPlot(properties, figsize=3)
        >>> hargs = dict(bins=100, cmap='viridis')
        >>> hist = G.plot_hist(udata, **hargs)


        """
        opts = dict()
        opts.update(args)

        # Univariate: 1D histogram (same style as plot_sample)
        if getattr(self, "_univariate", False):
            ax = self.axs[0][0]
            hargs_1d = {k: v for k, v in opts.items() if k != "cmap"}
            if "bins" not in hargs_1d:
                hargs_1d["bins"] = min(50, max(10, len(data) // 20))
            if "density" not in hargs_1d:
                hargs_1d["density"] = True
            hargs_1d.setdefault("label", "sample histogram")
            ax.hist(data[:, 0], **hargs_1d)
            ax.yaxis.set_label_position("left")
            ax.set_ylabel("density")
            # Legend (univariate): if no twin yet, add legend for histogram only
            handles, labels = ax.get_legend_handles_labels()
            if handles and getattr(self, "_ax_twin", None) is None:
                ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles),
                    frameon=False,
                )
                self.fig.subplots_adjust(top=0.88)
            self.set_ranges()
            self.set_tick_params()
            self.tight_layout()
            if not getattr(self, "_watermark_added", False):
                multimin_watermark(
                    ax, frac=0.5
                )  # larger frac for single panel (match 2-panel size)
                self._watermark_added = True
            return []

        hist = []
        for i, propi in enumerate(self.properties):
            if self.dproperties[propi]["range"] is not None:
                xmin, xmax = self.dproperties[propi]["range"]
            else:
                xmin = data[:, i].min()
                xmax = data[:, i].max()
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue

                if self.dproperties[propj]["range"] is not None:
                    ymin, ymax = self.dproperties[propj]["range"]
                else:
                    ymin = data[:, j].min()
                    ymax = data[:, j].max()

                opts["range"] = [[xmin, xmax], [ymin, ymax]]
                h, xe, ye, im = self.axp[propi][propj].hist2d(
                    data[:, i], data[:, j], **opts
                )

                hist += [im]
                if colorbar:
                    # Create color bar
                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    divider = make_axes_locatable(self.axp[propi][propj])
                    cax = divider.append_axes("top", size="9%", pad=0.1)
                    self.fig.add_axes(cax)
                    cticks = np.linspace(h.min(), h.max(), 10)[2:-1]
                    self.fig.colorbar(
                        im,
                        ax=self.axp[propi][propj],
                        cax=cax,
                        orientation="horizontal",
                        ticks=cticks,
                    )
                    cax.xaxis.set_tick_params(
                        labelsize=0.5 * self.fs, direction="in", pad=-0.8 * self.fs
                    )
                    xt = cax.get_xticks()
                    xm = xt.mean()
                    m, e = Util.mantisa_exp(xm)
                    xtl = []
                    for x in xt:
                        xtl += ["%.1f" % (x / 10**e)]
                    cax.set_xticklabels(xtl)
                    cax.text(
                        0,
                        0.5,
                        r"$\times 10^{%d}$" % e,
                        ha="left",
                        va="center",
                        transform=cax.transAxes,
                        fontsize=6,
                        color="w",
                    )

        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
        return hist

    def scatter_plot(self, data, **args):
        """
        Scatter plot on all panels of the DensityPlot.
        Ex. G.scatter_plot(data, s=0.2, color='r')

        Parameters
        ----------
        data : numpy.ndarray
            Data to be histogramed (n=len(data)), numpy array (nxN).
        **args : dict
            All arguments of scatter method.

        Returns
        -------
        scatter : list
            List of scatter instances.

        Examples
        --------
        >>> sargs = dict(s=0.2, edgecolor='None', color='r')
        >>> hist = G.scatter_plot(udata, **sargs)


        """
        # Univariate: scatter on a twin y-axis so data range is independent of PDF/density
        if getattr(self, "_univariate", False):
            ax = self.axs[0][0]
            ax_twin = ax.twinx()
            x = data[:, 0]
            y_jitter = np.random.uniform(0, 1, size=len(x))
            sargs_1d = dict(args)
            sargs_1d.setdefault("label", "sample")
            sc = ax_twin.scatter(x, y_jitter, **sargs_1d)
            ax_twin.set_ylim(0, 1)
            ax_twin.set_yticks([])
            prop_name = self.properties[0]
            ax_twin.set_ylabel(
                "sample " + self.dproperties[prop_name]["label"], fontsize=self.fs
            )
            self._ax_twin = ax_twin  # store for reference
            # Legend: combine primary ax (e.g. histogram) and twin (sample scatter)
            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles),
                    frameon=False,
                )
                self.fig.subplots_adjust(top=0.88)  # room for legend above
            self.set_ranges()
            self.set_tick_params()
            self.tight_layout()
            if not getattr(self, "_watermark_added", False):
                multimin_watermark(
                    ax, frac=0.5
                )  # larger frac for single panel (match 2-panel size)
                self._watermark_added = True
            return [sc]

        scatter = []
        for i, propi in enumerate(self.properties):
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue
                scatter += [
                    self.axp[propi][propj].scatter(data[:, i], data[:, j], **args)
                ]

        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
        return scatter


# =============================================================================
# MULTIMIN
# =============================================================================


class ComposedMultiVariateNormal(MultiMinBase):
    r"""
    The Composed Multivariate Normal Distribution (CMND).

    We conjecture that any multivariate distribution function :math:`p(\tilde U):\Re^{N}\rightarrow\Re`,
    where :math:`\tilde U:(u_1,u_2,u_3,\ldots,u_N)` and :math:`u_i` are random variables, can be approximated
    with an arbitrary precision by a normalized linear combination of :math:`M` Multivariate Normal Distributions
    (MND):

    .. math::

        p(\tilde U) \approx \mathcal{C}_M(\tilde U; \{w_k\}_M, \{\mu_k\}_M, \{\Sigma_k\}_M) \equiv \sum_{i=1}^{M} w_i\;\mathcal{N}(\tilde U; \tilde \mu_i, \Sigma_i)

    where the multivariate normal :math:`\mathcal{N}(\tilde U; \tilde \mu, \Sigma)` with mean vector :math:`\tilde \mu`
    and covariance matrix :math:`\Sigma` is given by:

    .. math::

        \mathcal{N}(\tilde U; \tilde \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^{k} \det \Sigma}} \exp\left[-\frac{1}{2}(\tilde U - \tilde \mu)^{\rm T} \Sigma^{-1} (\tilde U - \tilde \mu)\right]

    The covariance matrix :math:`\Sigma` elements are defined as :math:`\Sigma_{ij} = \rho_{ij}\sigma_{i}\sigma_{j}`, where
    :math:`\sigma_i` is the standard deviation of :math:`u_i` and :math:`\rho_{ij}` is the correlation coefficient among variable
    :math:`u_i` and :math:`u_j` (:math:`-1<\rho_{ij}<1`, :math:`\rho_{ii}=1`).

    The normalization condition on :math:`p(\tilde U)` implies that the set of weights :math:`\{w_k\}_M` are also normalized,
    i.e., :math:`\sum_i w_i=1`.

    **Truncated multivariate distributions**

    In real problems the domain of the variables is often bounded. Starting from the unbounded multivariate normal
    :math:`\mathcal{N}_k(\tilde U; \tilde \mu, \Sigma)`, let :math:`T \subset \{1,\ldots,k\}` be the set of indices of
    truncated variables and :math:`a_i < b_i` the truncation bounds for :math:`i \in T`. The truncation region is

    .. math::

        A_T = \left\{ \tilde U \in \mathbb{R}^k \;:\; a_i \le \tilde U_i \le b_i \;\; \forall i \in T \right\},

    with the remaining coordinates :math:`i \notin T` unbounded. The partially truncated multivariate normal is

    .. math::

        \mathcal{TN}_T(\tilde U; \tilde\mu, \Sigma, \mathbf{a}_T, \mathbf{b}_T)
        = \frac{\mathcal{N}(\tilde U; \tilde\mu, \Sigma) \, \mathbf{1}_{A_T}(\tilde U)}
               {Z_T(\tilde\mu, \Sigma, \mathbf{a}_T, \mathbf{b}_T)},

    where :math:`\mathbf{1}_{A_T}` is the indicator of :math:`A_T` and the normalization constant is

    .. math::

        Z_T(\tilde\mu, \Sigma, \mathbf{a}_T, \mathbf{b}_T)
        = \int_{A_T} \mathcal{N}(\tilde T; \tilde\mu, \Sigma) \, d\tilde T
        = \mathbb{P}_{\tilde T \sim \mathcal{N}(\tilde\mu,\Sigma)}\!\left( \tilde T \in A_T \right).

    When a finite ``domain`` is set (e.g. ``domain=[[0, 1], None]``), the CMND uses truncated normals for the
    bounded variables so that the PDF is zero outside the domain and the mixture remains normalized.

    Attributes
    ----------
    ngauss : int
        Number of composed MND.
    nvars : int
        Number of random variables.
    mus : numpy.ndarray
        Array with average (mu) of random variables (ngauss x nvars).
    weights : numpy.ndarray
        Array with weights of each MND (ngauss).
        NOTE: These weights are normalized at the end.
    sigmas : numpy.ndarray
        Standard deviation of each variable(ngauss x nvars).
    rhos : numpy.ndarray
        Elements of the upper triangle of the correlation matrix (ngauss x nvars x (nvars-1)/2).
    Sigmas : numpy.ndarray
        Array with covariance matrices for each MND (ngauss x nvars x nvars).
    params : numpy.ndarray
        Parameters of the distribution in flatten form including symmetric elements of the covariance
        matrix (ngauss*(1+nvars+nvars*(nvars+1)/2)).
    stdcorr : numpy.ndarray
        Parameters of the distribution in flatten form, including upper triangle of the correlation
        matrix (ngauss*(1+nvars+nvars*(nvars+1)/2)).

    Notes
    -----
    There are several ways of initialize a CMND:

    1. Providing: ngauss and nvars
       In this case the class is instantiated with zero means, unitary dispersion and
       covariance matrix equal to Ngasus identity matrices nvars x nvars.

    2. Providing: params, nvars
       In this case you have a flatted version of the parametes (weights, mus, Sigmas)
       and want to instantiate the system.  All parameters are set and no other action
       is required.

    3. Providing: stdcorr, nvars
       In this case you have a flatted version of the parametes (weights, mus, sigmas, rhos)
       and want to instantiate the system.  All parameters are set and no other action
       is required.

    4. Providing: weights, mus, Sigmas (optional), domain (optional), normalize_weights (optional)
       In this case the basic properties of the CMND are set.
       For univariate (one variable), mus may be a 1-D array of means, e.g. [0, 2],
       and Sigmas a 1-D array of variances, e.g. [1.0, 0.25].
       domain: list of length nvars; each element is None (unbounded) or [low, high]
       (finite support). Example: [None, [0, 1], None] for variable 1 in [0, 1].
       normalize_weights: if True (default), weights are scaled so sum(weights)=1 (proper PDF).
       If False, weights are used as-is (sum may != 1); useful for function fitting with
       an overall scale (e.g. FitFunctionCMND).

    Examples
    --------
    Example 1: Initialization using explicit arrays for means and weights.

    >>> # Define means for 2 Gaussian components in 2D
    >>> mus = [[0, 0], [1, 1]]
    >>> # Define weights (normalization is handled automatically)
    >>> weights = [0.1, 0]
    >>> # Create the CMND object
    >>> MND1 = ComposedMultiVariateNormal(mus=mus, weights=weights)
    >>> # Set covariance matrices explicitly
    >>> MND1.set_sigmas([[[1, 0.2], [0, 1]], [[1, 0], [0, 1]]])
    >>> print(MND1)

    Example 2: Initialization using a flattened parameter array.

    >>> # Flattened parameters: [weights..., mus..., flattened_covariances...]
    >>> params = [0.1, 0.9, 0, 0, 1, 1, 1, 0.2, 0.2, 1, 1, 0, 0, 1]
    >>> # Create CMND object specifying number of variables
    >>> MND2 = ComposedMultiVariateNormal(params=params, nvars=2)
    >>> print(MND2)
    """

    # Control behavior
    _ignoreWarnings = True

    def __init__(
        self,
        ngauss=0,
        nvars=0,
        params=None,
        stdcorr=None,
        weights=None,
        mus=None,
        Sigmas=None,
        domain=None,
        normalize_weights=True,
    ):
        self.normalize_weights = bool(normalize_weights)
        # Method 1: initialize a simple instance
        if ngauss > 0:
            mus = [[0] * nvars] * ngauss
            weights = [1 / ngauss] * ngauss
            Sigmas = [np.eye(nvars)] * ngauss
            self.__init__(
                mus=mus,
                weights=weights,
                Sigmas=Sigmas,
                domain=domain,
                normalize_weights=self.normalize_weights,
            )

        # Method 2: initialize from flatten parameters
        elif params is not None:
            self.set_params(params, nvars)
            self._set_domain(domain, self.nvars)
            self._Z_cache = None

        # Method 3: initialize from flatten parameters
        elif stdcorr is not None:
            self.set_stdcorr(stdcorr, nvars)
            self._set_domain(domain, self.nvars)
            self._Z_cache = None

        # Method 4: initialize from explicit arrays
        else:
            # Basic attributes
            mus = np.array(mus, dtype=float)
            if mus.ndim == 1:
                # Univariate: mus = [mu1, mu2, ...] -> (ngauss, 1)
                self.ngauss = len(mus)
                self.nvars = 1
                self.mus = mus.reshape(-1, 1)
            else:
                try:
                    mus[0, 0]
                except Exception as e:
                    Util.error_msg(
                        e,
                        "Parameter 'mus' must be a vector (1-D) or matrix, e.g. mus=[0,1] or mus=[[0,0]]",
                    )
                    raise
                self.mus = mus
                self.ngauss = len(mus)
                self.nvars = len(mus[0])

            # Weights and normalization
            if weights is None:
                self.weights = [1] + (self.ngauss - 1) * [0]
            elif len(weights) != self.ngauss:
                raise ValueError(
                    f"Length of weights array ({len(weights)}) must be equal to number of MND ({self.ngauss})"
                )
            else:
                self._apply_weights(weights)

            # Domain: list of (low, high) per variable; None or [a,b] for each
            self._set_domain(domain, self.nvars)
            self._Z_cache = None

            # Secondary attributes
            if Sigmas is None:
                self.Sigmas = None
                self.params = None
            else:
                self.set_sigmas(Sigmas)

        self._nerror = 0

    def _set_domain(self, domain, nvars):
        """Set _domain_bounds from domain. domain: list length nvars of None or [a,b]."""
        if domain is None:
            self._domain_bounds = [(-np.inf, np.inf)] * nvars
            return
        if len(domain) != nvars:
            raise ValueError(
                f"domain must have length nvars ({nvars}), got {len(domain)}"
            )
        bounds = []
        for i, d in enumerate(domain):
            if d is None:
                bounds.append((-np.inf, np.inf))
            else:
                a, b = float(d[0]), float(d[1])
                if a >= b:
                    raise ValueError(f"domain[{i}] must have lower < upper, got {d}")
                bounds.append((a, b))
        self._domain_bounds = bounds

    def set_domain(self, domain):
        """
        Set the support domain for each variable (finite or infinite).
        Ex. set_domain([None, [0, 1], None])

        Parameters
        ----------
        domain : list of length nvars
            Each element is None (unbounded) or [low, high] (finite interval).
            Example: [None, [0, 1], None] for variable 1 bounded to [0, 1].
        """
        self._set_domain(domain, self.nvars)
        self._Z_cache = None

    def update_params(self, weights=None, mus=None, sigmas=None, rhos=None):
        """Update CMND parameters in-place using FitCMND-like syntax.

        This method updates the internal parameters of the composed distribution
        (means, standard deviations, and correlations) using the same broadcasting
        rules as :meth:`FitCMND.set_initial_params`.

        Only arguments provided are updated; other parameters keep their current
        values.

        Parameters
        ----------
        weights : array-like, optional
            Mixture weights. Must have length ``ngauss`` (or length 1 when
            ``ngauss==1``). If ``normalize_weights`` is True (default), weights
            are scaled so that ``sum(weights)=1``; in all cases weights are kept
            non-negative.
        mus : array-like, optional
            Means. Shape ``(ngauss, nvars)`` or ``(nvars,)`` (same for all components).
        sigmas : array-like, optional
            Standard deviations. Shape ``(ngauss, nvars)`` or ``(nvars,)``.
        rhos : array-like, optional
            Correlation coefficients (upper triangle). Shape ``(ngauss, Ncorr)`` or
            ``(Ncorr,)`` where ``Ncorr = nvars*(nvars-1)/2``.

        Returns
        -------
        None
        """
        weights = np.asarray(weights, dtype=float) if weights is not None else None
        mus = np.asarray(mus, dtype=float) if mus is not None else None
        sigmas = np.asarray(sigmas, dtype=float) if sigmas is not None else None
        rhos = np.asarray(rhos, dtype=float) if rhos is not None else None

        def _broadcast_2d(arr, shape_2d, name, dims_desc):
            """Broadcast to (ngauss, nvars) or (ngauss, Ncorr).

            If 1D, repeats the same row for all components.
            """
            if arr is None:
                return None
            arr = np.atleast_1d(arr)
            if arr.ndim == 1:
                if arr.shape[0] != shape_2d[1]:
                    raise ValueError(
                        f"{name} 1D must have length {shape_2d[1]} ({dims_desc}), got {arr.shape[0]}"
                    )
                arr = np.tile(arr, (self.ngauss, 1))
            else:
                arr = np.atleast_2d(arr)
                if arr.shape != shape_2d:
                    raise ValueError(
                        f"{name} must have shape {shape_2d} or ({dims_desc}), got {arr.shape}"
                    )
            return arr

        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if self.ngauss == 1 and w.size == 1:
                self._apply_weights(w)
            elif w.size != self.ngauss:
                raise ValueError(
                    f"weights must have length ngauss ({self.ngauss}), got {w.size}"
                )
            else:
                self._apply_weights(w)

        if getattr(self, "Sigmas", None) is not None:
            # Ensure sigmas/rhos exist and are consistent.
            self._check_sigmas()
        else:
            if sigmas is not None or rhos is not None:
                raise ValueError(
                    "Cannot reset sigmas/rhos because covariance matrices are not set; "
                    "initialize the CMND with Sigmas or stdcorr first."
                )

        if mus is not None:
            mus = _broadcast_2d(mus, (self.ngauss, self.nvars), "mus", "nvars")
            self.mus = np.asarray(mus, dtype=float)

        if sigmas is not None:
            sigmas = _broadcast_2d(sigmas, (self.ngauss, self.nvars), "sigmas", "nvars")
            self.sigmas = np.asarray(sigmas, dtype=float)

        if rhos is not None:
            rhos = _broadcast_2d(
                rhos, (self.ngauss, self._n_offdiag()), "rhos", "Ncorr"
            )
            self.rhos = np.asarray(rhos, dtype=float)

        if sigmas is not None or rhos is not None:
            self.Sigmas = Stats.calc_covariance_from_correlations(
                self.sigmas, self.rhos
            )
            self._check_sigmas()

        self._flatten_params()
        self._flatten_stdcorr()
        self._Z_cache = None

    def _n_offdiag(self):
        """Number of off-diagonal correlation coefficients per component."""
        return int(self.nvars * (self.nvars - 1) / 2)

    def _in_domain(self, X):
        """Return mask of points inside the domain box. X: (N x nvars) or (nvars,)."""
        X = np.atleast_2d(X)
        out = np.ones(X.shape[0], dtype=bool)
        for j, (lo, hi) in enumerate(self._domain_bounds):
            out &= (X[:, j] >= lo) & (X[:, j] <= hi)
        return out

    def _normalization_constant(self, k, mc_samples=50000):
        """Integral of Gaussian k over the domain box (cached). For nvars>=2 uses
        scipy.stats.multivariate_normal.cdf (Genz-type quadrature) when available,
        else Monte Carlo."""
        from scipy.stats import multivariate_normal as mvn

        if getattr(self, "_Z_cache", None) is not None and self._Z_cache[k] is not None:
            return self._Z_cache[k]
        if self._Z_cache is None:
            self._Z_cache = [None] * self.ngauss
        mu = self.mus[k]
        Sigma = self.Sigmas[k]
        bounds = self._domain_bounds
        if self.nvars == 1:
            self._Z_cache[k] = 1.0
            return 1.0
        # Multivariate: Z = P(lower < X < upper). Use scipy MVN CDF (Genz algorithm) if available.
        lower = np.array([bounds[j][0] for j in range(self.nvars)])
        upper = np.array([bounds[j][1] for j in range(self.nvars)])
        try:
            # cdf(upper, mean=mu, cov=Sigma, lower_limit=lower) = P(lower < X < upper)
            Z = float(mvn.cdf(upper, mean=mu, cov=Sigma, lower_limit=lower))
        except (AttributeError, TypeError, ValueError):
            # Old scipy or unsupported: fall back to Monte Carlo
            samples = mvn.rvs(mu, Sigma, size=mc_samples)
            in_box = self._in_domain(samples)
            Z = np.mean(in_box).astype(float)
        if Z <= 0:
            Z = 1e-300
        self._Z_cache[k] = Z
        return Z

    def set_sigmas(self, Sigmas):
        """
        Set the value of list of covariance matrices. After setting Sigmas it update params and stdcorr.
        Ex. set_sigmas([[[1, 0.2], [0.2, 1]]])

        Parameters
        ----------
        Sigmas : list or numpy.ndarray
            Array of covariance matrices. For univariate (nvars=1), may be a 1-D
            array of variances, e.g. [1.0, 0.25] for two components.


        """
        Sigmas = np.array(Sigmas, dtype=float)
        if Sigmas.ndim == 1:
            # Univariate: list of variances -> (ngauss, 1, 1)
            self.Sigmas = np.array([[[s]] for s in Sigmas])
        else:
            self.Sigmas = Sigmas
        self._check_sigmas()
        self._flatten_params()
        self._flatten_stdcorr()
        self._Z_cache = None

    def set_params(self, params, nvars):
        """
        Set the properties of the CMND from flatten params. After setting it generate flattend stdcorr
        and normalize weights.
        Ex. set_params(params, nvars=2)

        Parameters
        ----------
        params : list or numpy.ndarray
            Flattened parameters.
        nvars : int
            Number of variables.


        """
        if nvars == 0 or len(params) == 0:
            raise ValueError(
                f"When setting from flat params, nvars ({nvars}) cannot be zero"
            )
        self._unflatten_params(params, nvars)
        self._normalize_weights(self.weights)
        self._Z_cache = None
        return

    def set_stdcorr(self, stdcorr, nvars):
        """
        Set the properties of the CMND from flatten stdcorr. After setting it generate flattened
        params and normalize weights.
        Ex. set_stdcorr(stdcorr, nvars=2)

        Parameters
        ----------
        stdcorr : list or numpy.ndarray
            Flattened standard deviations and correlations.
        nvars : int
            Number of variables.


        """
        if nvars == 0 or len(stdcorr) == 0:
            raise ValueError(
                f"When setting from flat params, nvars ({nvars}) cannot be zero"
            )
        self._unflatten_stdcorr(stdcorr, nvars)
        self._normalize_weights(self.weights)
        self._Z_cache = None
        return

    def _apply_weights(self, weights):
        """
        Set weights. If normalize_weights is True, scale so sum(weights)=1.
        If False, use weights as-is (allows sum != 1 for function fitting).
        Weights are kept non-negative in both cases.
        """
        w = np.array(weights, dtype=float)
        w = np.maximum(w, 1e-10)
        if self.normalize_weights:
            s = w.sum()
            if s <= 0:
                s = 1.0
            w = w / s
        self.weights = w

    def _normalize_weights(self, weights):
        """
        Normalize weights in such a way that sum(weights)=1.
        Delegates to _apply_weights for compatibility.
        """
        self._apply_weights(weights)

    def _flatten_params(self):
        """
        Flatten params


        """
        self._check_params(self.Sigmas)

        # Flatten covariance matrix
        SF = [
            Stats.flatten_symmetric_matrix(self.Sigmas[i]).tolist()
            for i in range(self.ngauss)
        ]
        self.params = np.concatenate(
            (self.weights.flatten(), self.mus.flatten(), list(itertools.chain(*SF)))
        )
        self.Npars = len(self.params)  # ngauss*(1+nvars+Nvar*(nvars+1)/2)

    def _flatten_stdcorr(self):
        """
        Flatten stdcorr


        """
        self._check_params(self.sigmas)

        # Flatten stds. and correlations
        self.stdcorr = np.concatenate(
            (
                self.weights.flatten(),
                self.mus.flatten(),
                self.sigmas.flatten(),
                self.rhos.flatten(),
            )
        )
        self.Ncor = len(self.stdcorr)

    def _unflatten_params(self, params, nvars):
        """
        Unflatten properties from params


        """

        self.params = np.array(params)
        self.Npars = len(self.params)

        factor = int(1 + nvars + nvars * (nvars + 1) / 2)

        if (self.Npars % factor) != 0:
            raise AssertionError(
                f"The number of parameters {self.Npars} is incompatible with the provided number of variables ({nvars})"
            )

        # Number of gaussian functions
        ngauss = int(self.Npars / factor)

        # Get the weights
        i = 0
        weights = self.params[i:ngauss]
        i += ngauss

        # Get the mus
        mus = self.params[i : i + ngauss * nvars].reshape(ngauss, nvars)
        i += ngauss * nvars

        # Get the sigmas
        Nsym = int(nvars * (nvars + 1) / 2)
        Sigmas = np.zeros((ngauss, nvars, nvars))
        [
            Stats.unflatten_symmetric_matrix(F, Sigmas[i])
            for i, F in enumerate(
                self.params[i : i + ngauss * Nsym].reshape(ngauss, Nsym)
            )
        ]

        # Apply weights (normalize or use as-is per self.normalize_weights)
        self._normalize_weights(weights)

        # Check Sigmas
        self.nvars = nvars
        self.ngauss = ngauss
        self.mus = mus
        self.Sigmas = Sigmas
        self._check_sigmas()

        # Flatten correlations
        self._flatten_stdcorr()

    def _unflatten_stdcorr(self, stdcorr, nvars):
        """
        Unflatten properties from stdcorr


        """

        self.stdcorr = np.array(stdcorr)
        self.Ncor = len(self.stdcorr)

        factor = int(1 + nvars + nvars * (nvars + 1) / 2)

        if (self.Ncor % factor) != 0:
            raise AssertionError(
                f"The number of parameters {self.Ncor} is incompatible with the provided number of variables ({nvars})"
            )

        # Number of gaussian functions
        ngauss = int(self.Ncor / factor)

        # Get the weights
        i = 0
        weights = self.stdcorr[i:ngauss]
        i += ngauss

        # Get the mus
        mus = self.stdcorr[i : i + ngauss * nvars].reshape(ngauss, nvars)
        i += ngauss * nvars

        # Get the sigmas
        sigmas = self.stdcorr[i : i + ngauss * nvars].reshape(ngauss, nvars)
        i += ngauss * nvars

        # Get the rhos
        Noff = int(nvars * (nvars - 1) / 2)
        rhos = self.stdcorr[i : i + ngauss * Noff].reshape(ngauss, Noff)

        # Apply weights (normalize or use as-is per self.normalize_weights)
        self._normalize_weights(weights)

        # Set properties
        self.nvars = nvars
        self.ngauss = ngauss
        self.mus = mus
        self.sigmas = sigmas
        self.rhos = rhos

        # Generate Sigma
        self.Sigmas = Stats.calc_covariance_from_correlations(self.sigmas, self.rhos)
        self._check_sigmas()

        # Flatten params
        self._flatten_params()

    def _check_sigmas(self):
        """
        Check value of sigmas


        """
        self._check_params(self.Sigmas)

        # Check matrix
        if len(self.Sigmas) != self.ngauss:
            raise ValueError(
                f"You provided {len(self.Sigmas)} matrix, but ngauss={self.ngauss} are required"
            )

        elif self.Sigmas[0].shape != (self.nvars, self.nvars):
            raise ValueError(
                f"Matrices have wrong dimensions ({self.Sigmas[0].shape}). It should be {self.nvars}x{self.nvars}"
            )

        # Symmetrize
        for i in range(self.ngauss):
            self.Sigmas[i] = np.triu(self.Sigmas[i]) + np.tril(self.Sigmas[i].T, -1)
            """
            #This check can be done, but it can be a problem when fitting
            if not np.all(np.linalg.eigvals(self.Sigmas[i])>0):
                raise ValueError(f"Matrix {i+1}, {self.Sigmas[i].tolist()} is not positive semidefinite.")
            """

        # Get sigmas and correlations
        self.sigmas, self.rhos = Stats.calc_correlations_from_covariances(self.Sigmas)

    def _check_params(self, checkvar=None):
        """
        Check if parameters are set.


        """
        if checkvar is None:
            raise AssertionError(
                "You must first set the parameters (Sigmas, mus, etc.)"
            )

    def pdf(self, X):
        """
        Compute the PDF.
        Ex. cmnd.pdf([1.0, 0.5])

        Parameters
        ----------
        X : float or numpy.ndarray
            Point in the nvars-dimensional space (scalar when nvars=1, or length nvars).

        Returns
        -------
        p : float
            PDF value at X.


        """
        self._check_params(self.params)
        self._nerror = 0
        X = np.atleast_1d(np.asarray(X, dtype=float))
        if X.ndim == 1:
            if self.nvars == 1:
                X2 = X.reshape(-1, 1)
            elif len(X) == self.nvars:
                X2 = X.reshape(1, -1)
            else:
                raise ValueError(
                    f"Point X has length {len(X)} but this CMND has nvars={self.nvars}"
                )
        elif X.ndim == 2:
            if X.shape[1] != self.nvars:
                raise ValueError(
                    f"Points X have {X.shape[1]} dimensions but this CMND has nvars={self.nvars}"
                )
            X2 = X
        else:
            raise ValueError(
                "X must be a point (length nvars) or array of points (N x nvars)"
            )
        in_dom = self._in_domain(X2)
        has_finite_domain = any(
            np.isfinite(self._domain_bounds[j][0])
            or np.isfinite(self._domain_bounds[j][1])
            for j in range(self.nvars)
        )
        from scipy.stats import multivariate_normal as multinorm

        value = np.zeros(X2.shape[0])
        if has_finite_domain:
            value[~in_dom] = 0.0
        for k, (w, muvec, Sigma) in enumerate(zip(self.weights, self.mus, self.Sigmas)):
            try:
                if has_finite_domain and self.nvars == 1:
                    lo, hi = self._domain_bounds[0]
                    mu0 = float(muvec.ravel()[0])
                    sig = np.sqrt(float(Sigma.ravel()[0]))
                    a = (lo - mu0) / sig if np.isfinite(lo) else -np.inf
                    b = (hi - mu0) / sig if np.isfinite(hi) else np.inf
                    val_k = w * truncnorm.pdf(X2[:, 0], a, b, loc=mu0, scale=sig)
                elif has_finite_domain and self.nvars >= 2:
                    Zk = self._normalization_constant(k)
                    val_k = np.atleast_1d(w * multinorm.pdf(X2, muvec, Sigma) / Zk)
                    val_k = val_k.copy()
                    val_k[~in_dom] = 0.0
                else:
                    val_k = w * multinorm.pdf(X2, muvec, Sigma)
                value += val_k
            except Exception as error:
                if not self._ignoreWarnings:
                    print(
                        f"Error: {error}, params = {self.params.tolist()}, stdcorr = {self.params.tolist()}"
                    )
                    self._nerror += 1
        return value.item() if value.size == 1 else value

    def plot_pdf(
        self,
        properties=None,
        ranges=None,
        figsize=3,
        grid_size=200,
        cmap="Spectral_r",
        colorbar=False,
    ):
        """Plot only the PDF of the CMND.

        For univariate distributions (``nvars==1``) this produces a single curve.
        For multivariate distributions (``nvars>=2``) this produces density panels
        using :class:`DensityPlot`. For ``nvars>2`` the density shown in each panel
        is a 2D *slice* of the full PDF, evaluated at the weighted mean for the
        remaining variables.

        Parameters
        ----------
        properties : list or dict, optional
            Axis specification, same format accepted by :meth:`plot_sample`.
        ranges : list, optional
            Ranges per variable; used when ``properties`` is a list or None.
        figsize : int, optional
            Size of each axis (default 3).
        grid_size : int, optional
            Number of grid points per axis for density evaluation (default 200).
        cmap : str, optional
            Matplotlib colormap name (default ``'Spectral_r'``).
        colorbar : bool, optional
            Include a colorbar for the first density panel (default False).

        Returns
        -------
        G : DensityPlot
            Handle to the density plot grid.
        """
        self._check_params(self.params)
        properties = _props_to_properties(properties, self.nvars, ranges)
        G = DensityPlot(properties, figsize=figsize)

        def _var_range(var_index, prop_key):
            """Choose plotting range for a variable."""
            if properties[prop_key].get("range") is not None:
                return properties[prop_key]["range"]
            bounds = getattr(self, "_domain_bounds", None)
            if bounds is not None:
                lo, hi = bounds[var_index]
                if np.isfinite(lo) and np.isfinite(hi):
                    return [float(lo), float(hi)]
            mu_min = float(np.min(self.mus[:, var_index]))
            mu_max = float(np.max(self.mus[:, var_index]))
            sig_max = float(np.max(self.sigmas[:, var_index]))
            nsig = 4.0
            lo = mu_min - nsig * sig_max
            hi = mu_max + nsig * sig_max
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = mu_min - 1.0, mu_max + 1.0
            return [lo, hi]

        prop_keys = list(properties.keys())[: self.nvars]

        if self.nvars == 1:
            prop0 = prop_keys[0]
            x_min, x_max = _var_range(0, prop0)
            x_grid = np.linspace(float(x_min), float(x_max), int(grid_size))
            pdf_vals = np.asarray(self.pdf(x_grid.reshape(-1, 1)), dtype=float)
            ax = G.axs[0][0]
            ax.plot(x_grid, pdf_vals, "k-", lw=2, label="PDF")
            if pdf_vals.size:
                ax.set_ylim(0, float(np.max(pdf_vals)) * 1.05)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles),
                    frameon=False,
                )
                G.fig.subplots_adjust(top=0.88)
            G.set_ranges()
            G.set_tick_params()
            G.tight_layout()
            if not getattr(G, "_watermark_added", False):
                multimin_watermark(ax, frac=0.5)
                G._watermark_added = True
            return G

        w = np.asarray(self.weights, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(w) / max(1, w.size)
        else:
            w = w / w_sum
        base_point = np.average(self.mus, axis=0, weights=w)

        first_im = None
        for j in range(1, self.nvars):
            for i in range(j):
                propi = prop_keys[i]
                propj = prop_keys[j]
                ax = G.axp[propi][propj]

                x_min, x_max = _var_range(i, propi)
                y_min, y_max = _var_range(j, propj)

                xs = np.linspace(float(x_min), float(x_max), int(grid_size))
                ys = np.linspace(float(y_min), float(y_max), int(grid_size))
                xx, yy = np.meshgrid(xs, ys, indexing="xy")
                pts = np.column_stack([xx.ravel(), yy.ravel()])

                X_full = np.tile(base_point, (pts.shape[0], 1))
                X_full[:, i] = pts[:, 0]
                X_full[:, j] = pts[:, 1]
                zz = np.asarray(self.pdf(X_full), dtype=float).reshape(xx.shape)

                im = ax.pcolormesh(xs, ys, zz, shading="auto", cmap=cmap)
                if first_im is None:
                    first_im = (ax, im)

        if colorbar and first_im is not None:
            ax0, im0 = first_im
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("top", size="9%", pad=0.1)
            G.fig.add_axes(cax)
            vmin = float(np.nanmin(im0.get_array()))
            vmax = float(np.nanmax(im0.get_array()))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                cticks = np.linspace(vmin, vmax, 8)[1:-1]
            else:
                cticks = None
            G.fig.colorbar(im0, ax=ax0, cax=cax, orientation="horizontal", ticks=cticks)
            cax.xaxis.set_tick_params(
                labelsize=0.5 * G.fs, direction="in", pad=-0.8 * G.fs
            )

        G.set_labels()
        G.set_ranges()
        G.set_tick_params()
        G.tight_layout()
        if not getattr(G, "_watermark_added", False):
            multimin_watermark(G.axs[0][0], frac=1 / 4 * G.axs.shape[0])
            G._watermark_added = True
        return G

    def rvs(self, Nsam=1, max_tries=100000):
        """
        Generate a random sample of points following this Multivariate distribution. When domain is
        finite, samples are drawn inside the domain (rejection sampling for multivariate; truncated
        normal for univariate).
        Ex. cmnd.rvs(1000)

        Parameters
        ----------
        Nsam : int, optional
            Number of samples (default 1).
        max_tries : int, optional
            Maximum attempts per sample when using rejection sampling (default 100000).

        Returns
        -------
        rs : numpy.ndarray
            Samples (Nsam x nvars).
        """
        self._check_params(self.params)

        from scipy.stats import multivariate_normal as multinorm

        has_finite_domain = any(
            np.isfinite(self._domain_bounds[j][0])
            or np.isfinite(self._domain_bounds[j][1])
            for j in range(self.nvars)
        )
        w_probs = (
            np.array(self.weights) / np.sum(self.weights)
            if not self.normalize_weights
            else self.weights
        )
        Xs = np.zeros((Nsam, self.nvars))
        if not has_finite_domain:
            for i in range(Nsam):
                n = Stats.gen_index(w_probs)
                Xs[i] = multinorm.rvs(self.mus[n], self.Sigmas[n])
            return Xs
        if self.nvars == 1:
            lo, hi = self._domain_bounds[0]
            for i in range(Nsam):
                k = Stats.gen_index(w_probs)
                mu0 = float(self.mus[k].ravel()[0])
                sig = np.sqrt(float(self.Sigmas[k].ravel()[0]))
                a = (lo - mu0) / sig if np.isfinite(lo) else -np.inf
                b = (hi - mu0) / sig if np.isfinite(hi) else np.inf
                Xs[i, 0] = truncnorm.rvs(a, b, loc=mu0, scale=sig)
            return Xs
        # Multivariate finite domain: rejection sampling
        for i in range(Nsam):
            k = Stats.gen_index(w_probs)
            for _ in range(max_tries):
                x = multinorm.rvs(self.mus[k], self.Sigmas[k])
                if self._in_domain(x[np.newaxis, :]).item():
                    Xs[i] = x
                    break
            else:
                raise RuntimeError(
                    f"rvs: failed to draw a sample inside domain after {max_tries} tries. "
                    "Domain may be too narrow for the current covariance."
                )
        return Xs

    def sample_cmnd_likelihood(
        self, uparams, data=None, pmap=None, tset="stdcorr", scales=[], verbose=0
    ):
        """
        Compute the negative value of the logarithm of the likelihood of a sample.
        Ex. cmnd.sample_cmnd_likelihood(uparams, data=data, pmap=pmap)

        Parameters
        ----------
        uparams : numpy.ndarray
            Minimization parameters (unbound).
        data : numpy.ndarray, optional
            Data for which log_l is computed.
        pmap : function, optional
            Routine to map from minparams to params or stdcorr.
            Example:
            >>> def pmap(minparams):
            ...     stdcorr = np.array([1] + list(minparams))
            ...     stdcorr[-1:] -= 1
            ...     return stdcorr
        tset : str, optional
            Type of minimization parameters. Values "params", "stdcorr" (default "stdcorr").
        scales : list, optional
            List of scales for transforming uparams (unbound) in minparams (natural scale).
        verbose : int, optional
            Verbosity level (0, none, 1: input parameters, 2: full definition of the CMND) (default 0).

        Returns
        -------
        log_l : float
            Negative log-likelihood.


        """
        # Map unbound minimization parameters into their right range
        minparams = np.array(Util.t_if(uparams, scales, Util.u2f))

        # Map minimizaiton parameters into CMND parameters
        params = np.array(pmap(minparams))

        if verbose >= 1:
            print("*" * 80)
            print(f"Minimization parameters: {minparams.tolist()}")
            print(f"CMND parameters: {params.tolist()}")

        # Update CMND parameters according to type of minimization parameters
        if tset == "params":
            self.set_params(params, self.nvars)
        else:
            self.set_stdcorr(params, self.nvars)

        if verbose >= 2:
            print("CMND:")
            print(self)

        # Compute PDF for each point in data and sum
        pdf_vals = self.pdf(data)
        # Avoid log(0) when PDF underflows or is zero at boundaries (bounded case)
        pdf_vals = np.atleast_1d(pdf_vals)
        pdf_vals = np.maximum(pdf_vals, 1e-300)
        log_l = -np.log(pdf_vals).sum()

        if verbose >= 1:
            print(f"-log_l = {log_l:e}")

        return log_l

    def plot_sample(
        self,
        data=None,
        N=10000,
        properties=None,
        ranges=None,
        figsize=2,
        sargs=None,
        hargs=None,
    ):
        """
        Plot a sample of the CMND.
        Ex. plot_sample(N=1000, sargs=dict(s=0.5))

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to plot. If None it generate a sample.
        N : int, optional
            Number of points to generate the sample (default 10000).
        properties : list or dict, optional
            Property names or DensityPlot-style properties. List (e.g. properties=["x","y"]): each
            element is used as axis label, range=None. Dict: same as DensityPlot, e.g.
            properties=dict(x=dict(label=r"$x$", range=None), y=dict(label=r"$y$", range=[-1,1])).
        ranges : list, optional
            Ranges per variable; used only when properties is a list or None. Ex. ranges=[[-3,3],[-5,5]].
        figsize : int, optional
            Size of each axis (default 2).
        sargs : dict, optional
            Dictionary with options for the scatter plot. Default: dict(s=0.5, edgecolor=None, color='b').
        hargs : dict, optional
            Dictionary with options for the hist2d function. Ex. hargs=dict(bins=50).

        Returns
        -------
        G : matplotlib.figure.Figure or DensityPlot
            Graphic handle. If nvars = 2, it is a figure object, otherwise is a DensityPlot instance.

        Examples
        --------
        Example 1: Plotting a generated sample.

        >>> # Generate and plot 10000 points
        >>> G = CMND.plot_sample(N=10000, sargs=dict(s=1, c='r'))
        >>> # Plot with histogram bins
        >>> G = CMND.plot_sample(N=1000, sargs=dict(s=1, c='r'), hargs=dict(bins=20))

        Example 2: Plotting for a 2D distribution.

        >>> CMND = ComposedMultiVariateNormal(ngauss=1, nvars=2)
        >>> fig = CMND.plot_sample(N=1000, hargs=dict(bins=20), sargs=dict(s=1, c='r'))

        Example 3: Defining and plotting a complex CMND.

        >>> # Define parameters for 2 Gaussians in 2D
        >>> params = [0.1, 0.9, 0.0, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.0, 1.0]
        >>> MND2 = ComposedMultiVariateNormal(params=params, nvars=2)
        >>> # Print the object details
        >>> print(MND2)
        >>> # Calculate PDF at a point
        >>> print(MND2.pdf([1, 1]))


        """
        if data is None:
            self.data = self.rvs(N)
        else:
            self.data = np.copy(data)

        # If not provided, use default scatter plot arguments
        if hargs is None and sargs is None:
            sargs = dict(s=0.5, edgecolor=None, color="b")

        properties = _props_to_properties(properties, self.nvars, ranges)
        G = DensityPlot(properties, figsize=figsize)
        ymax = -1e100
        if hargs is not None:
            G.plot_hist(self.data, **hargs)
            if self.nvars == 1:
                ymax = max(ymax, G.axs[0][0].get_ylim()[1])
        if sargs is not None:
            G.scatter_plot(self.data, **sargs)
            if self.nvars == 1:
                ymax = max(ymax, G.axs[0][0].get_ylim()[1])

        # When the distribution is univariate, add the PDF curve and legend
        if self.nvars == 1:
            x_min, x_max = self.data[:, 0].min(), self.data[:, 0].max()
            margin = max(1e-6, 0.1 * (x_max - x_min))
            x_curve = np.linspace(x_min - margin, x_max + margin, 300)
            pdf_vals = self.pdf(x_curve.reshape(-1, 1))
            G.axs[0][0].plot(x_curve, pdf_vals, "k-", lw=2, label="PDF")
            ymax = max(ymax, pdf_vals.max())
            G.axs[0][0].set_ylim(0, ymax)
            # Legend: combine primary ax (histogram, PDF) and twin (sample scatter)
            handles, labels = G.axs[0][0].get_legend_handles_labels()
            if getattr(G, "_ax_twin", None) is not None:
                h2, l2 = G._ax_twin.get_legend_handles_labels()
                handles, labels = handles + h2, labels + l2
            G.axs[0][0].legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(handles),
                frameon=False,
            )
            G.fig.subplots_adjust(top=0.88)  # room for legend above
        else:
            G.fig.tight_layout()
        if not getattr(G, "_watermark_added", False):
            multimin_watermark(G.axs[0][0], frac=0.5)  # univariate: larger watermark
        return G

    def _str_params(self):
        """
        Generate strings explaining which quantities are stored in the flatten arrays params and stdcorr.

        It also generate and aray with the bounds applicable to the stdcorr parameters for purposes
        of minimization.

        Returns
        -------
        None

        Notes
        -----
            Set the value of:
                - str_params: list of properties in params, str
                - str_stdcorr: list of properties in stdcorr, str.
                - bnd_stdcorr: bounds of properties in stdcorr applicable for transforming to unbound.
        """
        str_params = "["
        bnd_stdcorr = "["
        # Probabilities
        for n in range(self.ngauss):
            str_params += f"p{n + 1},"
            bnd_stdcorr += f"1,"

        # Mus
        for n in range(self.ngauss):
            for i in range(self.nvars):
                str_params += f"μ{n + 1}_{i + 1},"
                bnd_stdcorr += f"0,"

        str_stdcorr = str_params
        # Std. devs
        for n in range(self.ngauss):
            for i in range(self.nvars):
                str_stdcorr += f"σ{n + 1}_{i + 1},"
                bnd_stdcorr += f"10,"

        # Sigmas
        for n in range(self.ngauss):
            for i in range(self.nvars):
                for j in range(self.nvars):
                    if j >= i:
                        str_params += f"Σ{n + 1}_{i + 1}{j + 1},"
                    if j > i:
                        str_stdcorr += f"ρ{n + 1}_{i + 1}{j + 1},"
                        bnd_stdcorr += f"2,"

        self.str_params = str_params.strip(",") + "]"
        self.str_stdcorr = str_stdcorr.strip(",") + "]"
        self.bnd_stdcorr = bnd_stdcorr.strip(",") + "]"

    def __str__(self):
        """
        Generates a string version of the object
        """
        self.Npars = len(self.params)
        self.Ncor = len(self.stdcorr)

        self._str_params()

        msg = f"""Composition of ngauss = {self.ngauss} gaussian multivariates of nvars = {self.nvars} random variables:
    Weights: {self.weights.tolist()}
    Number of variables: {self.nvars}
    Averages (μ): {self.mus.tolist()}
"""
        if self.Sigmas is None:
            msg += f"""    Sigmas: (Not defined yet)
    Params: (Not defined yet)
"""
        else:
            msg += f"""    Standard deviations (σ): {self.sigmas.tolist()}
    Correlation coefficients (ρ): {self.rhos.tolist()}

    Covariant matrices (Σ): 
        {self.Sigmas.tolist()}
    Flatten parameters: 
        With covariance matrix ({self.Npars}):
            {self.str_params}
            {self.params.tolist()}
        With std. and correlations ({self.Ncor}):
            {self.str_stdcorr}
            {self.stdcorr.tolist()}"""
        return msg

    def _fmt_py_literal(self, arr, decimals=6):
        """Format array as a Python list literal for source code."""
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return str(round(float(arr), decimals))
        if arr.ndim == 1:
            return "[" + ", ".join(str(round(float(x), decimals)) for x in arr) + "]"
        return "[" + ", ".join(self._fmt_py_literal(row, decimals) for row in arr) + "]"

    def _fmt_latex_array(self, arr, decimals=6, single_line=True):
        """Format array as LaTeX \\begin{array}...\\end{array}.

        Parameters
        ----------
        arr : array-like
            Vector or matrix to format.
        decimals : int
            Decimal places for numeric values.
        single_line : bool, optional
            If True (default), output has no newlines so each $...$ formula fits on one
            line. This avoids RST/PyPI long_description errors when the LaTeX is
            pasted into README or similar.
        """
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return str(round(float(arr), decimals))
        sep = " \\\\ " if single_line else " \\\\\n        "
        if arr.ndim == 1:
            body = sep.join(str(round(float(x), decimals)) for x in arr)
            if single_line:
                return "\\begin{array}{c} " + body + " \\end{array}"
            return "\\begin{array}{c}\n        " + body + "\n    \\end{array}"
        # matrix
        nrows, ncols = arr.shape
        rows = [
            " & ".join(str(round(float(arr[i, j]), decimals)) for j in range(ncols))
            for i in range(nrows)
        ]
        body = sep.join(rows)
        colspec = "c" * ncols
        if single_line:
            return "\\begin{array}{" + colspec + "} " + body + " \\end{array}"
        return "\\begin{array}{" + colspec + "}\n        " + body + "\n    \\end{array}"

    def _var_names(self, properties):
        """Return list of variable names of length nvars. If properties is None, use '1','2',...; else use keys (dict) or elements (sequence)."""
        if properties is None:
            return [str(i + 1) for i in range(self.nvars)]
        if hasattr(properties, "keys"):
            names = list(properties.keys())
        else:
            names = list(properties)
        if len(names) < self.nvars:
            names = names + [str(i + 1) for i in range(len(names), self.nvars)]
        elif len(names) > self.nvars:
            names = names[: self.nvars]
        return names

    def get_function(self, print_code=True, decimals=6, type="python", properties=None):
        """
        Return the source code of ``cmnd(X)`` and an executable function (type='python'),
        or LaTeX code with parameters in \\begin{array} (type='latex').
        Ex. code, cmnd = CMND.get_function()

        Parameters
        ----------
        print_code : bool, optional
            If True (default), print the string to screen so it can be copied.
        decimals : int, optional
            Number of decimal places for numeric literals (default 6).
        type : str, optional
            - ``'python'`` (default): return Python source and callable.
            - ``'latex'``: return LaTeX formula with explicit parameters and matrices in \\begin{array}.
            LaTeX output is single-line per formula so it can be pasted into README or other
            RST-derived long descriptions without triggering PyPI/twine markup errors.
        properties : dict or sequence, optional
            If None (default), variable subscripts in parameters are numeric (mu_1, sigma_1, ...).
            If a dict (e.g. from DensityPlot), its keys are used as variable names (mu_x, sigma_x, ...).
            If a sequence, its elements are used in order. Length must match the number of variables.

        Returns
        -------
        tuple (str, callable or None)
            First element: source code (Python or LaTeX). Second: callable cmnd if type='python', else None.

        Examples
        --------
        >>> code, cmnd = CMND.get_function()
        >>> latex_str, _ = CMND.get_function(type='latex')
        """
        if self.Sigmas is None:
            raise ValueError("Sigmas not set; cannot generate get_function()")
        if type == "latex":
            return self._get_function_latex(
                print_code=print_code, decimals=decimals, properties=properties
            )
        # type == "python"
        var_names = self._var_names(properties)
        bounds = getattr(self, "_domain_bounds", None)
        has_finite_domain = bounds is not None and any(
            np.isfinite(bounds[j][0]) or np.isfinite(bounds[j][1])
            for j in range(self.nvars)
        )
        if has_finite_domain:
            lines = [
                "import numpy as np",
                "from multimin import tnmd",
                "",
                "def cmnd(X):",
                "",
            ]
            if self.nvars == 1:
                a0 = round(float(bounds[0][0]), decimals)
                b0 = round(float(bounds[0][1]), decimals)
                lines.append("    a = {}".format(a0))
                lines.append("    b = {}".format(b0))
            else:
                a_parts = [
                    str(round(float(bounds[j][0]), decimals))
                    if np.isfinite(bounds[j][0])
                    else "-np.inf"
                    for j in range(self.nvars)
                ]
                b_parts = [
                    str(round(float(bounds[j][1]), decimals))
                    if np.isfinite(bounds[j][1])
                    else "np.inf"
                    for j in range(self.nvars)
                ]
                lines.append("    a = [{}]".format(", ".join(a_parts)))
                lines.append("    b = [{}]".format(", ".join(b_parts)))
            lines.append("")
        else:
            lines = ["from multimin import nmd", "", "def cmnd(X):", ""]
        univariate = self.nvars == 1
        for n in range(self.ngauss):
            i = n + 1
            w = round(float(self.weights[n]), decimals)
            mu = self.mus[n]
            Sigma = self.Sigmas[n]
            if has_finite_domain:
                Zk = self._normalization_constant(n)
                Zk = round(float(Zk), decimals)
            if univariate:
                vname = var_names[0]
                mu_val = round(float(mu.ravel()[0]), decimals)
                var_val = round(float(Sigma.ravel()[0]), decimals)
                if has_finite_domain:
                    lines.append("    mu{}_{} = {}".format(i, vname, mu_val))
                    lines.append("    sigma{}_{} = {}".format(i, vname, var_val))
                    lines.append(
                        "    n{} = tnmd(X, mu{}_{}, sigma{}_{}, a, b)".format(
                            i, i, vname, i, vname
                        )
                    )
                else:
                    sigma_val = round(float(np.sqrt(Sigma.ravel()[0])), decimals)
                    lines.append("    mu{}_{} = {}".format(i, vname, mu_val))
                    lines.append("    sigma{}_{} = {}".format(i, vname, sigma_val))
                    lines.append(
                        "    n{} = nmd(X, mu{}_{}, sigma{}_{})".format(
                            i, i, vname, i, vname
                        )
                    )
            else:
                mu_parts = [
                    "mu{}_{} = {}".format(
                        i, var_names[v], round(float(mu.ravel()[v]), decimals)
                    )
                    for v in range(self.nvars)
                ]
                lines.extend("    " + p for p in mu_parts)
                mu_list_str = ", ".join(
                    "mu{}_{}".format(i, var_names[v]) for v in range(self.nvars)
                )
                lines.append("    mu{} = [{}]".format(i, mu_list_str))
                Sigma_str = self._fmt_py_literal(Sigma, decimals)
                lines.append("    Sigma{} = {}".format(i, Sigma_str))
                if has_finite_domain:
                    lines.append("    Z{} = {}".format(i, Zk))
                    lines.append(
                        "    n{} = tnmd(X, mu{}, Sigma{}, a, b, Z=Z{})".format(
                            i, i, i, i
                        )
                    )
                else:
                    lines.append("    n{} = nmd(X, mu{}, Sigma{})".format(i, i, i))
            lines.append("")
        for n in range(self.ngauss):
            i = n + 1
            w = round(float(self.weights[n]), decimals)
            lines.append("    w{} = {}".format(i, w))
        lines.append("")
        # Return on multiple lines: w1*n1 + w2*n2 + ...
        return_terms = ["    return (", "        w1*n1"]
        for n in range(1, self.ngauss):
            return_terms.append("        + w{}*n{}".format(n + 1, n + 1))
        return_terms.append("    )")
        lines.extend(return_terms)
        code = "\n".join(lines)
        if print_code:
            print(code)
        # Execute the code to obtain the callable cmnd
        namespace = {"__builtins__": __builtins__}
        try:
            exec(code, namespace)
            func = namespace["cmnd"]
        except Exception:
            func = None
        return code, func

    def _get_function_latex(self, print_code=True, decimals=6, properties=None):
        """Build LaTeX string for the CMND PDF with parameters in \\begin{array}."""
        parts = []
        var_names = self._var_names(properties)
        bounds = getattr(self, "_domain_bounds", None)
        has_finite_domain = bounds is not None and any(
            np.isfinite(bounds[j][0]) or np.isfinite(bounds[j][1])
            for j in range(self.nvars)
        )
        univariate = self.nvars == 1

        if has_finite_domain:
            # List variables with finite domain and their bounds
            parts.append(
                "Finite domain. The following variables are truncated (the rest are unbounded):"
            )
            parts.append("")
            for j in range(self.nvars):
                lo, hi = bounds[j][0], bounds[j][1]
                if np.isfinite(lo) and np.isfinite(hi):
                    vname = var_names[j]
                    parts.append(
                        "- Variable $x_{{{}}}$ (index {}): domain $[{}, {}]$.".format(
                            vname,
                            j + 1,
                            round(float(lo), decimals),
                            round(float(hi), decimals),
                        )
                    )
            parts.append("")
            parts.append(
                "Truncation region: $A_T = \\{\\tilde{U} \\in \\mathbb{R}^k : a_i \\le \\tilde{U}_i \\le b_i \\;\\forall i \\in T\\}$, with $T$ the set of truncated indices."
            )
            parts.append("")

        def _mu_sigma_sub(k):
            """Subscript for mean/sigma: numeric 'k' or 'k,vname' when properties given."""
            if properties is None:
                return str(k), str(k)
            v = var_names[0] if univariate else None
            return (str(k) + "," + v) if v else str(k), (
                str(k) + "," + v
            ) if v else str(k)

        if univariate:
            if has_finite_domain:
                if self.ngauss == 1:
                    msub, ssub = _mu_sigma_sub(1)
                    parts.append(
                        "$$f(x) = w_1 \\, \\mathcal{TN}(x; \\mu_{{{}}}, \\sigma_{{{}}}, a, b)$$".format(
                            msub, ssub
                        )
                    )
                else:
                    terms = [
                        "w_{} \\, \\mathcal{{TN}}(x; \\mu_{{{}}}, \\sigma_{{{}}}, a, b)".format(
                            k, *_mu_sigma_sub(k)
                        )
                        for k in range(1, self.ngauss + 1)
                    ]
                    parts.append("$$f(x) = " + " + ".join(terms) + "$$")
            else:
                if self.ngauss == 1:
                    msub, ssub = _mu_sigma_sub(1)
                    parts.append(
                        "$$f(x) = w_1 \\, "
                        "\\mathcal{N}(x; \\mu_{{{}}}, \\sigma_{{{}}})$$".format(
                            msub, ssub
                        )
                    )
                else:
                    terms = [
                        "w_{} \\, \\mathcal{{N}}(x; \\mu_{{{}}}, \\sigma_{{{}}})".format(
                            k, *_mu_sigma_sub(k)
                        )
                        for k in range(1, self.ngauss + 1)
                    ]
                    parts.append("$$f(x) = " + " + ".join(terms) + "$$")
        else:
            if has_finite_domain:
                if self.ngauss == 1:
                    parts.append(
                        "$$f(\\mathbf{x}) = w_1 \\, "
                        "\\mathcal{TN}_T(\\mathbf{x}; \\boldsymbol{\\mu}_1, \\mathbf{\\Sigma}_1, \\mathbf{a}_T, \\mathbf{b}_T)$$"
                    )
                else:
                    terms = [
                        "w_{0} \\, \\mathcal{{TN}}_T(\\mathbf{{x}}; \\boldsymbol{{\\mu}}_{0}, \\mathbf{{\\Sigma}}_{0}, \\mathbf{{a}}_T, \\mathbf{{b}}_T)".format(
                            k
                        )
                        for k in range(1, self.ngauss + 1)
                    ]
                    parts.append("$$f(\\mathbf{x}) = " + " + ".join(terms) + "$$")
            else:
                if self.ngauss == 1:
                    parts.append(
                        "$$f(\\mathbf{x}) = w_1 \\, "
                        "\\mathcal{N}(\\mathbf{x}; \\boldsymbol{\\mu}_1, \\mathbf{\\Sigma}_1)$$"
                    )
                else:
                    terms = [
                        "w_{0} \\, \\mathcal{{N}}(\\mathbf{{x}}; \\boldsymbol{{\\mu}}_{0}, \\mathbf{{\\Sigma}}_{0})".format(
                            k
                        )
                        for k in range(1, self.ngauss + 1)
                    ]
                    parts.append("$$f(\\mathbf{x}) = " + " + ".join(terms) + "$$")
        parts.append("")
        parts.append("where")
        parts.append("")
        if has_finite_domain and not univariate:
            a_list = [
                round(float(bounds[j][0]), decimals)
                if np.isfinite(bounds[j][0])
                else "-\\infty"
                for j in range(self.nvars)
            ]
            b_list = [
                round(float(bounds[j][1]), decimals)
                if np.isfinite(bounds[j][1])
                else "\\infty"
                for j in range(self.nvars)
            ]
            a_str = ", ".join(str(x) for x in a_list)
            b_str = ", ".join(str(x) for x in b_list)
            parts.append(
                "Bounds (vectors): $\\mathbf{a}_T = ("
                + a_str
                + ")^\\top$, $\\mathbf{b}_T = ("
                + b_str
                + ")^\\top$."
            )
            parts.append("")
        for n in range(self.ngauss):
            k = n + 1
            w = round(float(self.weights[n]), decimals)
            mu = self.mus[n]
            Sigma = self.Sigmas[n]
            if univariate:
                mu_val = round(float(mu.ravel()[0]), decimals)
                msub, ssub = _mu_sigma_sub(k)
                if has_finite_domain:
                    var_val = round(float(Sigma.ravel()[0]), decimals)
                    a0 = round(float(bounds[0][0]), decimals)
                    b0 = round(float(bounds[0][1]), decimals)
                    parts.append(
                        "$$w_{0} = {1},\\quad \\mu_{{{2}}} = {3},\\quad \\sigma_{{{4}}}^2 = {5},\\quad a = {6},\\quad b = {7}$$".format(
                            k, w, msub, mu_val, ssub, var_val, a0, b0
                        )
                    )
                else:
                    sigma_val = round(float(np.sqrt(Sigma.ravel()[0])), decimals)
                    parts.append(
                        "$$w_{0} = {1},\\quad \\mu_{{{2}}} = {3},\\quad \\sigma_{{{4}}} = {5}$$".format(
                            k, w, msub, mu_val, ssub, sigma_val
                        )
                    )
            else:
                parts.append("$$w_{} = {}$$".format(k, w))
                mu_arr = self._fmt_latex_array(mu, decimals)
                sig_arr = self._fmt_latex_array(Sigma, decimals)
                parts.append(
                    "$$\\boldsymbol{{\\mu}}_{} = \\left( {}\\right)$$".format(k, mu_arr)
                )
                parts.append(
                    "$$\\mathbf{{\\Sigma}}_{} = \\left( {}\\right)$$".format(k, sig_arr)
                )
            parts.append("")
        # Definition of the (truncated) normal distribution
        if has_finite_domain:
            parts.append("Truncated normal. The unbounded normal is")
            parts.append("")
            if univariate:
                parts.append(
                    "$$\\mathcal{N}(x; \\mu, \\sigma) = "
                    "\\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right).$$"
                )
            else:
                parts.append(
                    "$$\\mathcal{N}(\\mathbf{x}; \\boldsymbol{\\mu}, \\mathbf{\\Sigma}) = "
                    "\\frac{1}{\\sqrt{(2\\pi)^{{k}} \\det \\mathbf{\\Sigma}}} "
                    "\\exp\\left[-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^{\\top} "
                    "\\mathbf{\\Sigma}^{{-1}} (\\mathbf{x}-\\boldsymbol{\\mu})\\right].$$"
                )
            parts.append("")
            parts.append(
                "The truncation region is $A_T = \\{\\tilde{U} \\in \\mathbb{R}^k : a_i \\le \\tilde{U}_i \\le b_i \\;\\forall i \\in T\\}$. The partially truncated normal is"
            )
            parts.append("")
            parts.append(
                "$$\\mathcal{TN}_T(\\tilde{U}; \\tilde{\\mu}, \\Sigma, \\mathbf{a}_T, \\mathbf{b}_T) = "
                "\\frac{\\mathcal{N}(\\tilde{U}; \\tilde{\\mu}, \\Sigma) \\, \\mathbf{1}_{A_T}(\\tilde{U})}"
                "{Z_T(\\tilde{\\mu}, \\Sigma, \\mathbf{a}_T, \\mathbf{b}_T)},$$"
            )
            parts.append("")
            parts.append(
                "where $\\mathbf{1}_{A_T}$ is the indicator of $A_T$ and the normalization constant is"
            )
            parts.append("")
            parts.append(
                "$$Z_T(\\tilde{\\mu}, \\Sigma, \\mathbf{a}_T, \\mathbf{b}_T) = "
                "\\int_{A_T} \\mathcal{N}(\\tilde{T}; \\tilde{\\mu}, \\Sigma) \\, d\\tilde{T} = "
                "\\mathbb{P}_{\\tilde{T} \\sim \\mathcal{N}(\\tilde{\\mu},\\Sigma)}(\\tilde{T} \\in A_T).$$"
            )
        else:
            parts.append("Here the normal distribution is defined as:")
            parts.append("")
            if univariate:
                parts.append(
                    "$$\\mathcal{N}(x; \\mu, \\sigma) = "
                    "\\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)$$"
                )
            else:
                parts.append(
                    "$$\\mathcal{N}(\\mathbf{x}; \\boldsymbol{\\mu}, \\mathbf{\\Sigma}) = "
                    "\\frac{1}{\\sqrt{(2\\pi)^{{k}} \\det \\mathbf{\\Sigma}}} "
                    "\\exp\\left[-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^{\\top} "
                    "\\mathbf{\\Sigma}^{{-1}} (\\mathbf{x}-\\boldsymbol{\\mu})\\right]$$"
                )
        latex = "\n".join(parts).strip()
        if print_code:
            print(latex)
        return latex, None

    def tabulate(self, sort_by="weight", return_df=True, type="df", properties=None):
        """
        Build a table of CMND parameters: weights, means (mu_i), standard deviations (sigma_i),
        and correlations (rho_ij, i < j). Diagonal rho_ii are 1 by definition and omitted.
        Ex. df = CMND.tabulate(sort_by='weight')

        Parameters
        ----------
        sort_by : str, optional
            How to order the rows (one row per Gaussian component):
            - ``'weight'``: by weight descending (heaviest first).
            - ``'distance'``: by Euclidean distance of the mean vector to the origin ascending
              (closest to origin first).
        return_df : bool, optional
            If True (default), return a pandas DataFrame. If False, print the table and return None.
            Ignored when type='latex'.
        type : str, optional
            - ``'df'`` (default): return pandas DataFrame (or print and return None).
            - ``'latex'``: return a LaTeX tabular string suitable for papers.
        properties : dict or sequence, optional
            If None (default), column headers use numeric subscripts (mu_1, sigma_1, ...).
            If a dict (e.g. from DensityPlot), its keys are used (mu_x, sigma_x, ...).
            If a sequence, its elements are used in order.

        Returns
        -------
        pandas.DataFrame, str, or None
            DataFrame when type='df' and return_df=True; LaTeX string when type='latex';
            None when type='df' and return_df=False.
        """
        import pandas as pd

        var_names = self._var_names(properties)
        # Column names: w, mu_*.., sigma_*.., rho_*..
        cols = ["w"]
        cols += [f"mu_{name}" for name in var_names]
        cols += [f"sigma_{name}" for name in var_names]
        for i in range(self.nvars):
            for j in range(i + 1, self.nvars):
                cols.append(f"rho_{var_names[i]}{var_names[j]}")

        # Build rows (one per component)
        order = np.arange(self.ngauss)
        if sort_by == "weight":
            order = np.argsort(-np.asarray(self.weights))
        elif sort_by == "distance":
            dist = np.linalg.norm(self.mus, axis=1)
            order = np.argsort(dist)
        else:
            raise ValueError(f"sort_by must be 'weight' or 'distance', got {sort_by!r}")

        sigmas = getattr(self, "sigmas", None)
        rhos = getattr(self, "rhos", None)
        rows = []
        for k in order:
            w = self.weights[k]
            mu = self.mus[k]
            row = [float(w)] + [float(mu[i]) for i in range(self.nvars)]
            if sigmas is not None:
                # For univariate, show variance (Sigma) to match user input Sigmas=[0.01, 0.03]
                if self.nvars == 1:
                    row += [float(self.Sigmas[k].ravel()[0])]
                else:
                    sig = sigmas[k]
                    row += [float(sig[i]) for i in range(self.nvars)]
            else:
                row += [np.nan] * self.nvars
            Noff = self.nvars * (self.nvars - 1) // 2
            if rhos is not None and rhos.size > 0:
                rho = rhos[k]
                row += [float(rho[i]) for i in range(len(rho))]
            else:
                row += [np.nan] * Noff
            rows.append(row)

        if type == "latex":
            latex_output = self._tabulate_latex(cols, order, rows, var_names=var_names)
            print(latex_output)
            return latex_output

        df = pd.DataFrame(rows, columns=cols, index=np.array(order) + 1)
        df.index.name = "component"
        if not return_df:
            print(df.to_string())
            return None
        return df

    def _tabulate_latex(self, cols, order, rows, decimals=4, var_names=None):
        """Build LaTeX tabular string for the parameter table."""
        if var_names is None:
            var_names = [str(i + 1) for i in range(self.nvars)]
        colspec = "l" + "r" * (len(cols))
        header_parts = ["$k$", "$w$"]
        for name in var_names:
            header_parts.append("$\\mu_{{{}}}$".format(name))
        for name in var_names:
            header_parts.append("$\\sigma_{{{}}}$".format(name))
        for i in range(self.nvars):
            for j in range(i + 1, self.nvars):
                header_parts.append(
                    "$\\rho_{{{}{}}}$".format(var_names[i], var_names[j])
                )
        header = " & ".join(header_parts)
        lines = [
            "\\begin{table*}\n\\begin{tabular}{" + colspec + "}",
            "\\hline",
            header + " \\\\",
            "\\hline",
        ]
        for idx, row in zip(np.array(order) + 1, rows):
            fmt_row = [str(idx)] + [
                "{:.{d}g}".format(x, d=decimals) if np.isfinite(x) else "---"
                for x in row
            ]
            lines.append(" & ".join(fmt_row) + " \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}\n\\end{table*}")
        return "\n".join(lines)


class FitCMND(MultiMinBase):
    r"""
    CMND Fitting handler.


    Theory
    ------
    The fitting procedure is based on the Maximum Likelihood Estimation (MLE) method.
    Given a dataset of :math:`S` points :math:`\tilde U_k` (e.g., unbound orbital elements), the likelihood
    :math:`\mathcal{L}` of the CMND parameters is given by:

    .. math::

        \mathcal{L}(\{w_k\}_M, \{\mu_k\}_M, \{\Sigma_k\}_M | \{\tilde U_k\}_S) = \prod_{i=1}^{S} \mathcal{C}_M(\tilde U_i)

    To find the optimal parameters, we minimize the negative normalized log-likelihood:

    .. math::

        -\frac{\log \mathcal{L}}{S} = -\frac{1}{S} \sum_{i=1}^{S} \log \mathcal{C}_M(\tilde U_i)

    This quantity represents the average negative log-probability density of the data under the model.
    Minimizing this value is equivalent to maximizing the likelihood that the observed data was
    generated by the CMND model.

    Attributes
    ----------
    ngauss : int
        Number of fitting MND.
    nvars : int
        Number of variables in each MND.
    cmnd : ComposedMultiVariateNormal
        Fitting object of the class CMND. This object will have the result of the fitting procedure.
    solution : scipy.optimize.OptimizeResult
        Once the fitting is completed the solution object is returned. Attributes include:
            - fun: value of the function in the minimum.
            - x: value of the minimization parameters in the minimum.
            - nit: number of iterations of the minimization algorithm.
            - nfev: number of evaluations of the function.
            - success: if True it implies that the minimization fullfills all the conditions.
    Ndim : int
        Number of mus.
    Ncorr : int
        Number of correlations.
    minparams : numpy.ndarray
        Array of minimization parameters at any stage in minimization process.
    scales : numpy.ndarray
        Array of scales to convert minparams to uparams (unbound) and viceversa.
    uparams : numpy.ndarray
        Array of unbound minimization parameters.

    Notes
    -----
    Objects of this class must be always initialized with the number of gaussians and the number
    of random variables.

    Examples
    --------
    Example: Fitting a synthetic dataset.

    >>> # 1. Generate synthetic data
    >>> np.random.seed(1)
    >>> weights = [[1.0]]
    >>> mus = [[1.0, 0.5, -0.5]]
    >>> sigmas = [[1, 1.2, 2.3]]
    >>> angles = [[10*Angle.Deg, 30*Angle.Deg, 20*Angle.Deg]]
    >>> Sigmas = Stats.calc_covariance_from_rotation(sigmas, angles)
    >>> CMND = ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
    >>> data = CMND.rvs(10000)

    >>> # 2. Initialize the fitter
    >>> F = FitCMND(ngauss=CMND.ngauss, nvars=CMND.nvars)

    >>> # 3. Set bounds (optional but recommended)
    >>> bounds = F.set_bounds(bounds=(0.1, 0.9*F._sigmax))

    >>> # 4. Run the fit
    >>> F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True), method=None, bounds=bounds)

    >>> # 5. Inspect results
    >>> print(F.cmnd)
    >>> G = F.plot_fit(figsize=3, hargs=dict(bins=30, cmap='YlGn'), sargs=dict(s=0.5, edgecolor='None', color='r'))

    >>> # 6. Save/Load results
    >>> F.save_fit("/tmp/fit.pkl", useprefix=False)
    >>> F._load_fit("/tmp/fit.pkl")

    """

    # Constants
    # Maximum value of sigma
    _sigmax = 10
    _ignoreWarnings = True

    def update_params(self, weights=None, mus=None, sigmas=None, rhos=None):
        """
        Update the parameters of the CMND.

        Parameters
        ----------
        weights : array_like, optional
            Weights of the gaussians.
        mus : array_like, optional
            Means of the gaussians.
        sigmas : array_like, optional
            Standard deviations of the gaussians.
        rhos : array_like, optional
            Correlation coefficients of the gaussians.
        """
        if weights is not None:
            self.weights = np.array(weights)
            if self.normalize_weights:
                self.weights /= np.sum(self.weights)

        if mus is not None:
            mus = np.array(mus)
            if mus.shape != self.mus.shape:
                raise ValueError(
                    f"Shape of mus {mus.shape} does not match current shape {self.mus.shape}"
                )
            self.mus = mus

        update_cov = False
        if sigmas is not None:
            sigmas = np.array(sigmas)
            if sigmas.shape != self.sigmas.shape:
                raise ValueError(
                    f"Shape of sigmas {sigmas.shape} does not match current shape {self.sigmas.shape}"
                )
            self.sigmas = sigmas
            update_cov = True

        if rhos is not None:
            rhos = np.array(rhos)
            if rhos.shape != self.rhos.shape:
                raise ValueError(
                    f"Shape of rhos {rhos.shape} does not match current shape {self.rhos.shape}"
                )
            self.rhos = rhos
            update_cov = True

        if update_cov:
            self.Sigmas = Stats.calc_covariance_from_correlations(
                self.sigmas, self.rhos
            )

        self.params = self.flatten_params()

    def __init__(self, data=None, ngauss=1, nvars=None, domain=None, objfile=None):
        """
        Initialize FitCMND object.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Array with data (Nsam x nvars). If provided, nvars is inferred unless explicitly given.
        ngauss : int
            Number of Gaussian components.
        nvars : int, optional
            Number of variables. If None and data is provided, inferred from data.
        domain : list, optional
            Domain for each variable: None (unbounded) or [low, high] per variable.
            Example: [None, [0, 1], None] for variable 1 bounded to [0, 1].
        objfile : str, optional
            Path to a pickled fit to load.
        """

        if objfile is not None:
            self._load_fit(objfile)
        elif isinstance(data, str):
            self._load_fit(data)
        else:
            # Check data and infer nvars
            self.data = None
            if data is not None:
                self.data = np.asarray(data)
                if self.data.ndim != 2:
                    raise ValueError(
                        f"Data must be 2D array (Nsam x nvars), got shape {self.data.shape}"
                    )

                data_nvars = self.data.shape[1]
                if nvars is None:
                    nvars = data_nvars
                elif nvars != data_nvars:
                    raise ValueError(
                        f"Provided nvars={nvars} does not match data dimension {data_nvars}"
                    )

            if nvars is None:
                # If no data and no nvars, default to 2 (backward compatibility or default)
                nvars = 2

            # Basic attributes
            self.ngauss = ngauss
            self.nvars = nvars
            self.Ndim = ngauss * nvars
            self.Ncorr = int(nvars * (nvars - 1) / 2)
            self.domain = domain

            # Define the model cmnds
            self.cmnd = ComposedMultiVariateNormal(
                ngauss=ngauss, nvars=nvars, domain=domain
            )

            # Set parameters
            self.set_params()

        # Report
        print("Loading a FitCMND object.")
        print(f"Number of gaussians: {self.ngauss}")
        print(f"Number of variables: {self.nvars}")
        print(f"Number of dimensions: {self.Ndim}")
        if self.data is not None:
            print(f"Number of samples: {len(self.data)}")
        else:
            print("Number of samples: 0 (No data provided)")
        if self.domain is not None:
            print(f"Domain: {self.domain}")

        # Compute the log-likelihood of the data
        if self.data is not None:
            print(f"Log-likelihood per point (-log L/N): {self.norm_log_l()}")

        # Other
        self.fig = None
        self.prefix = ""

    def set_params(self, mu=0.5, sigma=1.0, rho=0.5):
        """
        Set the value of the basic params (minparams, scales, etc.). It updates minparams, scales
        and uprams.
        Ex. F.set_params(mu=0.5, sigma=1.0, rho=0.5)

        Parameters
        ----------
        mu : float, optional
            Value of all initial mus (default 0.5).
        sigma : float, optional
            Value of all initial sigmas (default 1.0).
        rho : float, optional
            Value of all initial rhos (default 0.5).

        Returns
        -------
        None
        """
        # Define the initial parameters
        #         mus             sigmas          correlations
        minparams = (
            [mu] * self.Ndim
            + [sigma] * self.Ndim
            + [1 + rho] * self.ngauss * self.Ncorr
        )
        scales = (
            [0] * self.Ndim
            + [self._sigmax] * self.Ndim
            + [2] * self.ngauss * self.Ncorr
        )
        if self.ngauss > 1:
            self.extrap = []
            minparams = [1 / self.ngauss] * self.ngauss + minparams
            scales = [1] * self.ngauss + scales
        else:
            self.extrap = [1]

        self.minparams = np.array(minparams)
        # When domain is finite, spread initial mus inside each variable's domain
        bounds = getattr(self.cmnd, "_domain_bounds", None)
        if bounds is not None and self.ngauss > 1:
            for j in range(self.nvars):
                lo, hi = bounds[j][0], bounds[j][1]
                if np.isfinite(lo) or np.isfinite(hi):
                    a = lo if np.isfinite(lo) else hi - 1.0
                    b = hi if np.isfinite(hi) else lo + 1.0
                    for i in range(self.ngauss):
                        idx = self.ngauss + i * self.nvars + j
                        self.minparams[idx] = a + (i + 1) / (self.ngauss + 1) * (b - a)
        self.scales = np.array(scales)
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        self._initial_params_set_by_user = False

    def set_initial_params(self, mus=None, sigmas=None, rhos=None):
        """
        Set initial values for the minimization parameters (means, standard deviations, correlations).
        Only the arguments provided are updated; the rest keep their current values.
        Ex. F.set_initial_params(mus=[[0.2, 0.3], [0.8, 0.7]], sigmas=[[0.1, 0.2], [0.1, 0.2]])

        Parameters
        ----------
        mus : array-like, optional
            Initial means. Shape (ngauss, nvars) or (nvars,) to use the same means for all components.
            Example: [0.2, 0.3] or [[0.2, 0.3], [0.8, 0.7]].
        sigmas : array-like, optional
            Initial standard deviations. Shape (ngauss, nvars) or (nvars,) to use the same for all components.
        rhos : array-like, optional
            Initial correlation coefficients (upper triangle). Shape (ngauss, Ncorr) or (Ncorr,) to use the same for all.
            Ncorr = nvars*(nvars-1)/2.

        Returns
        -------
        None

        Examples
        --------
        >>> F = FitCMND(ngauss=2, nvars=2)
        >>> F.set_initial_params(mus=[0.2, 0.3], sigmas=[0.1, 0.1])  # same for both components
        >>> F.set_initial_params(mus=[[0.2, 0.3], [0.8, 0.7]], sigmas=[[0.1, 0.2], [0.1, 0.2]])
        >>> F.fit_data(data)
        """
        mus = np.asarray(mus, dtype=float) if mus is not None else None
        sigmas = np.asarray(sigmas, dtype=float) if sigmas is not None else None
        rhos = np.asarray(rhos, dtype=float) if rhos is not None else None

        def _broadcast_2d(arr, shape_2d, name, dims_desc):
            """Broadcast to (ngauss, nvars) or (ngauss, Ncorr). If 1D, same row for all components."""
            if arr is None:
                return None
            arr = np.atleast_1d(arr)
            if arr.ndim == 1:
                # Same values for all components
                if arr.shape[0] != shape_2d[1]:
                    raise ValueError(
                        "{} 1D must have length {} ({}), got {}".format(
                            name, shape_2d[1], dims_desc, arr.shape[0]
                        )
                    )
                arr = np.tile(arr, (self.ngauss, 1))
            else:
                arr = np.atleast_2d(arr)
                if arr.shape != shape_2d:
                    raise ValueError(
                        "{} must have shape {} or ({}), got {}".format(
                            name, shape_2d, dims_desc, arr.shape
                        )
                    )
            return arr

        # minparams layout: [weights] + [mus] + [sigmas] + [1+rhos]
        if self.ngauss > 1:
            off_mu = self.ngauss
            off_sig = self.ngauss + self.Ndim
            off_rho = self.ngauss + 2 * self.Ndim
        else:
            off_mu = 0
            off_sig = self.nvars
            off_rho = self.nvars * 2

        if mus is not None:
            mus = _broadcast_2d(mus, (self.ngauss, self.nvars), "mus", "nvars")
            if self.ngauss > 1:
                self.minparams[off_mu : off_mu + self.Ndim] = mus.ravel()
            else:
                self.minparams[off_mu : off_mu + self.nvars] = mus.ravel()

        if sigmas is not None:
            sigmas = _broadcast_2d(sigmas, (self.ngauss, self.nvars), "sigmas", "nvars")
            if self.ngauss > 1:
                self.minparams[off_sig : off_sig + self.Ndim] = sigmas.ravel()
            else:
                self.minparams[off_sig : off_sig + self.nvars] = sigmas.ravel()

        if rhos is not None:
            rhos = _broadcast_2d(rhos, (self.ngauss, self.Ncorr), "rhos", "Ncorr")
            # Internally we store 1+rho so the unbound param is in a good range
            if self.ngauss > 1:
                self.minparams[off_rho : off_rho + self.ngauss * self.Ncorr] = (
                    1.0 + rhos.ravel()
                )
            else:
                self.minparams[off_rho : off_rho + self.Ncorr] = 1.0 + rhos.ravel()

        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        self._initial_params_set_by_user = True

        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.nvars)

    def _stdcorr_to_minparams(self, stdcorr):
        """Inverse of pmap: stdcorr -> minparams (for use in normalized fit)."""
        minparams = np.array(stdcorr[len(self.extrap) :], dtype=float)
        if self.ngauss * self.Ncorr > 0:
            minparams[-self.ngauss * self.Ncorr :] += 1
        return minparams

    def _stdcorr_from_weights_mus_sigmas_rhos(self, weights, mus, sigmas, rhos):
        """Build stdcorr from unpacked arrays (rhos as correlations, not 1+rho).
        Returns stdcorr without extrap (for use with cmnd.set_stdcorr).
        To get FitCMND's stdcorr format (with extrap), prepend self.extrap."""
        return np.concatenate(
            [
                np.asarray(weights).ravel(),
                np.asarray(mus).ravel(),
                np.asarray(sigmas).ravel(),
                np.asarray(rhos).ravel(),
            ]
        )

    def _init_params_from_data_finite_domain(self, data):
        """When domain is finite, set initial mus from data percentiles and modest sigmas so the optimizer starts near the peaks."""
        bounds = getattr(self.cmnd, "_domain_bounds", None)
        if bounds is None or self.ngauss < 2:
            return
        data = np.asarray(data)
        if data.size == 0:
            return
        # Flatten so we can index by variable
        if data.ndim == 1:
            data = np.reshape(data, (-1, 1))
        n = data.shape[0]
        for j in range(self.nvars):
            lo, hi = bounds[j][0], bounds[j][1]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            col = data[:, j]
            # Initial mus = percentiles of data along this dimension, sorted and clipped
            pct = np.linspace(5, 95, self.ngauss)
            mus_init = np.percentile(col, pct)
            mus_init = np.sort(mus_init)
            mus_init = np.clip(mus_init, lo, hi)
            for i in range(self.ngauss):
                idx = self.ngauss + i * self.nvars + j
                self.minparams[idx] = mus_init[i]
            # Initial sigmas: start with a fraction of domain width so we don't drift to flat
            width = hi - lo
            sigma_init = min(0.25 * width, max(0.05 * width, np.std(col) * 0.5))
            for i in range(self.ngauss):
                idx = self.ngauss + self.Ndim + i * self.nvars + j
                if idx < len(self.minparams):
                    self.minparams[idx] = sigma_init
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)

    def pmap(self, minparams):
        """
        Mapping routine used in sample_cmnd_likelihood. Mapping may change depending on the
        complexity of the parameters to be minimized.
        Ex. stdcorr = F.pmap(minparams)

        Parameters
        ----------
        minparams : numpy.ndarray
            Minimization parameters.

        Returns
        -------
        stdcorr : numpy.ndarray
            Flatten parameters with correlations.
        """
        stdcorr = np.array(self.extrap + list(minparams))
        if self.ngauss * self.Ncorr > 0:
            stdcorr[-self.ngauss * self.Ncorr :] -= 1
        return stdcorr

    def log_l(self, data=None):
        """
        Value of the -log(Likelihood).
        Ex. F.log_l(data)

        Parameters
        ----------
        data : numpy.ndarray, optional
            Array with data (Nsam x nvars). If None, uses self.data.

        Returns
        -------
        log_l : float
            Value of the -log(Likelihood).
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        log_l = self.cmnd.sample_cmnd_likelihood(
            self.uparams, data=data, pmap=self.pmap, tset="stdcorr", scales=self.scales
        )
        return log_l

    def norm_log_l(self, data=None):
        """
        Calculate the normalized log-likelihood of the data.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data for which to calculate the log-likelihood. If None, uses self.data.

        Returns
        -------
        float
            Normalized log-likelihood per point (-log L/N).
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        data = np.asarray(data)
        log_likelihood = self.log_l(data) / len(data)
        return log_likelihood

    def fit_data(
        self, data=None, verbose=0, advance=0, normalize=False, progress=False, **args
    ):
        """
        Minimization procedure. It updates the solution attribute.
        Ex. F.fit_data(data, verbose=0, tol=1e-3)

        Parameters
        ----------
        data : numpy.ndarray, optional
            Array with data (Nsam x nvars). If None, uses self.data.
        verbose : int, optional
            Verbosity level for the sample_cmnd_likelihood routine (default 0).
        advance : int, optional
            If larger than 0 show advance each "advance" iterations (default 0).
        normalize : bool, optional
            If True and domain is finite, fit in normalized space (each variable scaled to [0, 1]
            using the domain bounds). Improves conditioning when variables have very different
            scales (e.g. [0, 1.3], [0, 1], [0, 180]) and can yield more stable, reproducible minima.
        progress : str or bool, optional
            - "tqdm" or True: show a tqdm progress bar (requires tqdm).
            - "text": show text-based progress (equivalent to setting advance > 0).
            - False: no progress output (unless advance is set).
        **args : dict
            Options of the minimize routine (eg. tol=1e-6).
            A particularly interesting parameter is the minimization method.
            Available methods:

            * Slow but sure: Powell
            * Fast but unsure: CG, BFGS, COBYLA, SLSQP

        Returns
        -------
        None

        Examples
        --------
        >>> F = FitCMND(1, 3)
        >>> F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True))
        >>> F.fit_data(data, normalize=True)  # recommended when variable scales differ a lot
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        # Handle progress options
        use_tqdm = False
        if progress == "tqdm" or (isinstance(progress, bool) and progress):
            use_tqdm = True
        elif progress == "text":
            if not advance:
                advance = 1

        pbar = None
        if use_tqdm:
            try:
                from tqdm.auto import tqdm

                maxiter = args.get("maxiter") or args.get("options", {}).get("maxiter")
                pbar = tqdm(total=maxiter, desc="Minimizing")
            except ImportError:
                print("tqdm not installed, progress bar disabled.")

        if advance or pbar is not None:
            advance = int(advance) if advance else 0
            self.neval = 0

            def _advance(X, show=False):
                if pbar is not None:
                    pbar.update(1)
                if advance > 0:
                    if self.neval == 0:
                        print(f"Iterations:")
                    if self.neval % advance == 0 or show:
                        vars = np.array2string(
                            X,
                            separator=", ",
                            precision=4,
                            max_line_width=np.inf,
                            formatter={"float_kind": lambda x: f"{x:.2g}"},
                        )
                        fun = self.cmnd.sample_cmnd_likelihood(
                            X, data, self.pmap, "stdcorr", self.scales, verbose
                        )
                        print(
                            f"Iter {self.neval}:\n\tVars: {vars}\n\tLogL/N: {fun / len(data)}"
                        )
                self.neval += 1
        else:
            _advance = None

        self.data = np.copy(data)
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.cmnd._ignoreWarnings = self._ignoreWarnings
        self.minargs = dict(method="Powell")
        self.minargs.update(args)

        # When domain is finite, set bounds so mus stay inside the domain and sigmas stay > 0
        bounds_orig = getattr(self.cmnd, "_domain_bounds", None)
        has_finite_domain = bounds_orig is not None and any(
            np.isfinite(bounds_orig[j][0]) or np.isfinite(bounds_orig[j][1])
            for j in range(self.nvars)
        )

        # Optional: fit in normalized space [0,1] per variable for better conditioning
        if normalize and has_finite_domain:
            scale = np.zeros(self.nvars)
            offset = np.zeros(self.nvars)
            for j in range(self.nvars):
                lo, hi = bounds_orig[j][0], bounds_orig[j][1]
                if np.isfinite(lo) and np.isfinite(hi):
                    offset[j], scale[j] = lo, hi - lo
                else:
                    offset[j] = np.nanmin(data[:, j])
                    scale[j] = max(np.nanmax(data[:, j]) - offset[j], 1e-10)
            data_fit = (data - offset) / scale
            domain_norm = tuple([0.0, 1.0] for _ in range(self.nvars))
            self.cmnd._domain_bounds = [tuple(x) for x in domain_norm]
            # Transform current minparams to normalized space
            stdcorr = self.pmap(self.minparams)
            i = len(self.extrap)
            # For ngauss==1 there are no weights in stdcorr after extrap; next are mus
            w = np.array([1.0]) if self.ngauss == 1 else stdcorr[i : i + self.ngauss]
            mus = stdcorr[
                i + (self.ngauss if self.ngauss > 1 else 0) : i
                + (self.ngauss if self.ngauss > 1 else 0)
                + self.Ndim
            ].reshape(self.ngauss, self.nvars)
            sigmas = stdcorr[
                i + (self.ngauss if self.ngauss > 1 else 0) + self.Ndim : i
                + (self.ngauss if self.ngauss > 1 else 0)
                + 2 * self.Ndim
            ].reshape(self.ngauss, self.nvars)
            rhos = stdcorr[
                i + (self.ngauss if self.ngauss > 1 else 0) + 2 * self.Ndim : i
                + (self.ngauss if self.ngauss > 1 else 0)
                + 2 * self.Ndim
                + self.ngauss * self.Ncorr
            ].reshape(self.ngauss, self.Ncorr)
            mus_norm = (mus - offset) / scale
            sigmas_norm = sigmas / scale
            stdcorr_norm_no_extrap = self._stdcorr_from_weights_mus_sigmas_rhos(
                w, mus_norm, sigmas_norm, rhos
            )
            # For ngauss==1, minparams has no weight; for ngauss>1, use _stdcorr_to_minparams
            if self.ngauss == 1:
                # stdcorr_norm_no_extrap = [w, mus, sigmas, rhos] = [1, 3, 3, 3] = 10 elements
                # minparams = [mus, sigmas, 1+rhos] = [3, 3, 3] = 9 elements
                self.minparams = np.concatenate(
                    [
                        stdcorr_norm_no_extrap[1 : 1 + self.Ndim],  # mus
                        stdcorr_norm_no_extrap[
                            1 + self.Ndim : 1 + 2 * self.Ndim
                        ],  # sigmas
                        stdcorr_norm_no_extrap[1 + 2 * self.Ndim :] + 1.0,  # 1+rhos
                    ]
                )
            else:
                stdcorr_norm = np.concatenate([self.extrap, stdcorr_norm_no_extrap])
                self.minparams = self._stdcorr_to_minparams(stdcorr_norm)
            self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
            sigma_hi_norm = min(0.99 * self._sigmax, 2.0)
            bounds_tuple = self.set_bounds(bounds=(1e-6, sigma_hi_norm))
            self.minargs["bounds"] = bounds_tuple
            if self.minargs.get("method") == "Powell":
                self.minargs["method"] = "L-BFGS-B"
            try:
                self.solution = minimize(
                    self.cmnd.sample_cmnd_likelihood,
                    self.uparams,
                    callback=_advance,
                    args=(data_fit, self.pmap, "stdcorr", self.scales, verbose),
                    **self.minargs,
                )
            finally:
                if pbar is not None:
                    pbar.close()
            if advance:
                _advance(self.solution.x, show=True)
            self.minparams = Util.t_if(self.solution.x, self.scales, Util.u2f)
            stdcorr_norm = self.pmap(self.minparams)
            i = len(self.extrap)
            off = i + (self.ngauss if self.ngauss > 1 else 0)
            w = (
                np.array([1.0])
                if self.ngauss == 1
                else stdcorr_norm[i : i + self.ngauss]
            )
            mus_norm = stdcorr_norm[off : off + self.Ndim].reshape(
                self.ngauss, self.nvars
            )
            sigmas_norm = stdcorr_norm[off + self.Ndim : off + 2 * self.Ndim].reshape(
                self.ngauss, self.nvars
            )
            rhos = stdcorr_norm[
                off + 2 * self.Ndim : off + 2 * self.Ndim + self.ngauss * self.Ncorr
            ].reshape(self.ngauss, self.Ncorr)
            mus_orig = mus_norm * scale + offset
            sigmas_orig = sigmas_norm * scale
            stdcorr_orig = self._stdcorr_from_weights_mus_sigmas_rhos(
                w, mus_orig, sigmas_orig, rhos
            )
            self.cmnd._domain_bounds = bounds_orig
            self.cmnd.set_stdcorr(stdcorr_orig, self.nvars)
            # Convert cmnd's stdcorr to FitCMND's minparams (for ngauss=1, minparams has no weight)
            if self.ngauss == 1:
                mus_flat = mus_orig.ravel()
                sigmas_flat = sigmas_orig.ravel()
                rhos_flat = (rhos + 1.0).ravel()
                self.minparams = np.concatenate([mus_flat, sigmas_flat, rhos_flat])
            else:
                stdcorr_orig_with_extrap = np.concatenate([self.extrap, stdcorr_orig])
                self.minparams = self._stdcorr_to_minparams(stdcorr_orig_with_extrap)
            self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
            self._update_prefix()
            return

        if has_finite_domain and "bounds" not in args:
            # bounds from domain; sigma: lower 1e-6 to avoid div by zero; upper limited so fit cannot go flat
            sigma_hi = 0.99 * self._sigmax
            domain_widths = [
                self.cmnd._domain_bounds[j][1] - self.cmnd._domain_bounds[j][0]
                for j in range(self.nvars)
                if np.isfinite(self.cmnd._domain_bounds[j][0])
                and np.isfinite(self.cmnd._domain_bounds[j][1])
            ]
            if domain_widths:
                # Cap sigma at ~2x largest bounded dimension width to avoid flat (uniform) optimum
                sigma_hi = min(sigma_hi, 2.0 * max(domain_widths))
            bounds_tuple = self.set_bounds(bounds=(1e-6, sigma_hi))
            self.minargs["bounds"] = bounds_tuple
            if self.minargs.get("method") == "Powell":
                self.minargs["method"] = "L-BFGS-B"
            # Data-based init: set initial mus (and sigmas) from data unless user set them via set_initial_params
            if not getattr(self, "_initial_params_set_by_user", False):
                self._init_params_from_data_finite_domain(data)

        try:
            self.solution = minimize(
                self.cmnd.sample_cmnd_likelihood,
                self.uparams,
                callback=_advance,
                args=(data, self.pmap, "stdcorr", self.scales, verbose),
                **self.minargs,
            )
        finally:
            if pbar is not None:
                pbar.close()
        if advance:
            _advance(self.solution.x, show=True)
        self.uparams = self.solution.x

        # Set the new params
        # Set the new params
        self.minparams = Util.t_if(self.uparams, self.scales, Util.u2f)
        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.nvars)
        self._update_prefix()

    def _load_fit(self, objfile):
        """
        Load a fit object from file.


        """
        F = pickle.load(open(objfile, "rb"))
        for k in F.__dict__.keys():
            setattr(self, k, getattr(F, k))
        self._update_prefix()

    def _gen_qsample(self, n_sim, resample=False):
        """
        Generate or reuse the synthetic sample for Q-Q plots and quality checks.
        """
        if resample or not hasattr(self, "_qsample") or self._qsample is None:
            self._qsample = self.cmnd.rvs(n_sim)
        # Check size consistency if we are reusing
        elif len(self._qsample) != n_sim:
            # If the requested size is different, we must regenerate
            self._qsample = self.cmnd.rvs(n_sim)
        return self._qsample

    def quality_of_fit(
        self,
        data=None,
        n_sim=None,
        resample=False,
        plot_qq=False,
        figsize=5,
        ax=None,
        plot_line=True,
        title="Q-Q Plot",
        verbose=False,
    ):
        r"""
        Calculate goodness-of-fit statistics using the Kolmogorov-Smirnov distance and the Coefficient of Determination (:math:`R^2`) relative to the identity line.

        Optionally generates a Q-Q plot to visually assess the quality of the fit.

        The **Kolmogorov-Smirnov Statistic** (:math:`D_{KS}`) measures the maximum absolute distance between the Cumulative Distribution Function (CDF) of the observed data and that of the model. It represents the "worst point" of local deviation.

        .. math::

            D_{KS} = \sup_{x} | F_{obs}(x) - F_{model}(x) |

        The range of the statistic is :math:`D_{KS} \in [0, 1]`, where an acceptable value is :math:`D_{KS} \approx 0` (ideally < 0.05).

        The **Q-Q plot** is a non-parametric visual diagnostic tool used to assess whether a set of observed data follows a specific theoretical distribution. It compares the distribution of the **Negative Log-Likelihood** (:math:`S = -2 \ln \mathcal{L}`) of the real data against the model's prediction.

        If the fit is ideal, the points should align on the identity line :math:`y=x`. Deviations indicate:

        *   **Curved ends**: tail deviations ("Heavy" or "Light" tails).
        *   **Shift or slope change**: systematic errors in mean or variance.

        The **Coefficient of Determination on Identity** (:math:`R^2`) measures the variance explained by the model with respect to the ideal line :math:`y=x`. It penalizes both dispersion and bias.

        .. math::

            R^2 = 1 - \frac{\sum (y_{obs} - x_{model})^2}{\sum (y_{obs} - \bar{y}_{obs})^2}

        The range is :math:`R^2 \in (-\infty, 1]`, where an acceptable value is :math:`R^2 \approx 1` (ideally > 0.9).


        Parameters
        ----------
        data : numpy.ndarray, optional
            Observed data. If None, uses self.data.
        n_sim : int, optional
            Number of synthetic data points to generate. Default is 10 * len(data).
        resample : bool, optional
            If True, regenerate the synthetic sample even if one exists.
        plot_qq : bool, optional
            If True, generate the Q-Q plot. Default is False.
        figsize : int or tuple, optional
            Size of the figure (if creating a new one).
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None and plot_qq=True, creates a new figure.
        plot_line : bool, optional
            If True, plot the identity line y=x.
        title : str, optional
            Title of the plot.
        verbose : bool, optional
            If True, print the results.

        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'ks_dist': Kolmogorov-Smirnov distance between S_obs and S_sim (lower is better)
            - 'ks_pval': p-value from KS test (note: not strictly valid as parameters are estimated)
            - 'r2_identity': R-squared coefficient regarding identity line (closer to 1 is better)
            - 'ax': The axes object if plot_qq=True, else None.

        """
        from scipy.stats import ks_2samp

        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        data = np.asarray(data)
        n_samples = len(data)

        if n_sim is None:
            n_sim = 10 * n_samples

        if verbose:
            print(
                f"Calculating fit quality statistics for {n_samples} observed points..."
            )

        # 1. Calculate S_obs = -2 * ln(L(x_obs))
        pdf_vals_obs = self.cmnd.pdf(data)
        pdf_vals_obs = np.maximum(pdf_vals_obs, 1e-300)
        S_obs = -2 * np.log(pdf_vals_obs)
        S_obs.sort()

        # 2. Generate synthetic data and calculate S_sim
        if verbose:
            print(f"Generating {n_sim} synthetic points (resample={resample})...")
        data_sim = self._gen_qsample(n_sim, resample=resample)
        pdf_vals_sim = self.cmnd.pdf(data_sim)
        pdf_vals_sim = np.maximum(pdf_vals_sim, 1e-300)
        S_sim = -2 * np.log(pdf_vals_sim)
        S_sim.sort()

        # 3. Kolmogorov-Smirnov Statistic
        d_ks, p_val = ks_2samp(S_obs, S_sim)

        # 4. R2 relative to identity (y=x)
        # Interpolate S_sim to match quantiles of S_obs for pairwise comparison
        percentiles = np.linspace(0, 100, n_samples)
        S_sim_interp = np.percentile(S_sim, percentiles)

        # Residuals from identity line (y=x means S_obs = S_sim_interp)
        residuos = S_obs - S_sim_interp
        ss_res = np.sum(residuos**2)
        # Total sum of squares of S_obs around its mean
        ss_tot = np.sum((S_obs - np.mean(S_obs)) ** 2)

        # Avoid division by zero if ss_tot is 0
        if ss_tot == 0:
            r2_identity = 0.0  # or nan
        else:
            r2_identity = 1 - (ss_res / ss_tot)

        if verbose:
            print(f"Fit Quality Results:")
            print(f"  KS Distance (lower is better)     : {d_ks:.4f}")
            print(f"  R2 vs Identity (closer to 1 is better): {r2_identity:.4f}")

        # 5. Plot
        ret_ax = None
        if plot_qq:
            from matplotlib import pyplot as plt

            if ax is None:
                if isinstance(figsize, int):
                    figsize = (figsize, figsize)
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            self.fig_qq = fig
            self.ax_qq = ax

            label = f"Data vs Model ($R^2$={r2_identity:.4f})"
            self.ax_qq.scatter(S_sim_interp, S_obs, alpha=0.5, label=label)

            if plot_line:
                # Identity line range
                min_val = min(S_obs.min(), S_sim_interp.min())
                max_val = max(S_obs.max(), S_sim_interp.max())
                self.ax_qq.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    label="Perfect Fit (y=x)",
                )

            self.ax_qq.set_xlabel("Theoretical Quantiles (-2 Log Likelihood)")
            self.ax_qq.set_ylabel("Observed Quantiles (-2 Log Likelihood)")
            self.ax_qq.set_title(title)
            self.ax_qq.legend(loc="upper left")
            self.ax_qq.grid(True, alpha=0.3)

            # Add KS statistic to the plot
            self.ax_qq.text(
                0.95,
                0.05,
                rf"$D_{{KS}} = {d_ks:.3f}$",
                transform=self.ax_qq.transAxes,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            multimin_watermark(self.ax_qq)

        return {"ks_dist": d_ks, "r2_identity": r2_identity}

    def plot_fit(
        self,
        N=10000,
        figsize=2,
        properties=None,
        ranges=None,
        hargs=None,
        sargs=None,
    ):
        """
        Plot the result of the fitting procedure.
        Ex. F.plot_fit(figsize=3, hargs=dict(bins=30), sargs=dict(s=0.5, color='r'))

        Parameters
        ----------
        N : int, optional
            number of points used to build a representation of the marginal distributions (default 10000).
        figsize : int, optional
            Size of each axis (default 2).
        properties : list or dict, optional
            Property names or DensityPlot-style properties. List: each element as axis label,
            range=None. Dict: same as DensityPlot (keys with 'label' and optional 'range').
        ranges : list, optional
            Ranges per variable; used only when properties is a list or None.
        hargs : dict, optional
            Dictionary with options for the hist2d function (or 1D hist when nvars=1).
            For univariate fits, if provided the sample is shown as a histogram; otherwise
            the sample is shown as a scatter plot.
        sargs : dict, optional
            Dictionary with options for the scatter plot. Default: dict(s=0.5, edgecolor=None, color='b').
            For univariate fits, used when hargs is not provided (fit is always shown as PDF).

        Returns
        -------
        G : matplotlib.figure.Figure or DensityPlot
            Graphic handle.

        Examples
        --------
        >>> F = FitCMND(1, 3)
        >>> F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True))
        >>> G = F.plot_fit(figsize=3, hargs=dict(bins=30, cmap='YlGn'), sargs=dict(s=0.5, edgecolor=None, color='r'))
        """
        if hargs is None:
            hargs = dict()
        if sargs is None:
            sargs = dict(s=0.5, edgecolor=None, color="b")
        properties = _props_to_properties(properties, self.nvars, ranges)

        from matplotlib import pyplot as plt

        if self.nvars == 1:
            # Univariate: show fitted PDF; show sample as histogram (if hargs) and/or scatter (if sargs or neither)
            G = DensityPlot(properties, figsize=figsize)
            ax = G.axs[0][0]
            if hargs:
                G.plot_hist(self.data, **hargs)
            if sargs or (not hargs and not sargs):
                G.scatter_plot(self.data, **sargs)
            # Overlay fitted PDF (same axis when we have hist, else twin when only scatter)
            x_min, x_max = self.data[:, 0].min(), self.data[:, 0].max()
            margin = max(1e-6, 0.1 * (x_max - x_min))
            x_curve = np.linspace(x_min - margin, x_max + margin, 300)
            pdf_vals = self.cmnd.pdf(x_curve.reshape(-1, 1))
            if hargs:
                ax.plot(x_curve, pdf_vals, "k-", lw=2, label="PDF")
            else:
                ax.plot(x_curve, pdf_vals, "k-", lw=2, label="PDF")
                ax.set_ylabel("PDF")
                ax.set_ylim(0, None)
            # Legend: combine primary ax (histogram, PDF) and twin (sample scatter) if present
            handles, labels = ax.get_legend_handles_labels()
            if getattr(G, "_ax_twin", None) is not None:
                h2, l2 = G._ax_twin.get_legend_handles_labels()
                handles, labels = handles + h2, labels + l2
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(handles),
                frameon=False,
            )
            G.fig.subplots_adjust(top=0.88)  # room for legend above
            if not getattr(G, "_watermark_added", False):
                multimin_watermark(
                    G.axs[0][0], frac=0.5
                )  # univariate: larger watermark
            self.fig = G.fig
            return G
        if self.nvars >= 2:
            Xfits = self.cmnd.rvs(N)
            G = DensityPlot(properties, figsize=figsize)
            G.plot_hist(Xfits, **hargs)
            G.scatter_plot(self.data, **sargs)
            G.fig.tight_layout()
            multimin_watermark(G.axs[0][0])
            self.fig = G.fig
            return G
        else:
            # nvars == 2: 2D hist2d + scatter
            Xfits = self.cmnd.rvs(N)
            keys = list(properties.keys())
            fig = plt.figure(figsize=(5, 5))
            ax = fig.gca()
            ax.hist2d(Xfits[:, 0], Xfits[:, 1], **hargs)
            ax.scatter(self.data[:, 0], self.data[:, 1], **sargs)
            ax.grid()
            ax.set_xlabel(properties[keys[0]]["label"])
            ax.set_ylabel(properties[keys[1]]["label"])
            multimin_watermark(ax)
            fig.tight_layout()
            self.fig = fig
            return fig

    def _inv_params(self, stdcorr):
        """
        Invert stdcorr to minparams.


        """
        minparams = np.copy(stdcorr)
        minparams[-self.ngauss * self.Ncorr :] += 1
        self.minparams = minparams[1:] if self.ngauss == 1 else minparams

    def _update_prefix(self, myprefix=None):
        """
        Update prefix of fit.

        Prefix has two parts: the number of gaussians used and a hash computed from the object.

        Prefix change if:
            - Data change.
            - Initial minimization parameters change (e.g. if the fit is ran twice)
            - Minimization parameters are changed.
            - Bounds are changed.

        Alternative prefix:
        >>> self.hash = md5(pickle.dumps([self.ngauss, self.data])).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.__dict__)).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.minparams)).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.cmnd)).hexdigest()[:5]
        """
        self.hash = md5(pickle.dumps(self)).hexdigest()[:5]
        prefix_str = ""
        if myprefix is not None:
            prefix_str = f"_{myprefix}"
        self.prefix = f"{self.ngauss}cmnd{prefix_str}_{self.hash}"

    def save_fit(self, objfile=None, useprefix=True, myprefix=None):
        """
        Pickle the result of a fit.
        Ex. F.save_fit(objfile='fit.pkl')

        Parameters
        ----------
        objfile : str, optional
            Name of the file where the fit will be stored. If None, the name is set
            by the routine as FitCMND.pkl.
        useprefix : bool, optional
            Use a prefix in the filename of the pickle file (default True).
            The prefix is normally {ngauss}cmnd_{hash}.
        myprefix : str, optional
            Custom prefix.

        Examples
        --------
        If objfile="fit.pkl", the final filename will be fit-1mnd_asa33.pkl
        """
        self.fig = None
        self._update_prefix(myprefix)
        if objfile is None:
            objfile = f"/tmp/FitCMND.pkl"
        if useprefix:
            parts = os.path.splitext(objfile)
            objfile = f"{parts[0]}-{self.prefix}{parts[1]}"

        # Avoid saving the temporary sample
        if hasattr(self, "_qsample"):
            del self._qsample

        # Avoid saving the Q-Q plot figures
        if hasattr(self, "fig_qq"):
            del self.fig_qq
        if hasattr(self, "ax_qq"):
            del self.ax_qq

        pickle.dump(self, open(objfile, "wb"))

    def set_bounds(self, boundw=None, bounds=None, boundr=None, boundsm=None):
        """
        Set the minimization parameters.
        Ex. F.set_bounds(boundsm=((-2, 1), (-3, 0)))

        Parameters
        ----------
        boundw : tuple, optional
            Bound of weights (default (-np.inf, np.inf)).
        bounds : tuple or list, optional
            Bounds of weights, mus, sigmas and rhos of each variable.
        boundr : tuple, optional
            Bound of rhos (default (-np.inf, np.inf)).
        boundsm : tuple of tuples, optional
            Bounds of averages (default (-np.inf, np.inf)).
            Normally the bounds on averages must be expressed in this way:
            boundsm = ((-min_1, max_1), (-min_2, max_2), ...)
            Example for nvars = 2:
            boundsm = ((-2, 1), (-3, 0))

        Returns
        -------
        bounds : tuple
            Formatted bounds for minimization.
        """
        if boundsm is None:
            # Use domain as bounds for means when domain is finite
            if getattr(self.cmnd, "_domain_bounds", None) is not None:
                boundsm = tuple(
                    (self.cmnd._domain_bounds[j][0], self.cmnd._domain_bounds[j][1])
                    for j in range(self.nvars)
                )
            else:
                boundsm = ((-np.inf, np.inf),) * self.nvars

        # Regular bounds
        if boundw is None:
            boundw = (-np.inf, np.inf)
        else:
            boundw = tuple([Util.f2u(bw, 1) for bw in boundw])

        if bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = tuple([Util.f2u(bs, self._sigmax) for bs in bounds])

        if boundr is None:
            boundr = (-np.inf, np.inf)
        else:
            boundr = tuple([Util.f2u(1 + br, 2) for br in boundr])

        bounds = (
            *((boundw,) * self.ngauss),
            *(boundsm * self.ngauss),
            *((bounds,) * self.nvars * self.ngauss),
            *((boundr,) * self.ngauss * int(self.nvars * (self.nvars - 1) / 2)),
        )
        self.bounds = bounds

        if self.ngauss == 1:
            bounds = bounds[1:]
        self.cmnd._str_params()
        return bounds


class FitFunctionCMND(MultiMinBase):
    """
    Fit a CMND to a function evaluated on a mesh (instead of to data points).

    The objective is to approximate the function by a composed multivariate normal
    distribution (CMND) by minimizing a loss between the function values and the
    CMND PDF on the same grid.

    Initialize with data=(Xmesh, Fmesh) and ngauss:
    - Univariate: Xmesh and Fmesh are 1D arrays (same length).
    - Multivariate: Xmesh is a tuple of ndarrays (e.g. from np.meshgrid),
      Fmesh is an array of the same shape as the grid.

    Attributes
    ----------
    X : numpy.ndarray
        Grid points, shape (N, nvars).
    F : numpy.ndarray
        Function values at the grid points, shape (N,).
    ngauss, nvars, cmnd, minparams, scales, uparams
        Same as in FitCMND (inherited).
    normalization : float
        Scale factor fitted so that normalization * cmnd.pdf(X) approximates F.
        Set by fit_data(); None before fitting.

    See Also
    --------
    quality_of_fit : Report R² and RMSE after fitting (call after fit_data).
    """

    _sigmax = 10
    _ignoreWarnings = True

    def __init__(self, data, ngauss=1):
        """
        Initialize FitFunctionCMND from function-on-mesh data.

        The domain is always set from the data (min/max of Xmesh per variable).

        Parameters
        ----------
        data : tuple (Xmesh, Fmesh)
            Xmesh: 1D array (univariate) or tuple of arrays (meshgrid, multivariate).
            Fmesh: function values at those points; same length or shape as the grid.
        ngauss : int
            Number of Gaussian components.

        Examples
        --------
        >>> F = mn.FitFunctionCMND(data=(Xmesh, Fmesh), ngauss=3)
        """
        if data is None:
            raise ValueError("data must be provided as (Xmesh, Fmesh)")

        if not isinstance(data, (tuple, list)):
            raise ValueError(
                "data must be a tuple or list of exactly 2 elements (Xmesh, Fmesh); "
                f"got {type(data).__name__}"
            )
        if len(data) != 2:
            raise ValueError(
                "data must have exactly 2 elements (Xmesh, Fmesh); "
                f"got {len(data)} elements"
            )

        Xmesh, Fmesh = data[0], data[1]
        self._X, self._F, nvars = self._parse_function_mesh_data(Xmesh, Fmesh)
        self.X = self._X
        self.F = self._F

        # Internal normalization for better numerical stability during optimization
        self._Fmax = float(np.max(np.abs(self.F)))
        if self._Fmax > 0:
            self._Fnorm = self.F / self._Fmax
        else:
            self._Fnorm = self.F
            self._Fmax = 1.0

        self.ngauss = ngauss
        self.nvars = nvars
        self.Ndim = ngauss * nvars
        self.Ncorr = int(nvars * (nvars - 1) / 2)
        self.domain = [
            [float(self.X[:, j].min()), float(self.X[:, j].max())] for j in range(nvars)
        ]

        self.cmnd = ComposedMultiVariateNormal(
            ngauss=ngauss, nvars=nvars, domain=self.domain, normalize_weights=False
        )
        self.normalization = None
        self._weight_scale = 100.0
        widths = [self.domain[j][1] - self.domain[j][0] for j in range(nvars)]
        if widths:
            self._sigmax = max(self._sigmax, 2.0 * max(widths))
        self.set_params()

        print("Loading a FitFunctionCMND object.")
        print(f"Number of gaussians: {self.ngauss}")
        print(f"Number of variables: {self.nvars}")
        print(f"Number of dimensions: {self.Ndim}")
        print(f"Number of grid points: {len(self.F)}")
        if self.domain is not None:
            print(f"Domain: {self.domain}")

    def set_params(self, mu=0.5, sigma=1.0, rho=0.5):
        """
        Set the value of the basic params (minparams, scales, uparams).
        Ex. F.set_params(mu=0.5, sigma=1.0, rho=0.5)
        """
        minparams = (
            [mu] * self.Ndim
            + [sigma] * self.Ndim
            + [1 + rho] * self.ngauss * self.Ncorr
        )
        scales = (
            [0] * self.Ndim
            + [self._sigmax] * self.Ndim
            + [2] * self.ngauss * self.Ncorr
        )
        if self.ngauss > 1:
            self.extrap = []
            minparams = [1 / self.ngauss] * self.ngauss + minparams
            w_scale = getattr(self, "_weight_scale", 1.0)
            scales = [w_scale] * self.ngauss + scales
        else:
            self.extrap = [1]

        self.minparams = np.array(minparams)
        bounds = getattr(self.cmnd, "_domain_bounds", None)
        if bounds is not None and self.ngauss > 1:
            for j in range(self.nvars):
                lo, hi = bounds[j][0], bounds[j][1]
                if np.isfinite(lo) or np.isfinite(hi):
                    a = lo if np.isfinite(lo) else hi - 1.0
                    b = hi if np.isfinite(hi) else lo + 1.0
                    for i in range(self.ngauss):
                        idx = self.ngauss + i * self.nvars + j
                        self.minparams[idx] = a + (i + 1) / (self.ngauss + 1) * (b - a)
        self.scales = np.array(scales)
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        self._initial_params_set_by_user = False

    def pmap(self, minparams):
        """Map minparams to stdcorr for cmnd.set_stdcorr."""
        stdcorr = np.array(self.extrap + list(minparams))
        if self.ngauss * self.Ncorr > 0:
            stdcorr[-self.ngauss * self.Ncorr :] -= 1
        return stdcorr

    def set_bounds(self, boundw=None, bounds=None, boundr=None, boundsm=None):
        """Set bounds for minimization. Returns formatted bounds tuple."""
        if boundsm is None:
            if getattr(self.cmnd, "_domain_bounds", None) is not None:
                boundsm = tuple(
                    (self.cmnd._domain_bounds[j][0], self.cmnd._domain_bounds[j][1])
                    for j in range(self.nvars)
                )
            else:
                boundsm = ((-np.inf, np.inf),) * self.nvars
        if boundw is None:
            boundw = (-np.inf, np.inf)
        else:
            boundw = tuple([Util.f2u(bw, 1) for bw in boundw])
        if bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = tuple([Util.f2u(bs, self._sigmax) for bs in bounds])
        if boundr is None:
            boundr = (-np.inf, np.inf)
        else:
            boundr = tuple([Util.f2u(1 + br, 2) for br in boundr])
        bounds = (
            *((boundw,) * self.ngauss),
            *(boundsm * self.ngauss),
            *((bounds,) * self.nvars * self.ngauss),
            *((boundr,) * self.ngauss * int(self.nvars * (self.nvars - 1) / 2)),
        )
        self.bounds = bounds
        if self.ngauss == 1:
            bounds = bounds[1:]
        self.cmnd._str_params()
        return bounds

    def set_initial_params(self, mus=None, sigmas=None, rhos=None):
        """Set initial values for the minimization parameters."""
        mus = np.asarray(mus, dtype=float) if mus is not None else None
        sigmas = np.asarray(sigmas, dtype=float) if sigmas is not None else None
        rhos = np.asarray(rhos, dtype=float) if rhos is not None else None

        def _broadcast_2d(arr, shape_2d, name, dims_desc):
            if arr is None:
                return None
            arr = np.atleast_1d(arr)
            if arr.ndim == 1:
                if arr.shape[0] != shape_2d[1]:
                    raise ValueError(
                        f"{name} 1D must have length {shape_2d[1]} ({dims_desc}), got {arr.shape[0]}"
                    )
                arr = np.tile(arr, (self.ngauss, 1))
            else:
                arr = np.atleast_2d(arr)
                if arr.shape != shape_2d:
                    raise ValueError(
                        f"{name} must have shape {shape_2d} or ({dims_desc}), got {arr.shape}"
                    )
            return arr

        if self.ngauss > 1:
            off_mu = self.ngauss
            off_sig = self.ngauss + self.Ndim
            off_rho = self.ngauss + 2 * self.Ndim
        else:
            off_mu = 0
            off_sig = self.nvars
            off_rho = self.nvars * 2

        if mus is not None:
            mus = _broadcast_2d(mus, (self.ngauss, self.nvars), "mus", "nvars")
            if self.ngauss > 1:
                self.minparams[off_mu : off_mu + self.Ndim] = mus.ravel()
            else:
                self.minparams[off_mu : off_mu + self.nvars] = mus.ravel()
        if sigmas is not None:
            sigmas = _broadcast_2d(sigmas, (self.ngauss, self.nvars), "sigmas", "nvars")
            if self.ngauss > 1:
                self.minparams[off_sig : off_sig + self.Ndim] = sigmas.ravel()
            else:
                self.minparams[off_sig : off_sig + self.nvars] = sigmas.ravel()
        if rhos is not None:
            rhos = _broadcast_2d(rhos, (self.ngauss, self.Ncorr), "rhos", "Ncorr")
            if self.ngauss > 1:
                self.minparams[off_rho : off_rho + self.ngauss * self.Ncorr] = (
                    1.0 + rhos.ravel()
                )
            else:
                self.minparams[off_rho : off_rho + self.Ncorr] = 1.0 + rhos.ravel()

        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        self._initial_params_set_by_user = True

        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.nvars)

    def _init_params_from_grid(self, use_function_support=True):
        """
        Set initial mus/sigmas from grid (self.X) when domain is finite.
        If use_function_support and self.F is available, spread mus only over
        points where F > 0.01*max(F) so components start in the support of the function.
        """
        bounds = getattr(self.cmnd, "_domain_bounds", None)
        if bounds is None or self.ngauss < 2:
            return
        X = np.asarray(self.X)
        if X.size == 0:
            return
        F = np.asarray(self.F).ravel() if use_function_support else None
        threshold = (
            max(0.01 * np.nanmax(F), 1e-300)
            if F is not None and F.size == X.shape[0]
            else None
        )
        for j in range(self.nvars):
            lo, hi = bounds[j][0], bounds[j][1]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            col = X[:, j]
            if threshold is not None:
                mask = F >= threshold
                if np.sum(mask) >= 2:
                    col = col[mask]
            pct = np.linspace(5, 95, self.ngauss)
            mus_init = np.percentile(col, pct)
            mus_init = np.sort(mus_init)
            mus_init = np.clip(mus_init, lo, hi)
            for i in range(self.ngauss):
                idx = self.ngauss + i * self.nvars + j
                self.minparams[idx] = mus_init[i]
            width = np.ptp(col) if len(col) >= 2 else (hi - lo)
            sigma_init = min(
                0.25 * (hi - lo),
                0.2 * (hi - lo) / max(1, self.ngauss),
                max(0.05 * width, np.std(col) * 0.5),
            )
            for i in range(self.ngauss):
                idx = self.ngauss + self.Ndim + i * self.nvars + j
                if idx < len(self.minparams):
                    self.minparams[idx] = sigma_init
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        if self.ngauss > 1:
            self._set_initial_weights_from_function()

    def _init_params_from_function_support(self, frac=0.01):
        """
        Set initial mus/sigmas from the support of the function (where F > frac*max(F)).
        Used when domain is not set so the optimizer starts in the region where the function is non-zero.
        """
        F = np.asarray(self.F).ravel()
        X = np.asarray(self.X)
        if X.size == 0 or F.size == 0:
            return
        threshold = max(frac * np.nanmax(F), 1e-300)
        mask = F >= threshold
        if not np.any(mask):
            mask = np.ones(F.size, dtype=bool)
        for j in range(self.nvars):
            col = X[mask, j]
            if len(col) < 2:
                col = X[:, j]
            pct = np.linspace(5, 95, self.ngauss)
            mus_init = np.percentile(col, pct)
            mus_init = np.sort(mus_init)
            for i in range(self.ngauss):
                idx = self.ngauss + i * self.nvars + j
                self.minparams[idx] = mus_init[i]
            width = np.ptp(col)
            sigma_init = min(
                self._sigmax * 0.5,
                0.2 * width / max(1, self.ngauss),
                max(0.05 * width, np.std(col) * 0.5),
            )
            if width <= 0:
                sigma_init = 1.0
            for i in range(self.ngauss):
                idx = self.ngauss + self.Ndim + i * self.nvars + j
                if idx < len(self.minparams):
                    self.minparams[idx] = sigma_init
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
        if self.ngauss > 1:
            self._set_initial_weights_from_function()

    def _set_initial_weights_from_function(self, n_sigma_window=2.0):
        """
        Set initial weights from the function F by integrating F in a window around each mu.
        So peaks with more area get larger initial weight; avoids starting with equal weights.
        """
        if self.ngauss < 2:
            return
        X = np.asarray(self.X)
        F = np.asarray(self.F).ravel()
        if X.size == 0 or F.size != X.shape[0]:
            return
        mus = np.array(
            [self.minparams[self.ngauss + i * self.nvars] for i in range(self.ngauss)]
        )
        sigmas = np.array(
            [
                self.minparams[self.ngauss + self.Ndim + i * self.nvars]
                for i in range(self.ngauss)
            ]
        )
        x = X[:, 0] if self.nvars == 1 else X
        if self.nvars == 1:
            x = np.asarray(x).ravel()
        masses = np.zeros(self.ngauss)
        for i in range(self.ngauss):
            if self.nvars == 1:
                half = n_sigma_window * max(sigmas[i], 1e-6)
                in_window = (x >= mus[i] - half) & (x <= mus[i] + half)
            else:
                dist = np.linalg.norm(X - mus[i], axis=1)
                half = n_sigma_window * max(np.mean(sigmas), 1e-6)
                in_window = dist <= half
            masses[i] = np.sum(F[in_window])
        if np.sum(masses) <= 0:
            return
        weights = masses / np.sum(masses)
        weights = np.clip(weights, 1e-6, 1.0 - 1e-6)
        weights = weights / np.sum(weights)
        self.minparams[: self.ngauss] = weights
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)

    def _loss_function(self, uparams):
        """
        Least-squares loss between F and (normalization * cmnd.pdf(X)).
        Normalization is computed analytically at each evaluation to minimize the residual sum of squares.
        Uses internally normalized F for better numerical stability.
        """
        minparams = Util.t_if(uparams, self.scales, Util.u2f)
        params = self.pmap(minparams)
        self.cmnd.set_stdcorr(params, self.nvars)
        pdf = self.cmnd.pdf(self.X)
        pdf = np.maximum(np.atleast_1d(pdf).astype(float), 1e-300)
        c = np.dot(self._Fnorm, pdf) / np.dot(pdf, pdf)
        residual = self._Fnorm - c * pdf
        return np.sum(residual**2)

    def fit_data(self, verbose=0, advance=0, mode="lsq", atol=None, rtol=None, maxgauss=50, wtol=1e-2, **args):
        """
        Fit the CMND to the function on the mesh by nonlinear least squares.

        Minimizes sum over grid points of (F - normalization * cmnd.pdf(X))^2.
        The normalization is obtained analytically at each step so that the residual
        sum of squares is minimized for the current CMND parameters. After fitting,
        self.normalization holds the scale so that normalization * cmnd.pdf(X) ≈ F.

        Ex. F.fit_data(tol=1e-6, options=dict(maxiter=500))

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (default 0).
        advance : int, optional
            If > 0, print progress every advance iterations (default 0).
        mode : str, optional
            Mode of fitting. Options: "lsq" (default), "multimodal", "noisy", "noisy_multimodal", or "adaptive".
            - "lsq": Standard least-squares fit with all parameters free.
            - "multimodal": Detects all peaks, sets ngauss to number of peaks found,
              fixes means to peak positions, fits only sigmas and weights.
            - "noisy": Smooths data first (~3% of points), detects peaks, selects
              the ngauss most prominent peaks, uses them as initial values for means,
              but leaves all parameters FREE to optimize. Best for noisy data.
            - "noisy_multimodal": Smooths data first (~3% of points), detects ALL peaks,
              sets ngauss to number of peaks found, fixes means at those positions,
              fits only sigmas and weights. Like "multimodal" but with smoothing for noisy data.
            - "adaptive": Incrementally increases ngauss from 1 until convergence criteria
              are met (controlled by atol and rtol). Automatically finds optimal number
              of Gaussians.
        atol : float, optional
            Absolute tolerance for R² in adaptive mode. Stops when R² >= atol (e.g. 0.99).
            Only used when mode="adaptive".
        rtol : float, optional
            Relative tolerance for R² improvement in adaptive mode. Stops when
            the average improvement over the last 3 ngauss values is less than rtol
            (e.g. 0.01). Evaluates improvement trend to avoid stopping prematurely.
            Only used when mode="adaptive".
        maxgauss : int, optional
            Maximum number of Gaussians to try in adaptive mode (default 50).
            Only used when mode="adaptive".
        wtol : float, optional
            Weight tolerance for pruning spurious Gaussians in adaptive mode (default 1e-2).
            After finding optimal ngauss, removes Gaussians with weights < wtol and
            refits with remaining components. Helps eliminate spurious components.
            Only used when mode="adaptive".
        **args
            Passed to scipy.optimize.minimize (e.g. method, tol, options).
        """
        from scipy.signal import find_peaks

        self.cmnd._ignoreWarnings = self._ignoreWarnings
        minargs = dict(method="Powell")
        minargs.update(args)

        # Handle adaptive mode
        if mode == "adaptive":
            if atol is None and rtol is None:
                raise ValueError("adaptive mode requires at least one of 'atol' or 'rtol' to be specified")
            
            if verbose > 0:
                print(f"Adaptive mode: starting with ngauss=1, maxgauss={maxgauss}")
                if atol is not None:
                    print(f"  Stop criterion: R² >= {atol}")
                if rtol is not None:
                    print(f"  Stop criterion: avg(ΔR²) over last 3 < {rtol}")
            
            r2_history = []  # Keep history of last 3 R² values
            best_ngauss = 1
            best_r2 = -np.inf
            
            for ng in range(1, maxgauss + 1):
                # Re-initialize with new ngauss
                self.ngauss = ng
                self.Ndim = self.ngauss * self.nvars
                self.Ncorr = int(self.nvars * (self.nvars - 1) / 2)
                self.cmnd = ComposedMultiVariateNormal(
                    ngauss=self.ngauss,
                    nvars=self.nvars,
                    domain=self.domain,
                    normalize_weights=False,
                )
                self.set_params()
                
                # Fit with standard lsq mode (reuse code by calling fit_data recursively with mode="lsq")
                # To avoid infinite recursion, we call the fitting code directly
                if verbose > 0:
                    print(f"\n  Trying ngauss={ng}...")
                
                # Initialize parameters if not set by user
                if not getattr(self, "_initial_params_set_by_user", False):
                    bounds_orig = getattr(self.cmnd, "_domain_bounds", None)
                    has_finite_domain = bounds_orig is not None and any(
                        np.isfinite(bounds_orig[j][0]) or np.isfinite(bounds_orig[j][1])
                        for j in range(self.nvars)
                    )
                    if has_finite_domain:
                        self._init_params_from_grid()
                    else:
                        self._init_params_from_function_support()
                
                # Perform optimization
                self.neval = 0
                def _advance_adaptive(x, show=False):
                    if advance > 0 and (self.neval % advance == 0 or show):
                        if self.neval == 0:
                            print(f"    Iterations:")
                        loss = self._loss_function(x)
                        print(f"      Iter {self.neval}: loss = {loss:.6g}")
                    self.neval += 1
                
                callback = _advance_adaptive if advance else None
                
                self.solution = minimize(
                    self._loss_function,
                    self.uparams,
                    callback=callback,
                    **minargs,
                )
                if advance:
                    _advance_adaptive(self.solution.x, show=True)
                
                self.uparams = self.solution.x
                self.minparams = Util.t_if(self.uparams, self.scales, Util.u2f)
                params = self.pmap(self.minparams)
                self.cmnd.set_stdcorr(params, self.nvars)
                pdf = self.cmnd.pdf(self.X)
                pdf = np.maximum(np.atleast_1d(pdf).astype(float), 1e-300)
                self.normalization = float(np.dot(self.F, pdf) / np.dot(pdf, pdf))
                
                # Calculate R²
                out = self.quality_of_fit(verbose=False)
                r2_current = out["R2"]
                
                if verbose > 0:
                    print(f"    R² = {r2_current:.6f}")
                
                # Add to history
                r2_history.append(r2_current)
                
                # Update best
                if r2_current > best_r2:
                    best_r2 = r2_current
                    best_ngauss = ng
                
                # Check stopping criteria
                stop_atol = atol is not None and r2_current >= atol
                
                # Evaluate rtol based on average improvement over last 3 values
                stop_rtol = False
                if rtol is not None and len(r2_history) >= 3:
                    # Get last 3 R² values
                    last_3 = r2_history[-3:]
                    # Calculate improvements between consecutive values
                    improvements = [last_3[i+1] - last_3[i] for i in range(len(last_3)-1)]
                    avg_improvement = np.mean(improvements)
                    stop_rtol = avg_improvement < rtol
                    
                    if verbose > 1:
                        print(f"      Last 3 R²: {last_3}")
                        print(f"      Improvements: {improvements}")
                        print(f"      Avg improvement: {avg_improvement:.6f}")
                
                if stop_atol or stop_rtol:
                    if verbose > 0:
                        if stop_atol:
                            print(f"\n✅ Stopped: R² = {r2_current:.6f} >= atol = {atol}")
                        if stop_rtol:
                            last_3 = r2_history[-3:]
                            improvements = [last_3[i+1] - last_3[i] for i in range(len(last_3)-1)]
                            avg_improvement = np.mean(improvements)
                            print(f"\n✅ Stopped: avg(ΔR²) over last 3 = {avg_improvement:.6f} < rtol = {rtol}")
                        print(f"Final ngauss = {ng}")
                    
                    # Prune spurious Gaussians with small weights
                    self._prune_and_refit_adaptive(wtol, verbose, advance, minargs)
                    return
            
            # Reached maxgauss
            if verbose > 0:
                print(f"\n⚠️  Reached maxgauss = {maxgauss}")
                print(f"Final R² = {best_r2:.6f} at ngauss = {best_ngauss}")
            
            # Prune spurious Gaussians with small weights
            self._prune_and_refit_adaptive(wtol, verbose, advance, minargs)
            return

        # Handle multimodal mode
        if mode == "multimodal":
            from scipy.signal import peak_widths, peak_prominences

            F_flat = np.asarray(self.F).ravel()
            # Detect peaks
            # Use a relative height threshold
            height_threshold = 0.05 * np.max(F_flat) if np.max(F_flat) > 0 else None
            peaks, properties = find_peaks(F_flat, height=height_threshold)

            if len(peaks) == 0:
                print(
                    "Warning: No peaks detected in multimodal mode. Fallback to existing configuration."
                )
            else:
                self.ngauss = len(peaks)

                # Calculate widths and prominences
                widths_samples, _, _, _ = peak_widths(F_flat, peaks, rel_height=0.5)
                prominences, _, _ = peak_prominences(F_flat, peaks)

                # Re-initialize CMND and parameters structure with new ngauss
                self.Ndim = self.ngauss * self.nvars
                self.Ncorr = int(self.nvars * (self.nvars - 1) / 2)
                self.cmnd = ComposedMultiVariateNormal(
                    ngauss=self.ngauss,
                    nvars=self.nvars,
                    domain=self.domain,
                    normalize_weights=False,
                )
                self.set_params()

                has_weights = self.ngauss > 1
                off_mu = self.ngauss if has_weights else 0
                off_sig = off_mu + self.Ndim

                # Set means to peak positions
                mus_peaks = self.X[peaks]

                # Set mus
                for i in range(self.ngauss):
                    idx_base = off_mu + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = mus_peaks[i]

                # Set sigmas based on widths
                # widths_samples is FWHM in indices.
                # Convert to physical units.
                # Assuming approximate uniform grid for width estimation
                if len(self.X) > 1:
                    dx = abs((self.X[-1, 0] - self.X[0, 0]) / (len(self.X) - 1))
                else:
                    dx = 1.0

                sigma_conversion = 1.0 / 2.355  # FWHM to sigma (approx)

                for i in range(self.ngauss):
                    width_physical = widths_samples[i] * dx
                    # Use a fraction of FWHM as sigma estimate
                    sigma_est = float(width_physical * sigma_conversion)
                    sigma_est = float(np.clip(sigma_est, 1e-6, 0.99 * self._sigmax))

                    idx_base = off_sig + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = sigma_est

                # Set weights based on prominences (or peak heights)
                # minparams[0:ngauss] are weights (if ngauss > 1)
                if has_weights:
                    weights = prominences / np.sum(prominences)
                    self.minparams[: self.ngauss] = weights

                # Function to create tight bounds for fixed parameters
                def fix_val(val):
                    return (val - 1e-9, val + 1e-9)

                # Construct bounds

                # Recalculate component bounds
                # Weights
                boundw = (-np.inf, np.inf)  # Transformed weights

                # Sigmas
                # We want standard bounds for sigmas.
                # In multimodal mode: restrict sigma to 2 * domain_width
                bounds_orig = getattr(self.cmnd, "_domain_bounds", None)
                domain_widths = [
                    bounds_orig[j][1] - bounds_orig[j][0]
                    for j in range(self.nvars)
                    if np.isfinite(bounds_orig[j][0]) and np.isfinite(bounds_orig[j][1])
                ]
                sigma_hi = 0.99 * self._sigmax
                if domain_widths:
                    width_max = max(domain_widths)
                    sigma_hi = min(sigma_hi, 2.0 * width_max)

                bound_sigma_trans = (
                    Util.f2u(1e-6, self._sigmax),
                    Util.f2u(sigma_hi, self._sigmax),
                )

                # Rhos
                bound_rho_trans = (-np.inf, np.inf)

                # Bounds list construction
                new_bounds = []

                # Weights (if ngauss > 1)
                if has_weights:
                    new_bounds.extend([boundw] * self.ngauss)

                # Means (Fixed)
                for i in range(self.ngauss):
                    for j in range(self.nvars):
                        mu_val = mus_peaks[i, j]
                        new_bounds.append((mu_val - 1e-9, mu_val + 1e-9))

                # Sigmas
                new_bounds.extend([bound_sigma_trans] * self.Ndim)

                # Rhos
                new_bounds.extend([bound_rho_trans] * (self.ngauss * self.Ncorr))

                minargs["bounds"] = tuple(new_bounds)

                # Sync uparams
                self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)

                self._initial_params_set_by_user = True

                if minargs.get("method") == "Powell":
                    minargs["method"] = "L-BFGS-B"

        # Handle noisy mode (free means - uses peak positions as initial values)
        if mode == "noisy":
            from scipy.signal import peak_widths, peak_prominences
            from scipy.ndimage import uniform_filter1d

            if self.nvars != 1:
                raise ValueError("mode='noisy' is only supported for univariate functions (nvars=1)")

            F_flat = np.asarray(self.F).ravel()
            
            # Smooth the data using a window ~2-3% of points
            n_points = len(F_flat)
            window_size = max(7, int(n_points * 0.03))
            if window_size % 2 == 0:
                window_size += 1
            
            F_smooth = uniform_filter1d(F_flat, size=window_size, mode='nearest')
            
            # Detect peaks in smoothed data
            height_threshold = 0.05 * np.max(F_smooth) if np.max(F_smooth) > 0 else None
            prominence_threshold = 0.005 * np.max(F_smooth) if np.max(F_smooth) > 0 else None
            peaks, properties = find_peaks(F_smooth, height=height_threshold, prominence=prominence_threshold)
            
            if len(peaks) == 0:
                print("Warning: No peaks detected in noisy mode. Using default initialization.")
            else:
                # Select the ngauss most prominent peaks
                widths_samples, _, _, _ = peak_widths(F_smooth, peaks, rel_height=0.5)
                prominences, _, _ = peak_prominences(F_smooth, peaks)
                
                if len(peaks) < self.ngauss:
                    print(
                        f"Warning: Only {len(peaks)} peaks detected but ngauss={self.ngauss}. "
                        f"Using all {len(peaks)} detected peaks."
                    )
                    num_peaks_to_use = len(peaks)
                    selected_indices = np.arange(len(peaks))
                else:
                    num_peaks_to_use = self.ngauss
                    # Sort by prominence and select top ngauss
                    sorted_indices = np.argsort(prominences)[::-1]
                    selected_indices = sorted_indices[:self.ngauss]
                
                selected_peaks = peaks[selected_indices]
                selected_widths = widths_samples[selected_indices]
                selected_prominences = prominences[selected_indices]
                
                # Sort selected peaks by position for consistency
                position_order = np.argsort(selected_peaks)
                selected_peaks = selected_peaks[position_order]
                selected_widths = selected_widths[position_order]
                selected_prominences = selected_prominences[position_order]
                
                # Set initial parameters based on detected peaks
                has_weights = self.ngauss > 1
                off_mu = self.ngauss if has_weights else 0
                off_sig = off_mu + self.Ndim
                
                # Set means to peak positions (as INITIAL VALUES, not fixed)
                mus_peaks = self.X[selected_peaks]
                for i in range(num_peaks_to_use):
                    idx_base = off_mu + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = mus_peaks[i]
                
                # Set sigmas based on widths
                if len(self.X) > 1:
                    dx = abs((self.X[-1, 0] - self.X[0, 0]) / (len(self.X) - 1))
                else:
                    dx = 1.0
                
                sigma_conversion = 1.0 / 2.355
                for i in range(num_peaks_to_use):
                    width_physical = selected_widths[i] * dx
                    sigma_est = float(width_physical * sigma_conversion)
                    sigma_est = float(np.clip(sigma_est, 1e-6, 0.99 * self._sigmax))
                    idx_base = off_sig + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = sigma_est
                
                # Set weights based on prominences
                if has_weights:
                    weights = selected_prominences / np.sum(selected_prominences)
                    self.minparams[: num_peaks_to_use] = weights
                
                # Update uparams and mark as user-set
                self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
                self._initial_params_set_by_user = True

        # Handle noisy_multimodal mode (fixed means at peak positions)
        elif mode == "noisy_multimodal":
            from scipy.signal import peak_widths, peak_prominences
            from scipy.ndimage import uniform_filter1d

            if self.nvars != 1:
                raise ValueError("mode='noisy' is only supported for univariate functions (nvars=1)")

            F_flat = np.asarray(self.F).ravel()
            
            # Smooth the data using a window ~2-3% of points for effective noise reduction
            # while preserving multiple peaks
            n_points = len(F_flat)
            # Window size: approximately 3% of points (balance between smoothing and peak preservation)
            window_size = max(7, int(n_points * 0.03))
            if window_size % 2 == 0:  # Make it odd for symmetry
                window_size += 1
            
            F_smooth = uniform_filter1d(F_flat, size=window_size, mode='nearest')
            
            # Detect peaks in smoothed data with balanced thresholds
            # Height threshold: 5% of max to catch significant peaks
            height_threshold = 0.05 * np.max(F_smooth) if np.max(F_smooth) > 0 else None
            # Prominence threshold: 7% of max (balanced between catching real peaks and avoiding noise)
            prominence_threshold = 0.07 * np.max(F_smooth) if np.max(F_smooth) > 0 else None
            peaks, properties = find_peaks(F_smooth, height=height_threshold, prominence=prominence_threshold)
            
            if len(peaks) == 0:
                print(
                    "Warning: No peaks detected in noisy_multimodal mode. Fallback to existing configuration."
                )
            else:
                # Use ALL detected peaks (like multimodal mode)
                self.ngauss = len(peaks)
                # Re-initialize with new ngauss
                self.Ndim = self.ngauss * self.nvars
                self.Ncorr = int(self.nvars * (self.nvars - 1) / 2)
                self.cmnd = ComposedMultiVariateNormal(
                    ngauss=self.ngauss,
                    nvars=self.nvars,
                    domain=self.domain,
                    normalize_weights=False,
                )
                self.set_params()
                
                # Calculate widths and prominences
                widths_samples, _, _, _ = peak_widths(F_smooth, peaks, rel_height=0.5)
                prominences, _, _ = peak_prominences(F_smooth, peaks)
                
                has_weights = self.ngauss > 1
                off_mu = self.ngauss if has_weights else 0
                off_sig = off_mu + self.Ndim
                
                # Set means to peak positions
                mus_peaks = self.X[peaks]
                
                for i in range(self.ngauss):
                    idx_base = off_mu + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = mus_peaks[i]
                
                # Set sigmas based on widths
                if len(self.X) > 1:
                    dx = abs((self.X[-1, 0] - self.X[0, 0]) / (len(self.X) - 1))
                else:
                    dx = 1.0
                
                sigma_conversion = 1.0 / 2.355  # FWHM to sigma
                
                for i in range(self.ngauss):
                    width_physical = widths_samples[i] * dx
                    sigma_est = float(width_physical * sigma_conversion)
                    sigma_est = float(np.clip(sigma_est, 1e-6, 0.99 * self._sigmax))
                    idx_base = off_sig + i * self.nvars
                    self.minparams[idx_base : idx_base + self.nvars] = sigma_est
                
                # Set weights based on prominences
                if has_weights:
                    weights = prominences / np.sum(prominences)
                    self.minparams[: self.ngauss] = weights
                
                # Construct bounds (means fixed, sigmas and weights free)
                boundw = (-np.inf, np.inf)
                bounds_orig = getattr(self.cmnd, "_domain_bounds", None)
                domain_widths = [
                    bounds_orig[j][1] - bounds_orig[j][0]
                    for j in range(self.nvars)
                    if np.isfinite(bounds_orig[j][0]) and np.isfinite(bounds_orig[j][1])
                ]
                sigma_hi = 0.99 * self._sigmax
                if domain_widths:
                    width_max = max(domain_widths)
                    sigma_hi = min(sigma_hi, 2.0 * width_max)
                
                bound_sigma_trans = (
                    Util.f2u(1e-6, self._sigmax),
                    Util.f2u(sigma_hi, self._sigmax),
                )
                bound_rho_trans = (-np.inf, np.inf)
                
                new_bounds = []
                if has_weights:
                    new_bounds.extend([boundw] * self.ngauss)
                
                # Fix means
                for i in range(self.ngauss):
                    for j in range(self.nvars):
                        mu_val = mus_peaks[i, j]
                        new_bounds.append((mu_val - 1e-9, mu_val + 1e-9))
                
                # Free sigmas
                new_bounds.extend([bound_sigma_trans] * self.Ndim)
                # Free rhos
                new_bounds.extend([bound_rho_trans] * (self.ngauss * self.Ncorr))
                
                minargs["bounds"] = tuple(new_bounds)
                self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)
                self._initial_params_set_by_user = True
                
                if minargs.get("method") == "Powell":
                    minargs["method"] = "L-BFGS-B"

        bounds_orig = getattr(self.cmnd, "_domain_bounds", None)
        has_finite_domain = bounds_orig is not None and any(
            np.isfinite(bounds_orig[j][0]) or np.isfinite(bounds_orig[j][1])
            for j in range(self.nvars)
        )
        if has_finite_domain and "bounds" not in args and mode not in ["multimodal", "noisy_multimodal"]:
            domain_widths = [
                bounds_orig[j][1] - bounds_orig[j][0]
                for j in range(self.nvars)
                if np.isfinite(bounds_orig[j][0]) and np.isfinite(bounds_orig[j][1])
            ]
            sigma_hi = 0.99 * self._sigmax
            if domain_widths:
                width_max = max(domain_widths)
                sigma_hi = min(sigma_hi, 0.25 * width_max)
            bounds_tuple = self.set_bounds(bounds=(1e-6, sigma_hi))
            minargs["bounds"] = bounds_tuple
            if minargs.get("method") == "Powell":
                minargs["method"] = "L-BFGS-B"
            if not getattr(self, "_initial_params_set_by_user", False):
                self._init_params_from_grid()
        else:
            if not getattr(self, "_initial_params_set_by_user", False):
                self._init_params_from_function_support()

        self.neval = 0

        def _advance(x, show=False):
            if advance > 0 and (self.neval % advance == 0 or show):
                if self.neval == 0:
                    print("Iterations:")
                loss = self._loss_function(x)
                print(f"  Iter {self.neval}: loss = {loss:.6g}")
            self.neval += 1

        callback = _advance if advance else None

        self.solution = minimize(
            self._loss_function,
            self.uparams,
            callback=callback,
            **minargs,
        )
        if advance:
            _advance(self.solution.x, show=True)

        self.uparams = self.solution.x
        self.minparams = Util.t_if(self.uparams, self.scales, Util.u2f)
        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.nvars)
        pdf = self.cmnd.pdf(self.X)
        pdf = np.maximum(np.atleast_1d(pdf).astype(float), 1e-300)
        self.normalization = float(np.dot(self.F, pdf) / np.dot(pdf, pdf))

    def _prune_and_refit_adaptive(self, wtol, verbose, advance, minargs):
        """
        Prune Gaussians with small weights and refit with remaining components.
        
        Used in adaptive mode to eliminate spurious Gaussians after convergence.
        
        Parameters
        ----------
        wtol : float
            Weight threshold. Gaussians with weight < wtol are removed.
        verbose : int
            Verbosity level.
        advance : int
            Print progress every advance iterations.
        minargs : dict
            Minimization arguments.
        """
        # Get current weights
        weights = self.cmnd.weights.ravel()
        
        if verbose > 0:
            print(f"\n🔍 Pruning Gaussians with weights < {wtol}:")
            print(f"   Current weights: {weights}")
        
        # Identify Gaussians to keep (weight >= wtol)
        keep_mask = weights >= wtol
        n_keep = np.sum(keep_mask)
        
        if n_keep == self.ngauss:
            if verbose > 0:
                print(f"   ✅ All {self.ngauss} Gaussians have sufficient weight. No pruning needed.")
            return
        
        if n_keep == 0:
            if verbose > 0:
                print(f"   ⚠️  All Gaussians have weight < {wtol}. Keeping at least 1.")
            # Keep the Gaussian with largest weight
            keep_mask = np.zeros_like(weights, dtype=bool)
            keep_mask[np.argmax(weights)] = True
            n_keep = 1
        
        # Extract parameters of Gaussians to keep
        mus_keep = self.cmnd.mus[keep_mask]
        sigmas_keep = self.cmnd.sigmas[keep_mask]
        weights_keep = weights[keep_mask]
        
        if verbose > 0:
            n_removed = self.ngauss - n_keep
            print(f"   🗑️  Removing {n_removed} Gaussian(s) with small weights")
            print(f"   ✅ Keeping {n_keep} Gaussian(s)")
            print(f"   Refitting with pruned components...")
        
        # Re-initialize with reduced ngauss
        self.ngauss = n_keep
        self.Ndim = self.ngauss * self.nvars
        self.Ncorr = int(self.nvars * (self.nvars - 1) / 2)
        self.cmnd = ComposedMultiVariateNormal(
            ngauss=self.ngauss,
            nvars=self.nvars,
            domain=self.domain,
            normalize_weights=False,
        )
        self.set_params()
        
        # Set initial parameters from pruned components
        self.set_initial_params(mus=mus_keep, sigmas=sigmas_keep)
        
        # Perform optimization with standard lsq
        self.neval = 0
        def _advance_prune(x, show=False):
            if advance > 0 and (self.neval % advance == 0 or show):
                if self.neval == 0:
                    print(f"      Iterations:")
                loss = self._loss_function(x)
                print(f"        Iter {self.neval}: loss = {loss:.6g}")
            self.neval += 1
        
        callback = _advance_prune if advance else None
        
        self.solution = minimize(
            self._loss_function,
            self.uparams,
            callback=callback,
            **minargs,
        )
        if advance:
            _advance_prune(self.solution.x, show=True)
        
        self.uparams = self.solution.x
        self.minparams = Util.t_if(self.uparams, self.scales, Util.u2f)
        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.nvars)
        pdf = self.cmnd.pdf(self.X)
        pdf = np.maximum(np.atleast_1d(pdf).astype(float), 1e-300)
        self.normalization = float(np.dot(self.F, pdf) / np.dot(pdf, pdf))
        
        if verbose > 0:
            out = self.quality_of_fit(verbose=False)
            print(f"   ✅ Refit complete: ngauss = {self.ngauss}, R² = {out['R2']:.6f}")

    def quality_of_fit(self, verbose=True):
        """
        Report quality-of-fit statistics after fitting.

        Computes the coefficient of determination (R²) and the root mean
        square error (RMSE) between the mesh function F and the fitted
        model (normalization * cmnd.pdf(X)). Call after fit_data().

        R² (coefficient of determination)
            R² = 1 - SS_res / SS_tot, where
            SS_res = sum((F - F_pred)²),  SS_tot = sum((F - mean(F))²).
            R² = 1 means a perfect fit; R² = 0 means the model explains
            no variance beyond the mean; R² < 0 is possible if the fit
            is worse than using the constant mean(F).

        RMSE (root mean square error)
            RMSE = sqrt(mean((F - F_pred)²)). Same units as F; lower
            is better.

        Parameters
        ----------
        verbose : bool, optional
            If True (default), print a short report.

        Returns
        -------
        dict
            Keys: 'R2', 'RMSE', 'SS_res', 'SS_tot', 'n_points'.

        Raises
        ------
        ValueError
            If fit_data() has not been run (normalization is None).
        """
        if self.normalization is None:
            raise ValueError(
                "quality_of_fit requires a fitted model; run fit_data() first"
            )
        F_obs = np.asarray(self.F).ravel().astype(float)
        pdf = self.cmnd.pdf(self.X)
        pdf = np.maximum(np.atleast_1d(pdf).astype(float), 1e-300)
        F_pred = self.normalization * pdf
        F_pred = np.asarray(F_pred).ravel().astype(float)
        n = F_obs.size
        ss_res = float(np.sum((F_obs - F_pred) ** 2))
        mean_f = float(np.mean(F_obs))
        ss_tot = float(np.sum((F_obs - mean_f) ** 2))
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = float("nan")
        rmse = np.sqrt(np.mean((F_obs - F_pred) ** 2))
        out = {
            "R2": r2,
        }
        if verbose:
            print("Quality of fit (after fit_data):")
            print(f"  R²    = {out['R2']:.6g}  (1 = perfect, 0 = no better than mean)")
        return out

    def plot_fit(
        self,
        figsize=(8, 5),
        xlabel=None,
        ylabel=None,
        n_curve=300,
        sargs=None,
        largs=None,
        **kwargs,
    ):
        """
        Plot original function (data) and fitted CMND for univariate fits.

        Draws the mesh data (X, F) and the fitted model normalization * cmnd.pdf(X).
        Only implemented for nvars == 1.
        Ex. F.plot_fit(xlabel=r'$\\tau$', ylabel='ISR', sargs=dict(ms=2, color='C0'))

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (default (8, 5)).
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        n_curve : int, optional
            Number of points for the smooth PDF curve (default 300).
        sargs : dict, optional
            Options for the data (points) plot. Default: dict(ms=1.5, label='data').
            Passed to ax.plot for the (x, F) points.
        largs : dict, optional
            Options for the fit line (norm × CMND PDF). Default: dict(color='k',
            lw=2, label='fit (norm × CMND PDF)').
        **kwargs
            Extra arguments merged into the data plot when sargs is None; else ignored.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure handle.

        Examples
        --------
        >>> F.plot_fit(xlabel=r'$\\tau$', ylabel='ISR')
        >>> F.plot_fit(sargs=dict(ms=2, color='C0'), largs=dict(color='C1', lw=3))
        """
        if self.nvars != 1:
            raise NotImplementedError(
                "plot_fit is only implemented for univariate (nvars=1) fits"
            )
        from matplotlib import pyplot as plt

        if sargs is None:
            sargs = dict(ms=1.5, label="data")
            sargs.update(kwargs)
        else:
            sargs = dict(sargs)
            sargs.setdefault("label", "data")
            sargs.setdefault("ms", 1.5)
        if largs is None:
            largs = dict(color="k", lw=2, label="fit (norm × CMND PDF)")
        else:
            largs = dict(largs)
            largs.setdefault("label", "fit (norm × CMND PDF)")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x = np.asarray(self.X).ravel()
        f = np.asarray(self.F).ravel()

        ax.plot(x, f, "o", **sargs)

        if self.normalization is not None:
            x_min, x_max = x.min(), x.max()
            margin = max(1e-6, 0.05 * (x_max - x_min))
            x_curve = np.linspace(x_min - margin, x_max + margin, n_curve)
            pdf_vals = self.cmnd.pdf(x_curve.reshape(-1, 1))
            model_vals = self.normalization * np.atleast_1d(pdf_vals)
            ax.plot(x_curve, model_vals, "-", **largs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        fig.tight_layout()
        if not getattr(self, "_watermark_added_plot_fit", False):
            multimin_watermark(ax, frac=0.5)
            self._watermark_added_plot_fit = True
        self.fig = fig
        return fig

    def _parse_function_mesh_data(self, Xmesh, Fmesh):
        """
        Convert (Xmesh, Fmesh) from mesh/function format to points X (N, nvars) and values F (N,).

        For univariate: Xmesh and Fmesh are 1D arrays of the same length.
        For multivariate: Xmesh is a tuple of ndarrays (e.g. from np.meshgrid), Fmesh is an array
        of the same shape as the grid.

        Parameters
        ----------
        Xmesh : array-like or tuple of array-like
            Evaluation points. 1D array (n,) for univariate, or tuple of arrays for a grid.
        Fmesh : array-like
            Function values at those points. Shape (n,) for 1D or same shape as grid for mesh.

        Returns
        -------
        X : numpy.ndarray
            Points, shape (N, nvars).
        F : numpy.ndarray
            Values, shape (N,).
        nvars : int
            Number of variables.
        """
        Fmesh = np.asarray(Fmesh, dtype=float)
        if isinstance(Xmesh, (list, tuple)) and len(Xmesh) > 0:
            # Multivariate: meshgrid-style (x1, x2, ...)
            grids = [np.asarray(x, dtype=float) for x in Xmesh]
            nvars = len(grids)
            shape = grids[0].shape
            for g in grids:
                if g.shape != shape:
                    raise ValueError(
                        "Xmesh grid components must have the same shape; "
                        f"got {[g.shape for g in grids]}"
                    )
            if Fmesh.shape != shape:
                raise ValueError(
                    f"Fmesh shape {Fmesh.shape} must match grid shape {shape}"
                )
            X = np.column_stack([g.ravel() for g in grids])
            F = Fmesh.ravel()
        else:
            # Univariate: Xmesh and Fmesh are 1D
            Xmesh = np.asarray(Xmesh, dtype=float)
            if Xmesh.ndim != 1:
                raise ValueError(
                    "For univariate, Xmesh must be 1D; for multivariate use a tuple of arrays"
                )
            n = len(Xmesh)
            if Fmesh.size != n:
                raise ValueError(
                    f"Xmesh length ({n}) and Fmesh size ({Fmesh.size}) must match"
                )
            nvars = 1
            X = Xmesh.reshape(-1, 1)
            F = np.asarray(Fmesh, dtype=float).ravel()
        return X, F, nvars
