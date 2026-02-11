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
Utility and statistics classes for MultiMin package.

Contains:
- Util: General utility functions for data transformation and time tracking
- Stats: Statistical utilities for probability and covariance computations
"""

import os
import math
import string
import numpy as np
import spiceypy as spy
from time import time
from scipy.stats import norm, multivariate_normal as multinorm, truncnorm

# Import base class from package
from .base import MultiMinBase, ROOTDIR


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

    @staticmethod
    def docstring_summary(doc):
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

    @staticmethod
    def nmd(X, mu, Sigma):
        """PDF of a multivariate normal at x; mu=mean vector, Sigma=covariance matrix.
        Ex. Util.nmd(X, mu, Sigma)

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

    @staticmethod
    def tnmd(X, mu, Sigma, a, b, Z=None):
        """PDF of a truncated (multivariate) normal at X; single evaluation like nmd. Uses
        scipy.stats.truncnorm for 1D (one call). For nD, returns normal PDF in the box [a, b]
        divided by the normalization constant Z.
        Ex. Util.tnmd(X, mu, Sigma, a, b)

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
            Z = Util._norm_const_box(mu, Sig, a, b)
        out = np.where(in_box, np.maximum(raw / Z, 1e-300), 0.0)
        return float(out[0]) if out.size == 1 else out

    @staticmethod
    def _norm_const_box(mu, Sigma, a, b):
        """P(a < X < b) for X ~ N(mu, Sigma). Used when Z is None in tnmd (nD)."""
        n = len(mu)
        if n == 1:
            sig = np.sqrt(float(np.asarray(Sigma).ravel()[0]))
            return float(norm.cdf((b[0] - mu[0]) / sig) - norm.cdf((a[0] - mu[0]) / sig))
        draws = multinorm.rvs(mu, Sigma, size=50000, random_state=42)
        in_box = np.all((draws >= a) & (draws <= b), axis=1)
        return max(float(np.mean(in_box)), 1e-300)

    @staticmethod
    def props_to_properties(properties, nvars, ranges=None):
        """Convert properties (list or DensityPlot-style dict) to properties dict for DensityPlot.
        Ex. Util.props_to_properties(properties, nvars, ranges)

        - If properties is a dict: use as-is (DensityPlot-style); each value must have 'label'
          and optionally 'range'. Keys define variable names; first nvars keys are used.
        - If properties is a list or sequence: use each element as variable name and as axis
          label, with range=None unless ranges is provided (backward compatible).
        - If properties is None: use ascii_letters and ranges from the ranges argument.

        Parameters
        ----------
        properties : dict, list or None
            Properties specification.
        nvars : int
            Number of variables.
        ranges : list, optional
            Ranges for each variable.

        Returns
        -------
        dict
            Properties dictionary.
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
