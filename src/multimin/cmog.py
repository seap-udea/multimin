"""
C-optimized routines for MultiMin.
"""

import ctypes
import os
import sys
import numpy as np


def _load_lib():
    """Load the multimin shared library."""
    # This file is in src/multimin/cmog.py
    # Library is in src/multimin/lib/multimin.dylib
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Library is expected in the 'lib' subdirectory of the installed package
    lib_dir = os.path.join(curr_dir, "lib")

    candidates = [
        os.path.join(lib_dir, "multimin.dylib"),  # macOS
        os.path.join(lib_dir, "multimin.so"),  # Linux
    ]

    lib = None
    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                pass

    if lib is None:
        # If we can't load it, we might want to warn or raise, but for now let's just return None
        # and let the caller handle it (or fail when accessing attributes)
        return None

    # Bindings

    # void nmd_batch(double *X, int n_points, int k, double *mu, double *sigma, double *results)
    if hasattr(lib, "nmd_batch"):
        lib.nmd_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        lib.nmd_batch.restype = None

    # void tnmd_batch(double *X, int n_points, int k, double *mu, double *sigma,
    #                 double *lb, double *ub, double inv_Z, double *results)
    if hasattr(lib, "tnmd_batch"):
        lib.tnmd_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        lib.tnmd_batch.restype = None

    # void mog_batch(double *X, int n_points, int k, int n_comps, double *weights,
    #                double *mus, double *sigmas, double *results)
    if hasattr(lib, "mog_batch"):
        lib.mog_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        lib.mog_batch.restype = None

    # void tmog_batch(double *X, int n_points, int k, int n_comps, double *weights,
    #                double *mus, double *sigmas, double *lb, double *ub, double *inv_Zs,
    #                double *results)
    if hasattr(lib, "tmog_batch"):
        lib.tmog_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        lib.tmog_batch.restype = None

    return lib


_lib = _load_lib()
has_c_lib = _lib is not None


def nmd_c(X, mu, Sigma):
    """
    Compute Normal Multivariate Distribution PDF using C library.

    Parameters
    ----------
    X : array-like
        Points (n_points, k) or (k,).
    mu : array-like
        Mean vector (k,).
    Sigma : array-like
        Covariance matrix (k, k).

    Returns
    -------
    pdf : ndarray
        PDF values.
    """
    if not has_c_lib:
        raise RuntimeError("C library not loaded. Compile it first.")

    X = np.atleast_2d(X)
    n_points, k = X.shape
    X_flat = np.ascontiguousarray(X.flatten(), dtype=np.float64)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    Sigma_flat = np.ascontiguousarray(Sigma, dtype=np.float64).flatten()
    results = np.zeros(n_points, dtype=np.float64)

    _lib.nmd_batch(X_flat, n_points, k, mu, Sigma_flat, results)

    # If input was single point, return scalar? Util.nmd does.
    # But usually batch is for arrays. The C wrapper expects batch.
    # Let's return array always, caller handles scalar conversion if needed,
    # or consistent with get_function usage which passes X from meshgrid (usually large).
    return results


def tnmd_c(X, mu, Sigma, a, b, Z=1.0):
    """
    Compute Truncated Normal Multivariate Distribution PDF using C library.

    Parameters
    ----------
    X : array-like
        Points (n_points, k).
    mu : array-like
        Mean (k,).
    Sigma : array-like
        Covariance (k, k).
    a : array-like
        Lower bounds (k,).
    b : array-like
        Upper bounds (k,).
    Z : float
        Normalization constant.

    Returns
    -------
    pdf : ndarray
    """
    if not has_c_lib:
        raise RuntimeError("C library not loaded.")

    X = np.atleast_2d(X)
    n_points, k = X.shape
    X_flat = np.ascontiguousarray(X.flatten(), dtype=np.float64)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    Sigma_flat = np.ascontiguousarray(Sigma, dtype=np.float64).flatten()

    lb = np.ascontiguousarray(a, dtype=np.float64)
    ub = np.ascontiguousarray(b, dtype=np.float64)
    inv_Z = 1.0 / Z if Z > 0 else 1.0

    results = np.zeros(n_points, dtype=np.float64)

    _lib.tnmd_batch(X_flat, n_points, k, mu, Sigma_flat, lb, ub, inv_Z, results)
    return results


def mog_c(X, weights, mus, Sigmas):
    """
    Compute full MoG PDF using C library (sum of components).

    Parameters
    ----------
    X : array-like (n_points, k) or (k,)
    weights : array-like (n_comps,)
    mus : array-like (n_comps, k)
    Sigmas : array-like (n_comps, k, k)
    """
    if not has_c_lib:
        raise RuntimeError("C library not loaded.")

    X = np.atleast_2d(X)
    n_points, k = X.shape
    X_flat = np.ascontiguousarray(X.flatten(), dtype=np.float64)

    weights = np.ascontiguousarray(weights, dtype=np.float64)
    n_comps = len(weights)

    mus = np.ascontiguousarray(mus, dtype=np.float64).flatten()
    Sigmas = np.ascontiguousarray(Sigmas, dtype=np.float64).flatten()

    results = np.zeros(n_points, dtype=np.float64)

    _lib.mog_batch(X_flat, n_points, k, n_comps, weights, mus, Sigmas, results)
    return results


def tmog_c(X, weights, mus, Sigmas, a, b, Zs):
    """
    Compute full Truncated MoG PDF using C library (sum of components).

    Parameters
    ----------
    X : array-like (n_points, k) or (k,)
    weights : array-like (n_comps,)
    mus : array-like (n_comps, k)
    Sigmas : array-like (n_comps, k, k)
    a, b : array-like (k,) - bounds
    Zs : array-like (n_comps,) - normalization constants per component
    """
    if not has_c_lib:
        raise RuntimeError("C library not loaded.")

    X = np.atleast_2d(X)
    n_points, k = X.shape
    X_flat = np.ascontiguousarray(X.flatten(), dtype=np.float64)

    weights = np.ascontiguousarray(weights, dtype=np.float64)
    n_comps = len(weights)

    mus = np.ascontiguousarray(mus, dtype=np.float64).flatten()
    Sigmas = np.ascontiguousarray(Sigmas, dtype=np.float64).flatten()

    lb = np.ascontiguousarray(a, dtype=np.float64)
    ub = np.ascontiguousarray(b, dtype=np.float64)

    inv_Zs = np.ascontiguousarray(1.0 / np.array(Zs), dtype=np.float64)

    results = np.zeros(n_points, dtype=np.float64)

    _lib.tmog_batch(
        X_flat, n_points, k, n_comps, weights, mus, Sigmas, lb, ub, inv_Zs, results
    )
    return results
