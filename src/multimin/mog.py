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
Mixture of Gaussians (MoG) implementation.

Contains:
- MixtureOfGaussians: Main class for MoG representation and sampling
"""

import itertools
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal as multinorm, truncnorm

# Import from package modules
from .base import MultiMinBase
from .util import Util, Stats
from .plotting import multimin_watermark, MultiPlot
from .version import __version__


class MixtureOfGaussians(MultiMinBase):
    r"""
    The Mixture of Gaussians (MoG).

    We conjecture that any multivariate distribution function :math:`p(\tilde U):\Re^{N}\rightarrow\Re`,
    where :math:`\tilde U:(u_1,u_2,u_3,\ldots,u_N)` and :math:`u_i` are random variables, can be approximated
    with an arbitrary precision by a normalized linear combination of :math:`M` Multivariate Normal Distributions
    (MND):

    .. math::

        p(\tilde U) \approx \mathcal{C}_{M,k}(\tilde U; \{w_k\}_M, \{\mu_k\}_M, \{\Sigma_k\}_M) \equiv \sum_{i=1}^{M} w_i\;\mathcal{N}(\tilde U; \tilde \mu_i, \Sigma_i)

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

    When a finite ``domain`` is set (e.g. ``domain=[[0, 1], None]``), the MoG uses truncated normals for the
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
    There are several ways of initialize a MoG:

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
       In this case the basic properties of the MoG are set.
       For univariate (one variable), mus may be a 1-D array of means, e.g. [0, 2],
       and Sigmas a 1-D array of variances, e.g. [1.0, 0.25].
       domain: list of length nvars; each element is None (unbounded) or [low, high]
       (finite support). Example: [None, [0, 1], None] for variable 1 in [0, 1].
       normalize_weights: if True (default), weights are scaled so sum(weights)=1 (proper PDF).
       If False, weights are used as-is (sum may != 1); useful for function fitting with
       an overall scale (e.g. FitFunctionMoG).

    Examples
    --------
    Example 1: Initialization using explicit arrays for means and weights.

    >>> # Define means for 2 Gaussian components in 2D
    >>> mus = [[0, 0], [1, 1]]
    >>> # Define weights (normalization is handled automatically)
    >>> weights = [0.1, 0]
    >>> # Create the MoG object
    >>> MND1 = MixtureOfGaussians(mus=mus, weights=weights)
    >>> # Set covariance matrices explicitly
    >>> MND1.set_sigmas([[[1, 0.2], [0, 1]], [[1, 0], [0, 1]]])
    >>> print(MND1)

    Example 2: Initialization using a flattened parameter array.

    >>> # Flattened parameters: [weights..., mus..., flattened_covariances...]
    >>> params = [0.1, 0.9, 0, 0, 1, 1, 1, 0.2, 0.2, 1, 1, 0, 0, 1]
    >>> # Create MoG object specifying number of variables
    >>> MND2 = MixtureOfGaussians(params=params, nvars=2)
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

    def drop(self, index):
        """
        Drop one of the gaussians.
        Ex. mog.drop(0)

        Parameters
        ----------
        index : int
            Index of the gaussian to be dropped.
        """
        if index < 0 or index >= self.ngauss:
            raise ValueError(f"Index {index} out of range for ngauss={self.ngauss}")

        print(f"Components before dropping: {self.ngauss}")
        print(self.tabulate())

        # Remove the gaussian
        self.weights = np.delete(self.weights, index)
        self.mus = np.delete(self.mus, index, axis=0)
        self.Sigmas = np.delete(self.Sigmas, index, axis=0)
        self.ngauss -= 1

        # Check sigmas
        self._check_sigmas()

        # Update params
        self._flatten_params()
        self._flatten_stdcorr()

        # Update weights
        if self.normalize_weights:
            self._normalize_weights(self.weights)

        print(f"Dropped gaussian {index}")
        print(self.tabulate())

        self._Z_cache = None

    def update_params(self, weights=None, mus=None, sigmas=None, rhos=None):
        """Update MoG parameters in-place using FitMoG-like syntax.

        This method updates the internal parameters of the composed distribution
        (means, standard deviations, and correlations) using the same broadcasting
        rules as :meth:`FitMoG.set_initial_params`.

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
                    "initialize the MoG with Sigmas or stdcorr first."
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
        Set the properties of the MoG from flatten params. After setting it generate flattend stdcorr
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
        Set the properties of the MoG from flatten stdcorr. After setting it generate flattened
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
        Ex. mog.pdf([1.0, 0.5])

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
                    f"Point X has length {len(X)} but this MoG has nvars={self.nvars}"
                )
        elif X.ndim == 2:
            if X.shape[1] != self.nvars:
                raise ValueError(
                    f"Points X have {X.shape[1]} dimensions but this MoG has nvars={self.nvars}"
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
        """Plot only the PDF of the MoG.

        For univariate distributions (``nvars==1``) this produces a single curve.
        For multivariate distributions (``nvars>=2``) this produces density panels
        using :class:`MultiPlot`. For ``nvars>2`` the density shown in each panel
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
        G : MultiPlot
            Handle to the density plot grid.
        """
        self._check_params(self.params)
        properties = Util.props_to_properties(properties, self.nvars, ranges)
        G = MultiPlot(properties, figsize=figsize)

        G.mog_pdf(self, grid_size=grid_size, cmap=cmap, colorbar=colorbar)

        return G

    @Util.timer
    def rvs(self, Nsam=1, max_tries=100000):
        """
        Generate a random sample of points following this Multivariate distribution. When domain is
        finite, samples are drawn inside the domain (rejection sampling for multivariate; truncated
        normal for univariate).
        Ex. mog.rvs(1000)

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

    def sample_mog_likelihood(
        self, uparams, data=None, pmap=None, tset="stdcorr", scales=[], verbose=0
    ):
        """
        Compute the negative value of the logarithm of the likelihood of a sample.
        Ex. mog.sample_mog_likelihood(uparams, data=data, pmap=pmap)

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
            Verbosity level (0, none, 1: input parameters, 2: full definition of the MoG) (default 0).

        Returns
        -------
        log_l : float
            Negative log-likelihood.


        """
        # Map unbound minimization parameters into their right range
        minparams = np.array(Util.t_if(uparams, scales, Util.u2f))

        # Map minimizaiton parameters into MoG parameters
        params = np.array(pmap(minparams))

        if verbose >= 1:
            print("*" * 80)
            print(f"Minimization parameters: {minparams.tolist()}")
            print(f"MoG parameters: {params.tolist()}")

        # Update MoG parameters according to type of minimization parameters
        if tset == "params":
            self.set_params(params, self.nvars)
        else:
            self.set_stdcorr(params, self.nvars)

        if verbose >= 2:
            print("MoG:")
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
        Plot a sample of the MoG.
        Ex. plot_sample(N=1000, sargs=dict(s=0.5))

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to plot. If None it generate a sample.
        N : int, optional
            Number of points to generate the sample (default 10000).
        properties : list or dict, optional
            Property names or MultiPlot-style properties. List (e.g. properties=["x","y"]): each
            element is used as axis label, range=None. Dict: same as MultiPlot, e.g.
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
        Returns
        -------
        G : matplotlib.figure.Figure or MultiPlot
            Graphic handle. If nvars = 2, it is a figure object, otherwise is a MultiPlot instance.

        Examples
        --------
        Example 1: Plotting a generated sample.

        >>> # Generate and plot 10000 points
        >>> G = MoG.plot_sample(N=10000, sargs=dict(s=1, c='r'))
        >>> # Plot with histogram bins
        >>> G = MoG.plot_sample(N=1000, sargs=dict(s=1, c='r'), hargs=dict(bins=20))

        Example 2: Plotting for a 2D distribution.

        >>> MoG = MixtureOfGaussians(ngauss=1, nvars=2)
        >>> fig = MoG.plot_sample(N=1000, hargs=dict(bins=20), sargs=dict(s=1, c='r'))

        Example 3: Defining and plotting a complex MoG.

        >>> # Define parameters for 2 Gaussians in 2D
        >>> params = [0.1, 0.9, 0.0, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.0, 1.0]
        >>> MND2 = MixtureOfGaussians(params=params, nvars=2)
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

        properties = Util.props_to_properties(properties, self.nvars, ranges)
        G = MultiPlot(properties, figsize=figsize)
        ymax = -1e100
        if hargs is not None:
            G.sample_hist(self.data, **hargs)
            if self.nvars == 1:
                ymax = max(ymax, G.axs[0][0].get_ylim()[1])
        if sargs is not None:
            G.sample_scatter(self.data, **sargs)
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

    def get_function(
        self, print_code=True, decimals=6, type="python", properties=None, cmog=False
    ):
        """
        Return the source code of ``mog(X)`` and an executable function (type='python'),
        or LaTeX code with parameters in \\begin{array} (type='latex').
        Ex. code, mog = MoG.get_function()

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
            If a dict (e.g. from MultiPlot), its keys are used as variable names (mu_x, sigma_x, ...).
            If a sequence, its elements are used in order. Length must match the number of variables.
        cmog : bool, optional
            If True and type='python', generate code using the C-optimized routines ``cmog.nmd_c``
            and ``cmog.tnmd_c``. Default False.

        Returns
        -------
        tuple (str, callable or None)
            First element: source code (Python or LaTeX). Second: callable mog if type='python', else None.

        Examples
        --------
        >>> code, mog = MoG.get_function()
        >>> latex_str, _ = MoG.get_function(type='latex')
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

        if cmog:
            # C-optimized generation using batch functions
            imports = [
                "import numpy as np",
                "from multimin import Util, cmog",
            ]
            lines = imports + [""]

            # Helper to format array literals with indentation
            def fmt_arr_rows(arr, indent="    "):
                def fmt_val(x):
                    if np.isposinf(x):
                        return "np.inf"
                    elif np.isneginf(x):
                        return "-np.inf"
                    else:
                        return f"{x:.{decimals}g}"

                # arr is expected to be at least 1D
                if arr.ndim == 1:
                    # Single line
                    return (
                        indent
                        + "np.array(["
                        + ", ".join(fmt_val(x) for x in arr)
                        + "], dtype=np.float64)"
                    )
                elif arr.ndim == 2:
                    # One row per line
                    rows = []
                    rows.append(indent + "np.array([")
                    for row in arr:
                        row_str = ", ".join(fmt_val(x) for x in row)
                        rows.append(indent + f"    [{row_str}],")
                    rows.append(indent + "], dtype=np.float64)")
                    return "\n".join(rows)
                elif arr.ndim == 3:
                    # For 3D sigmas (n_comps, k, k)
                    # We can format this as a list of matrices
                    rows = []
                    rows.append(indent + "np.array([")
                    for matrix in arr:
                        rows.append(indent + "    [")
                        for row in matrix:
                            row_str = ", ".join(fmt_val(x) for x in row)
                            rows.append(indent + f"        [{row_str}],")
                        rows.append(indent + "    ],")
                    rows.append(indent + "], dtype=np.float64)")
                    return "\n".join(rows)
                return str(arr)

            # Weights (1D)
            w_str_lines = fmt_arr_rows(np.array(self.weights), indent="    ")

            # Mus (2D)
            mus_str_lines = fmt_arr_rows(np.array(self.mus), indent="    ")

            # Sigmas (3D)
            sigmas_str_lines = fmt_arr_rows(np.array(self.Sigmas), indent="    ")

            if has_finite_domain:
                # Bounds
                a_list = [
                    float(bounds[j][0]) if np.isfinite(bounds[j][0]) else -np.inf
                    for j in range(self.nvars)
                ]
                b_list = [
                    float(bounds[j][1]) if np.isfinite(bounds[j][1]) else np.inf
                    for j in range(self.nvars)
                ]
                a_str = fmt_arr_rows(np.array(a_list), indent="    ")
                b_str = fmt_arr_rows(np.array(b_list), indent="    ")

                # Zs
                Zs = np.array(
                    [float(self._normalization_constant(n)) for n in range(self.ngauss)]
                )
                Zs_str = fmt_arr_rows(Zs, indent="    ")

                # Define function with internal variables
                lines.append("def mog(X):")
                lines.append(f"    weights = {w_str_lines.strip()}")
                lines.append(f"    mus = {mus_str_lines.strip()}")
                lines.append(f"    Sigmas = {sigmas_str_lines.strip()}")
                lines.append(f"    a = {a_str.strip()}")
                lines.append(f"    b = {b_str.strip()}")
                lines.append(f"    Zs = {Zs_str.strip()}")
                lines.append(
                    "    return cmog.tmog_c(X, weights, mus, Sigmas, a, b, Zs)"
                )
            else:
                lines.append("def mog(X):")
                lines.append(f"    weights = {w_str_lines.strip()}")
                lines.append(f"    mus = {mus_str_lines.strip()}")
                lines.append(f"    Sigmas = {sigmas_str_lines.strip()}")
                lines.append("    return cmog.mog_c(X, weights, mus, Sigmas)")

        else:
            # Standard Python generation (loop over components)
            if has_finite_domain:
                imports = [
                    "import numpy as np",
                    "from multimin import Util",
                ]
            else:
                imports = ["from multimin import Util"]

            lines = imports + ["", "def mog(X):", ""]

            if has_finite_domain:
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
                            "    n{} = Util.tnmd(X, mu{}_{}, sigma{}_{}, a, b)".format(
                                i, i, vname, i, vname
                            )
                        )
                    else:
                        sigma_val = round(float(np.sqrt(Sigma.ravel()[0])), decimals)
                        lines.append("    mu{}_{} = {}".format(i, vname, mu_val))
                        lines.append("    sigma{}_{} = {}".format(i, vname, sigma_val))
                        lines.append(
                            "    n{} = Util.nmd(X, mu{}_{}, sigma{}_{})".format(
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
                            "    n{} = Util.tnmd(X, mu{}, Sigma{}, a, b, Z=Z{})".format(
                                i, i, i, i
                            )
                        )
                    else:
                        lines.append(
                            "    n{} = Util.nmd(X, mu{}, Sigma{})".format(i, i, i)
                        )
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
        # Execute the code to obtain the callable mog
        namespace = {"__builtins__": __builtins__}
        try:
            exec(code, namespace)
            func = namespace["mog"]
        except Exception as e:
            if print_code:
                print(f"Error executing generated code: {e}")
            func = None
        return code, func

    def _get_function_latex(self, print_code=True, decimals=6, properties=None):
        """Build LaTeX string for the MoG PDF with parameters in \\begin{array}."""
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
        Build a table of MoG parameters: weights, means (mu_i), standard deviations (sigma_i),
        and correlations (rho_ij, i < j). Diagonal rho_ii are 1 by definition and omitted.
        Ex. df = MoG.tabulate(sort_by='weight')

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
            If a dict (e.g. from MultiPlot), its keys are used (mu_x, sigma_x, ...).
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
