import numpy as np
# from matplotlib import pyplot as plt

# from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

# import math
import string

# from time import time, strftime
# from scipy.stats import multivariate_normal as multinorm
# from scipy.optimize import minimize
import pickle
from hashlib import md5
import os

from .util import Util, Stats
# from .plot import CornerPlot, multimin_watermark


class ComposedMultiVariateNormal(object):
    """
    A linear combination of multivariate normal distribution (MND) with special methods
    for specifying the parameters of the distributions.

    Attributes
    ----------
    Ngauss : int
        Number of composed MND.
    Nvars : int
        Number of random variables.
    mus : numpy.ndarray
        Array with average (mu) of random variables (Ngauss x Nvars).
    weights : numpy.ndarray
        Array with weights of each MND (Ngauss).
        NOTE: These weights are normalized at the end.
    sigmas : numpy.ndarray
        Standard deviation of each variable(Ngauss x Nvars).
    rhos : numpy.ndarray
        Elements of the upper triangle of the correlation matrix (Ngauss x Nvars x (Nvars-1)/2).
    Sigmas : numpy.ndarray
        Array with covariance matrices for each MND (Ngauss x Nvars x Nvars).
    params : numpy.ndarray
        Parameters of the distribution in flatten form including symmetric elements of the covariance
        matrix (Ngauss*(1+Nvars+Nvars*(Nvars+1)/2)).
    stdcorr : numpy.ndarray
        Parameters of the distribution in flatten form, including upper triangle of the correlation
        matrix (Ngauss*(1+Nvars+Nvars*(Nvars+1)/2)).

    Notes
    -----
    There are several ways of initialize a CMND:

    1. Providing: Ngauss and Nvars
       In this case the class is instantiated with zero means, unitary dispersion and
       covariance matrix equal to Ngasus identity matrices Nvars x Nvars.

    2. Providing: params, Nvars
       In this case you have a flatted version of the parametes (weights, mus, Sigmas)
       and want to instantiate the system.  All parameters are set and no other action
       is required.

    3. Providing: stdcorr, Nvars
       In this case you have a flatted version of the parametes (weights, mus, sigmas, rhos)
       and want to instantiate the system.  All parameters are set and no other action
       is required.

    4. Providing: weights, mus, Sigmas (optional)
       In this case the basic properties of the CMND are set.

    Examples
    --------
    >>> mus = [[0, 0], [1, 1]]
    >>> weights = [0.1, 0]
    >>> MND1 = ComposedMultiVariateNormal(mus=mus, weights=weights)
    >>> MND1.set_sigmas([[[1, 0.2], [0, 1]], [[1, 0], [0, 1]]])
    >>> print(MND1)

    >>> params = [0.1, 0.9, 0, 0, 1, 1, 1, 0.2, 0.2, 1, 1, 0, 0, 1]
    >>> MND2 = ComposedMultiVariateNormal(params=params, Nvars=2)
    >>> print(MND2)
    """

    # Control behavior
    _ignoreWarnings = True

    def __init__(
        self,
        Ngauss=0,
        Nvars=0,
        params=None,
        stdcorr=None,
        weights=None,
        mus=None,
        Sigmas=None,
    ):

        # Method 1: initialize a simple instance
        if Ngauss > 0:
            mus = [[0] * Nvars] * Ngauss
            weights = [1 / Ngauss] * Ngauss
            Sigmas = [np.eye(Nvars)] * Ngauss
            self.__init__(mus=mus, weights=weights, Sigmas=Sigmas)

        # Method 2: initialize from flatten parameters
        elif params is not None:
            self.set_params(params, Nvars)

        # Method 3: initialize from flatten parameters
        elif stdcorr is not None:
            self.set_stdcorr(stdcorr, Nvars)

        # Method 4: initialize from explicit arrays
        else:
            # Basic attributes
            mus = np.array(mus)
            try:
                mus[0, 0]
            except Exception as e:
                Util.error_msg(e, "Parameter 'mus' must be a matrix, eg. mus=[[0,0]]")
                raise
            self.mus = mus

            # Number of variables
            self.Ngauss = len(mus)
            self.Nvars = len(mus[0])

            # Weights and normalization
            if weights is None:
                self.weights = [1] + (self.Ngauss - 1) * [0]
            elif len(weights) != self.Ngauss:
                raise ValueError(
                    f"Length of weights array ({len(weights)}) must be equal to number of MND ({self.Ngauss})"
                )
            else:
                self._normalize_weights(weights)

            # Secondary attributes
            if Sigmas is None:
                self.Sigmas = None
                self.params = None
            else:
                self.set_sigmas(Sigmas)

        self._nerror = 0

    def set_sigmas(self, Sigmas):
        """
        Set the value of list of covariance matrices.

        After setting Sigmas it update params and stdcorr.

        Parameters
        ----------
        Sigmas : list or numpy.ndarray
            Array of covariance matrices.

        Attr. [HC]
        """
        self.Sigmas = np.array(Sigmas)
        self._check_sigmas()
        self._flatten_params()
        self._flatten_stdcorr()

    def set_params(self, params, Nvars):
        """
        Set the properties of the CMND from flatten params.

        After setting it generate flattend stdcorr and normalize weights.

        Parameters
        ----------
        params : list or numpy.ndarray
            Flattened parameters.
        Nvars : int
            Number of variables.

        Attr. [HC]
        """
        if Nvars == 0 or len(params) == 0:
            raise ValueError(
                f"When setting from flat params, Nvars ({Nvars}) cannot be zero"
            )
        self._unflatten_params(params, Nvars)
        self._normalize_weights(self.weights)
        return

    def set_stdcorr(self, stdcorr, Nvars):
        """
        Set the properties of the CMND from flatten stdcorr.

        After setting it generate flattened params and normalize weights.

        Parameters
        ----------
        stdcorr : list or numpy.ndarray
            Flattened standard deviations and correlations.
        Nvars : int
            Number of variables.

        Attr. [HC]
        """
        if Nvars == 0 or len(stdcorr) == 0:
            raise ValueError(
                f"When setting from flat params, Nvars ({Nvars}) cannot be zero"
            )
        self._unflatten_stdcorr(stdcorr, Nvars)
        self._normalize_weights(self.weights)
        return

    def _normalize_weights(self, weights):
        """
        Normalize weights in such a way that sum(weights)=1

        Attr. [HC]
        """
        self.weights = np.array(weights) / sum(np.array(weights))

    def _flatten_params(self):
        """
        Flatten params

        Attr. [HC]
        """
        self._check_params(self.Sigmas)

        # Flatten covariance matrix
        SF = [
            Stats.flatten_symmetric_matrix(self.Sigmas[i]).tolist()
            for i in range(self.Ngauss)
        ]
        self.params = np.concatenate(
            (self.weights.flatten(), self.mus.flatten(), list(itertools.chain(*SF)))
        )
        self.Npars = len(self.params)  # Ngauss*(1+Nvars+Nvar*(Nvars+1)/2)

    def _flatten_stdcorr(self):
        """
        Flatten stdcorr

        Attr. [HC]
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

    def _unflatten_params(self, params, Nvars):
        """
        Unflatten properties from params

        Attr. [HC]
        """

        self.params = np.array(params)
        self.Npars = len(self.params)

        factor = int(1 + Nvars + Nvars * (Nvars + 1) / 2)

        if (self.Npars % factor) != 0:
            raise AssertionError(
                f"The number of parameters {self.Npars} is incompatible with the provided number of variables ({Nvars})"
            )

        # Number of gaussian functions
        Ngauss = int(self.Npars / factor)

        # Get the weights
        i = 0
        weights = self.params[i:Ngauss]
        i += Ngauss

        # Get the mus
        mus = self.params[i : i + Ngauss * Nvars].reshape(Ngauss, Nvars)
        i += Ngauss * Nvars

        # Get the sigmas
        Nsym = int(Nvars * (Nvars + 1) / 2)
        Sigmas = np.zeros((Ngauss, Nvars, Nvars))
        [
            Stats.unflatten_symmetric_matrix(F, Sigmas[i])
            for i, F in enumerate(
                self.params[i : i + Ngauss * Nsym].reshape(Ngauss, Nsym)
            )
        ]

        # Normalize weights
        self._normalize_weights(weights)

        # Check Sigmas
        self.Nvars = Nvars
        self.Ngauss = Ngauss
        self.weights = weights
        self.mus = mus
        self.Sigmas = Sigmas
        self._check_sigmas()

        # Flatten correlations
        self._flatten_stdcorr()

    def _unflatten_stdcorr(self, stdcorr, Nvars):
        """
        Unflatten properties from stdcorr

        Attr. [HC]
        """

        self.stdcorr = np.array(stdcorr)
        self.Ncor = len(self.stdcorr)

        factor = int(1 + Nvars + Nvars * (Nvars + 1) / 2)

        if (self.Ncor % factor) != 0:
            raise AssertionError(
                f"The number of parameters {self.Ncor} is incompatible with the provided number of variables ({Nvars})"
            )

        # Number of gaussian functions
        Ngauss = int(self.Ncor / factor)

        # Get the weights
        i = 0
        weights = self.stdcorr[i:Ngauss]
        i += Ngauss

        # Get the mus
        mus = self.stdcorr[i : i + Ngauss * Nvars].reshape(Ngauss, Nvars)
        i += Ngauss * Nvars

        # Get the sigmas
        sigmas = self.stdcorr[i : i + Ngauss * Nvars].reshape(Ngauss, Nvars)
        i += Ngauss * Nvars

        # Get the rhos
        Noff = int(Nvars * (Nvars - 1) / 2)
        rhos = self.stdcorr[i : i + Ngauss * Noff].reshape(Ngauss, Noff)

        # Normalize weights
        self._normalize_weights(weights)

        # Set properties
        self.Nvars = Nvars
        self.Ngauss = Ngauss
        self.weights = weights
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

        Attr. [HC]
        """
        self._check_params(self.Sigmas)

        # Check matrix
        if len(self.Sigmas) != self.Ngauss:
            raise ValueError(
                f"You provided {len(self.Sigmas)} matrix, but Ngauss={self.Ngauss} are required"
            )

        elif self.Sigmas[0].shape != (self.Nvars, self.Nvars):
            raise ValueError(
                f"Matrices have wrong dimensions ({self.Sigmas[0].shape}). It should be {self.Nvars}x{self.Nvars}"
            )

        # Symmetrize
        for i in range(self.Ngauss):
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

        Attr. [HC]
        """
        if checkvar is None:
            raise AssertionError(
                "You must first set the parameters (Sigmas, mus, etc.)"
            )

    def pdf(self, X):
        """
        Compute the PDF.

        Parameters
        ----------
        X : numpy.ndarray
            Point in the Nvar-dimensional space (Nvar).

        Returns
        -------
        p : float
            PDF value at X.

        Attr. [HC]
        """
        self._check_params(self.params)
        self._nerror = 0
        value = 0

        from scipy.stats import multivariate_normal as multinorm

        for w, muvec, Sigma in zip(self.weights, self.mus, self.Sigmas):
            try:
                value += w * multinorm.pdf(X, muvec, Sigma)
            except Exception as error:
                if not self._ignoreWarnings:
                    print(
                        f"Error: {error}, params = {self.params.tolist()}, stdcorr = {self.params.tolist()}"
                    )
                    self._nerror += 1
                value += 0
        return value

    def rvs(self, Nsam=1):
        """
        Generate a random sample of points following this Multivariate distribution.

        Parameters
        ----------
        Nsam : int, optional
            Number of samples (default 1).

        Returns
        -------
        rs : numpy.ndarray
            Samples (Nsam x Nvars).

        Attr. [HC]
        """
        self._check_params(self.params)

        from scipy.stats import multivariate_normal as multinorm

        Xs = np.zeros((Nsam, self.Nvars))
        for i in range(Nsam):
            n = Stats.gen_index(self.weights)
            Xs[i] = multinorm.rvs(self.mus[n], self.Sigmas[n])
        return Xs

    def sample_cmnd_likelihood(
        self, uparams, data=None, pmap=None, tset="stdcorr", scales=[], verbose=0
    ):
        """
        Compute the negative value of the logarithm of the likelihood of a sample.

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

        Attr. [HC]
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
            self.set_params(params, self.Nvars)
        else:
            self.set_stdcorr(params, self.Nvars)

        if verbose >= 2:
            print("CMND:")
            print(self)

        # Compute PDF for each point in data and sum
        log_l = -np.log(self.pdf(data)).sum()

        if verbose >= 1:
            print(f"-log_l = {log_l:e}")

        return log_l

    def plot_sample(
        self,
        data=None,
        N=10000,
        props=None,
        ranges=None,
        figsize=2,
        sargs=dict(),
        hargs=None,
    ):
        """
        Plot a sample of the CMND.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to plot. If None it generate a sample.
        N : int, optional
            Number of points to generate the sample (default 10000).
        props : list, optional
            Array with the name of the properties. Ex. props=["x","y"].
        ranges : list, optional
            Array of ranges of the properties. Ex. ranges=[-3,3],[-5,5].
        figsize : int, optional
            Size of each axis (default 2).
        sargs : dict, optional
            Dictionary with options for the scatter plot. Ex. sargs=dict(color='r').
        hargs : dict, optional
            Dictionary with options for the hist2d function. Ex. hargs=dict(bins=50).

        Returns
        -------
        G : matplotlib.figure.Figure or CornerPlot
            Graphic handle. If Nvars = 2, it is a figure object, otherwise is a CornerPlot instance.

        Examples
        --------
        >>> G = CMND.plot_sample(N=10000, sargs=dict(s=1, c='r'))
        >>> G = CMND.plot_sample(N=1000, sargs=dict(s=1, c='r'), hargs=dict(bins=20))

        >>> CMND = ComposedMultiVariateNormal(Ngauss=1, Nvars=2)
        >>> fig = CMND.plot_sample(N=1000, hargs=dict(bins=20), sargs=dict(s=1, c='r'))

        >>> CMND = ComposedMultiVariateNormal(Ngauss=2, Nvars=3)
        >>> print(CMND)
        >>> mus = [[0, 0], [1, 1]]
        >>> weights = [0.1, 0.9]
        >>> Sigmas = [[[1, 0.2], [0, 1]], [[1, 0], [0, 1]]]
        >>> MND1 = ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
        >>> #MND1=ComposedMultiVariateNormal(mus=mus,weights=weights);MND1.set_sigmas(Sigmas)
        >>> print(MND1)
        >>> print(MND1.pdf([1, 1]))
        >>> params = [0.1, 0.9, 0.0, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.0, 1.0]
        >>> MND2 = ComposedMultiVariateNormal(params=params, Nvars=2)
        >>> print(MND2)
        >>> print(MND2.pdf([1, 1]))

        Attr. [HC]
        """
        if data is None:
            self.data = self.rvs(N)
        else:
            self.data = np.copy(data)

        properties = dict()
        for i in range(self.Nvars):
            symbol = string.ascii_letters[i] if props is None else props[i]
            rang = None if ranges is None else ranges[i]
            properties[symbol] = dict(label=f"${symbol}$", range=rang)

        from matplotlib import pyplot as plt
        from .plot import CornerPlot, multimin_watermark

        if self.Nvars > 2:
            G = CornerPlot(properties, figsize=figsize)
            if hargs is not None:
                G.plot_hist(self.data, **hargs)
            G.scatter_plot(self.data, **sargs)
            G.fig.tight_layout()
            return G
        else:
            keys = list(properties.keys())
            fig = plt.figure(figsize=(5, 5))
            ax = fig.gca()
            if hargs is not None:
                ax.hist2d(self.data[:, 0], self.data[:, 1], **hargs)
            # Experimental
            # sns.kdeplot(x=data[:,0],y=data[:,1],shade=True,ax=ax)
            ax.scatter(self.data[:, 0], self.data[:, 1], **sargs)
            ax.set_xlabel(properties[keys[0]]["label"])
            ax.set_ylabel(properties[keys[1]]["label"])
            multimin_watermark(ax)
            fig.tight_layout()
            return fig

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
        for n in range(self.Ngauss):
            str_params += f"p{n + 1},"
            bnd_stdcorr += f"1,"

        # Mus
        for n in range(self.Ngauss):
            for i in range(self.Nvars):
                str_params += f"μ{n + 1}_{i + 1},"
                bnd_stdcorr += f"0,"

        str_stdcorr = str_params
        # Std. devs
        for n in range(self.Ngauss):
            for i in range(self.Nvars):
                str_stdcorr += f"σ{n + 1}_{i + 1},"
                bnd_stdcorr += f"10,"

        # Sigmas
        for n in range(self.Ngauss):
            for i in range(self.Nvars):
                for j in range(self.Nvars):
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

        msg = f"""Composition of Ngauss = {self.Ngauss} gaussian multivariates of Nvars = {self.Nvars} random variables:
    Weights: {self.weights.tolist()}
    Number of variables: {self.Nvars}
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


class FitCMND:
    """
    CMND Fitting handler.

    Attributes
    ----------
    Ngauss : int
        Number of fitting MND.
    Nvars : int
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
    >>> np.random.seed(1)
    >>> weights = [[1.0]]
    >>> mus = [[1.0, 0.5, -0.5]]
    >>> sigmas = [[1, 1.2, 2.3]]
    >>> angles = [[10*Angle.Deg, 30*Angle.Deg, 20*Angle.Deg]]
    >>> Sigmas = Stats.calc_covariance_from_rotation(sigmas, angles)
    >>> CMND = ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
    >>> data = CMND.rvs(10000)
    >>> F = FitCMND(Ngauss=CMND.Ngauss, Nvars=CMND.Nvars)
    >>> F.cmnd._fevfreq = 200
    >>> bounds = None
    >>> # bounds = F.set_bounds(boundw=(0.1, 0.9))
    >>> # bounds = F.set_bounds(boundr=(-0.9, 0.9))
    >>> # bounds = F.set_bounds(bounds=(0.1, 0.9*F._sigmax))
    >>> # bounds = F.set_bounds(boundsm=((-3, 3), (-2, 2), (-2, 2)), boundw=(0.1, 0.9), bounds=(0.1, 0.9*F._sigmax), boundr=(-0.9, 0.9))
    >>> print(bounds)
    >>> Util.el_time(0)
    >>> # F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True), bounds=bounds)
    >>> F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True), method=None, bounds=bounds)
    >>> T = Util.el_time()
    >>> print(F.cmnd)
    >>> G = F.plot_fit(figsize=3, hargs=dict(bins=30, cmap='YlGn'), sargs=dict(s=0.5, edgecolor='None', color='r'))
    >>> F.save_fit("/tmp/fit.pkl", useprefix=False)
    >>> F._load_fit("/tmp/fit.pkl")
    >>> F.save_fit("/tmp/nuevo.pkl", useprefix=True, myprefix="test")

    """

    # Constants
    # Maximum value of sigma
    _sigmax = 10
    _ignoreWarnings = True

    def __init__(self, objfile=None, Ngauss=1, Nvars=2):
        """
        Initialize FitCMND object.

        Attribution
        -----------
        [HC] This method was mostly developed by human intelligences.
        """

        if objfile is not None:
            self._load_fit(objfile)
        else:
            # Basic attributes
            self.Ngauss = Ngauss
            self.Nvars = Nvars
            self.Ndim = Ngauss * Nvars
            self.Ncorr = int(Nvars * (Nvars - 1) / 2)

            # Define the model cmnds
            self.cmnd = ComposedMultiVariateNormal(Ngauss=Ngauss, Nvars=Nvars)

            # Set parameters
            self.set_params()

        # Other
        self.fig = None
        self.prefix = ""

    def set_params(self, mu=0.5, sigma=1.0, rho=0.5):
        """
        Set the value of the basic params (minparams, scales, etc.).

        It updates minparams, scales and uprams.

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
            + [1 + rho] * self.Ngauss * self.Ncorr
        )
        scales = (
            [0] * self.Ndim
            + [self._sigmax] * self.Ndim
            + [2] * self.Ngauss * self.Ncorr
        )
        if self.Ngauss > 1:
            self.extrap = []
            minparams = [1 / self.Ngauss] * self.Ngauss + minparams
            scales = [1] * self.Ngauss + scales
        else:
            self.extrap = [1]

        self.minparams = np.array(minparams)
        self.scales = np.array(scales)
        self.uparams = Util.t_if(self.minparams, self.scales, Util.f2u)

    def pmap(self, minparams):
        """
        Mapping routine used in sample_cmnd_likelihood.

        Mapping may change depending on the complexity of the parameters to be minimized.
        Here we assume that all parameters in the stdcorr vector is susceptible to be minimized
        (with the exception of weights in the case of Ngauss=1 when this parameter should not
        be included).

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
        stdcorr[-self.Ngauss * self.Ncorr :] -= 1
        return stdcorr

    def log_l(self, data):
        """
        Value of the -log(Likelihood).

        Parameters
        ----------
        data : numpy.ndarray
            Array with data (Nsam x Nvars).

        Returns
        -------
        log_l : float
            Value of the -log(Likelihood).
        """

        log_l = self.cmnd.sample_cmnd_likelihood(
            self.uparams, data=data, pmap=self.pmap, tset="stdcorr", scales=self.scales
        )
        return log_l

    def fit_data(self, data, verbose=0, advance=0, **args):
        """
        Minimization procedure.

        It updates the solution attribute.

        Parameters
        ----------
        data : numpy.ndarray
            Array with data (Nsam x Nvars).
        verbose : int, optional
            Verbosity level for the sample_cmnd_likelihood routine (default 0).
        advance : int, optional
            If larger than 0 show advance each "advance" iterations (default 0).
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
        """
        if advance:
            advance = int(advance)
            self.neval = 0

            def _advance(X, show=False):
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
        self.cmnd._ignoreWarnings = self._ignoreWarnings
        self.minargs = dict(method="Powell")
        self.minargs.update(args)
        from scipy.optimize import minimize

        self.solution = minimize(
            self.cmnd.sample_cmnd_likelihood,
            self.minparams,
            callback=_advance,
            args=(data, self.pmap, "stdcorr", self.scales, verbose),
            **self.minargs,
        )
        if advance:
            _advance(self.solution.x, show=True)
        self.uparams = self.solution.x

        # Set the new params
        # Set the new params
        self.minparams = Util.t_if(self.uparams, self.scales, Util.u2f)
        params = self.pmap(self.minparams)
        self.cmnd.set_stdcorr(params, self.Nvars)
        self._update_prefix()

    def _load_fit(self, objfile):
        """
        Load a fit object from file.

        Attribution
        -----------
        [HC] This method was mostly developed by human intelligences.
        """
        F = pickle.load(open(objfile, "rb"))
        for k in F.__dict__.keys():
            setattr(self, k, getattr(F, k))
        self._update_prefix()

    def plot_fit(
        self, N=10000, figsize=2, props=None, ranges=None, hargs=dict(), sargs=dict()
    ):
        """
        Plot the result of the fitting procedure.

        Parameters
        ----------
        N : int, optional
            number of points used to build a representation of the marginal distributions (default 10000).
        figsize : int, optional
            Size of each axis (default 2).
        props : list, optional
            Array with the name of the properties. Ex. props=["x","y"].
        ranges : list, optional
            Array of ranges of the properties. Ex. ranges=[-3,3],[-5,5].
        hargs : dict, optional
            Dictionary with options for the hist2d function. Ex. hargs=dict(bins=50).
        sargs : dict, optional
            Dictionary with options for the scatter plot. Ex. sargs=dict(color='r').

        Returns
        -------
        G : matplotlib.figure.Figure or CornerPlot
            Graphic handle.

        Examples
        --------
        >>> F = FitCMND(1, 3)
        >>> F.fit_data(data, verbose=0, tol=1e-3, options=dict(maxiter=100, disp=True))
        >>> G = F.plot_fit(figsize=3, hargs=dict(bins=30, cmap='YlGn'), sargs=dict(s=0.5, edgecolor='None', color='r'))
        """
        Xfits = self.cmnd.rvs(N)
        properties = dict()
        for i in range(self.Nvars):
            symbol = string.ascii_letters[i] if props is None else props[i]
            if ranges is not None:
                rang = ranges[i]
            else:
                rang = None
            properties[symbol] = dict(label=f"${symbol}$", range=rang)
            properties[symbol] = dict(label=f"${symbol}$", range=rang)

        from matplotlib import pyplot as plt
        from .plot import CornerPlot, multimin_watermark

        if self.Nvars > 2:
            G = CornerPlot(properties, figsize=figsize)
            G.plot_hist(Xfits, **hargs)
            G.scatter_plot(self.data, **sargs)
            G.fig.tight_layout()
            self.fig = G.fig
            return G
        else:
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

        Attribution
        -----------
        [HC] This method was mostly developed by human intelligences.
        """
        minparams = np.copy(stdcorr)
        minparams[-self.Ngauss * self.Ncorr :] += 1
        self.minparams = minparams[1:] if self.Ngauss == 1 else minparams

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
        >>> self.hash = md5(pickle.dumps([self.Ngauss, self.data])).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.__dict__)).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.minparams)).hexdigest()[:5]
        >>> self.hash = md5(pickle.dumps(self.cmnd)).hexdigest()[:5]
        """
        self.hash = md5(pickle.dumps(self)).hexdigest()[:5]
        if myprefix is not None:
            myprefix = f"_{myprefix}"
        self.prefix = f"{self.Ngauss}cmnd{myprefix}_{self.hash}"

    def save_fit(self, objfile=None, useprefix=True, myprefix=None):
        """
        Pickle the result of a fit.

        Parameters
        ----------
        objfile : str, optional
            Name of the file where the fit will be stored. If None, the name is set
            by the routine as FitCMND.pkl.
        useprefix : bool, optional
            Use a prefix in the filename of the pickle file (default True).
            The prefix is normally {Ngauss}cmnd_{hash}.
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
        pickle.dump(self, open(objfile, "wb"))

    def set_bounds(self, boundw=None, bounds=None, boundr=None, boundsm=None):
        """
        Set the minimization parameters.

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
            Example for Nvars = 2:
            boundsm = ((-2, 1), (-3, 0))

        Returns
        -------
        bounds : tuple
            Formatted bounds for minimization.
        """
        if boundsm is None:
            boundsm = ((-np.inf, np.inf),) * self.Nvars

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
            *((boundw,) * self.Ngauss),
            *(boundsm * self.Ngauss),
            *((bounds,) * self.Nvars * self.Ngauss),
            *((boundr,) * self.Ngauss * int(self.Nvars * (self.Nvars - 1) / 2)),
        )
        self.bounds = bounds

        if self.Ngauss == 1:
            bounds = bounds[1:]
        self.cmnd._str_params()
        return bounds
