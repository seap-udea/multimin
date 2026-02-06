import numpy as np
import math
from time import time
import spiceypy as spy
import os
from datetime import datetime
from multimin import ROOTDIR


class Util(object):
    """
    This abstract class contains useful methods for the package.

    Attr. [HC]
    """

    # Mathematical functions
    """
    #Interesting but it can be problematic
    sin=math.sin
    cos=math.cos
    log=math.log
    exp=math.exp
    """
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
        Attribution
        -----------
        [HC] This function was mostly developed by human intelligences.
        """
        return os.path.join(ROOTDIR, "data", filename)

    @staticmethod
    def f2u(x, s):
        """
        Convert from a finite interval [0,s] to an unbound one [-inf,inf].

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
        Attribution
        -----------
        [HC] This function was mostly developed by human intelligences.
        """
        return Util.log((x / s) / (1 - (x / s)))

    @staticmethod
    def u2f(t, s):
        """
        Convert from an unbound interval [-inf,inf] to a finite one [0,s].

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
        Attribution
        -----------
        [HC] This function was mostly developed by human intelligences.
        """
        return s / (1 + Util.exp(-t))

    @staticmethod
    def t_if(p, s, f):
        """
        Transform a set of parameters using a transformation function f and scales s.

        This routine allows the conversion from a finite interval [0,s] to an unbound one [-inf,inf]
        (using f=Util.f2u) or vice versa (using f=Util.u2f).

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
        Attribution
        -----------
        [HC] This function was mostly developed by human intelligences.
        """
        return [f(p[i], s[i]) if s[i] > 0 else p[i] for i in range(len(p))]

    @staticmethod
    def true_anomaly_to_mean_anomaly(e, f):
        """
        Convert true anomaly to mean anomaly.

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

        Notes
        -----
        This function:
        1. Calculates eccentric anomaly E from true anomaly f
        2. Uses Kepler's equation to calculate mean anomaly M from E and e

        The conversion uses the formula:
        - E = 2 * arctan(tan(f/2) / sqrt((1+e)/(1-e)))
        - M = E - e*sin(E) (Kepler's equation)
        """
        # Calculate eccentric anomaly E from true anomaly f
        E = 2 * np.arctan(np.tan(f / 2) / np.sqrt((1 + e) / (1 - e)))
        M = E - e * np.sin(E)

        return M

    def error_msg(error, msg):
        """
        Add a custom message msg to an error handle.

        Parameters
        ----------
        error : Exception
            Error handle (eg. except ValueError as error).
        msg : str
            Message to add to error.
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
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
        ms (miliseconds), s (seconds), min (minutes), h (hours), d (days)

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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
        """
        xa = np.abs(x)
        s = np.sign(x)
        try:
            exp = int(np.floor(np.log10(xa)))
            man = s * xa / 10 ** (exp)
        except OverflowError as e:
            man = exp = 0
        return man, exp


class Stats(object):
    """
    Abstract class with useful routines

    Attr. [HC]
    """

    # Golden ratio: required for golden gaussian.
    phi = (1 + 5**0.5) / 2

    def gen_index(probs):
        """
        Given a set of (normalized) probabilities, randomly generate an index n following the
        probabilities.

        For instance if we have 3 events with probabilities 0.1, 0.7, 0.2, gen_index will generate
        a number in the set (0,1,2) having those probabilities, ie. 1 will have 70% of probability.

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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
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
        Set a matrix with the terms of the off diagonal

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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
        """
        I, J = np.where(~np.eye(M.shape[0], dtype=bool))
        ffo = list(off[::-1])
        for i, j in zip(I, J):
            M[i, j] = ffo.pop() if j > i else 0
        M[:, :] = np.triu(M) + np.tril(M.T, -1)

    def calc_covariance_from_correlations(sigmas, rhos):
        """
        Compute covariance matrices from the standard deviations and correlations (rho).

        Parameters
        ----------
        sigmas : numpy.ndarray
            Array of values of standard deviation for variables (Ngauss x Nvars).
        rhos : numpy.ndarray
            Array with correlations (Ngauss x Nvars x (Nvars-1)/2).

        Returns
        -------
        Sigmas : numpy.ndarray
            Array with covariance matrices corresponding to these sigmas and rhos (Ngauss x Nvars x Nvars).

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

        Sources
        -------
        Based on: https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

        Attr. [HC]
        """
        try:
            Nvars = len(sigmas[0])
        except:
            raise AssertionError("Array of sigmas must be an array of arrays")
        try:
            Nrhos = len(rhos[0])
        except:
            raise AssertionError("Array of rhos must be an array of arrays")

        if Nrhos != int(Nvars * (Nvars - 1) / 2):
            raise AssertionError(
                f"Size of rhos ({Nrhos}) are incompatible with Nvars={Nvars}.  It should be Nvars(Nvars-1)/2={int(Nvars * (Nvars - 1) / 2)}."
            )

        Sigmas = np.array(len(sigmas) * [np.eye(Nvars)])
        for Sigma, sigma, rho in zip(Sigmas, sigmas, rhos):
            Stats.set_matrix_off_diagonal(Sigma, rho)
            Sigma *= np.outer(sigma, sigma)
        return Sigmas

    def calc_correlations_from_covariances(Sigmas):
        """
        Compute the standard deviations and corresponding correlation coefficients given a set of
        covariance matrices.

        Parameters
        ----------
        Sigmas : numpy.ndarray
            Array of covariance matrices (Ngauss x Nvars x Nvars).

        Returns
        -------
        sigmas : numpy.ndarray
            Array of standard deviations (Ngauss x Nvars).
        rhos : numpy.ndarray
            Array of correlation coefficients (Ngauss x Nvars * (Nvars-1) / 2).

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

        Attr. [HC]
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

        Parameters
        ----------
        sigmas : numpy.ndarray
            Array of values of standard deviation for variables (Ngauss x 3).
        angles : numpy.ndarray
            Euler angles expressing the directions of the principal axes of the distribution (Ngauss x 3).

        Returns
        -------
        Sigmas : numpy.ndarray
            Array with covariance matrices corresponding to these sigmas and angles (Ngauss x 3 x 3).
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
        """
        try:
            Nvars = len(sigmas[0])
        except:
            raise AssertionError("Sigmas must be an array of arrays")
        Sigmas = []
        for scale, angle in zip(sigmas, angles):
            L = np.identity(Nvars) * np.outer(np.ones(Nvars), scale)
            Rot = (
                spy.eul2m(-angle[0], -angle[1], -angle[2], 3, 1, 3)
                if Nvars == 3
                else spy.rotate(-angle[0], 3)[:2, :2]
            )
            Sigmas += [np.matmul(np.matmul(Rot, np.matmul(L, L)), np.linalg.inv(Rot))]

        return np.array(Sigmas)

    def flatten_symmetric_matrix(M):
        """
        Given a symmetric matrix the routine returns the flatten version of the Matrix.

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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
        """
        return M[np.triu_indices(M.shape[0], k=0)]

    def unflatten_symmetric_matrix(F, M):
        """
        Given a flatten version of a matrix, returns the symmetric matrix.

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
            Attribution
            -----------
            [HC] This class was mostly developed by human intelligences.
        """
        M[np.triu_indices(M.shape[0], k=0)] = np.array(F)
        M[:, :] = np.triu(M) + np.tril(M.T, -1)

    @staticmethod
    def download_cneos_fireballs(force_download=False):
        """
        Download CNEOS Fireball Database data in CSV format.

        This method downloads the CNEOS (Center for Near Earth Object Studies)
        Fireball Database data from NASA's JPL website. The data is saved in
        the package's data folder with a filename that includes the download date.

        Parameters
        ----------
        force_download : bool, default False
            If True, download the data even if a file for today already exists.
            If False, skip download if a file for today already exists.

        Returns
        -------
        str
            Full path to the downloaded CSV file. Returns None if download fails.

        Examples
        --------
        >>> from multimin.util import Util
        >>>
        >>> # Download CNEOS fireball data
        >>> csv_path = Util.download_cneos_fireballs()
        >>> print(f"Data saved to: {csv_path}")
        >>>
        >>> # Force download even if file exists
        >>> csv_path = Util.download_cneos_fireballs(force_download=True)

        Notes
        -----
        - The file is saved with format: `cneos_fireballs_YYYYMMDD.csv`
        - Requires the `requests` package to be installed.
        - The CNEOS website uses DataTables, and this method accesses the CSV
          export endpoint directly.
        - If a file for today already exists and `force_download=False`,
          the existing file path is returned.

        Attribution
        -----------
        [HC] This method was mostly developed by human intelligences.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "The 'requests' package is required for downloading CNEOS data. "
                "Install it with: pip install requests"
            )

        # CNEOS Fireball Database CSV export URL
        # The DataTable CSV export endpoint
        csv_url = "https://cneos.jpl.nasa.gov/fireballs/export.php"

        # Data directory
        data_dir = os.path.join(ROOTDIR, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Generate filename with current date
        today = datetime.now().strftime("%Y%m%d")
        filename = f"cneos_fireballs_{today}.csv"
        filepath = os.path.join(data_dir, filename)

        # Check if file already exists for today
        if os.path.exists(filepath) and not force_download:
            print(f"CNEOS fireball data for today already exists: {filepath}")
            return filepath

        try:
            # Set headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,application/csv,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }

            # Download the CSV data
            print(f"Downloading CNEOS fireball data from {csv_url}...")
            response = requests.get(csv_url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Check if we got CSV data
            content_type = response.headers.get("Content-Type", "").lower()
            if "csv" not in content_type and "text/plain" not in content_type:
                # Try to detect if it's CSV by checking first few bytes
                if not response.text.strip().startswith(("Date", "Peak", "Latitude")):
                    print("Warning: Response may not be in CSV format.")

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"Successfully downloaded CNEOS fireball data to: {filepath}")
            print(f"File size: {os.path.getsize(filepath)} bytes")

            return filepath

        except requests.exceptions.RequestException as e:
            print(f"Error downloading CNEOS fireball data: {e}")
            print("Please check your internet connection and try again.")
            return None
        except Exception as e:
            print(f"Unexpected error while downloading CNEOS fireball data: {e}")
            return None
