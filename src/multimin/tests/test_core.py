import os
import numpy as np
import multimin as mn
import matplotlib
import pytest
import warnings

# Set non-interactive backend to avoid opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Test gallery path
GALLERY_DIR = os.path.expanduser("~/.fargopy/test_gallery")

# Global constant for degree conversion
deg = np.pi / 180


@pytest.fixture(scope="module")
def setup_gallery():
    """Create the test gallery directory if it doesn't exist."""
    os.makedirs(GALLERY_DIR, exist_ok=True)
    yield
    # No cleanup intended, allowing manual inspection


@pytest.fixture
def cmnd_params():
    """Return common parameters for CMND setup used across tests."""
    weights = [0.5, 0.5]
    mus = np.array([[1.0, 0.5, -0.5], [1.0, -0.5, +0.5]])
    sigmas = [[1, 1.2, 2.3], [0.8, 0.2, 3.3]]
    angles = [
        [10 * deg, 30 * deg, 20 * deg],
        [-20 * deg, 0 * deg, 30 * deg],
    ]
    return weights, mus, sigmas, angles


def test_cmnd_evaluation(cmnd_params):
    """Test calculation of the CMND function value."""
    weights, mus, sigmas, angles = cmnd_params

    # Verify Stats module availability and function
    assert hasattr(mn, "Stats"), "Stats module not found in multimin"
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)

    # Initialize CMND
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)

    # Evaluate PDF at one of the mean positions
    val_mean = CMND.pdf(mus[0])
    assert val_mean > 0, "PDF value at mean should be positive"
    assert np.isfinite(val_mean), "PDF value should be finite"

    # Evaluate PDF at origin
    val_origin = CMND.pdf([0, 0, 0])
    assert val_origin > 0, "PDF value at origin should be positive"
    assert np.isfinite(val_origin), "PDF value should be finite"


def test_sample_generation(cmnd_params):
    """Test random sample generation and verify basic statistics."""
    np.random.seed(1)
    weights, mus, sigmas, angles = cmnd_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)

    n_samples = 5000
    data = CMND.rvs(n_samples)

    # Check shape
    assert data.shape == (n_samples, 3), f"Data shape mismatch: {data.shape}"

    # Check mean
    # Expected mean is weighted sum of component means
    # mus[0] = [1.0, 0.5, -0.5], mus[1] = [1.0, -0.5, 0.5]
    # weights = [0.5, 0.5]
    # Expected = [1.0, 0.0, 0.0]
    sample_mean = np.mean(data, axis=0)
    expected_mean = np.array([1.0, 0.0, 0.0])

    # Tolerance for stochastic process
    np.testing.assert_allclose(
        sample_mean, expected_mean, atol=0.2, err_msg="Sample mean deviation too large"
    )


def test_plotting(setup_gallery, cmnd_params):
    """Test generation of graphics saving to gallery without opening windows."""
    np.random.seed(1)
    weights, mus, sigmas, angles = cmnd_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
    data = CMND.rvs(1000)

    properties = dict(
        x=dict(label=r"$x$", range=None),
        y=dict(label=r"$y$", range=None),
        z=dict(label=r"$z$", range=None),
    )

    # Initialize plot
    # Using figsize=3 from notebook. Suppress tight_layout warning potentially triggered here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The figure layout has changed to tight"
        )
        G = mn.DensityPlot(properties, figsize=3)

    sargs = dict(s=1.2, edgecolor="None", color="r")

    # Generate scatter plot
    hist = G.scatter_plot(data, **sargs)
    assert hist is not None, "Scatter plot returned None"

    # Save figure
    output_filename = "multimin_core_test_plot.png"
    output_path = os.path.join(GALLERY_DIR, output_filename)

    # Ensure clean state
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        # Saving might also trigger layout warnings depending on backend behavior
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The figure layout has changed to tight"
            )
            plt.savefig(output_path)
    finally:
        plt.close("all")

    # Verification
    assert os.path.exists(output_path), f"Plot file {output_path} was not created"
    assert os.path.getsize(output_path) > 0, f"Plot file {output_path} is empty"


def test_cmnd_univariate():
    """Test ComposedMultiVariateNormal with nvars=1 (univariate Gaussians)."""
    # 2 univariate Gaussians: means 0 and 2, weights 0.5/0.5, variances 1 and 0.25
    mus = np.array([[0.0], [2.0]])
    weights = [0.5, 0.5]
    Sigmas = np.array([[[1.0]], [[0.25]]])  # std 1 and 0.5
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)

    assert CMND.nvars == 1 and CMND.ngauss == 2

    # PDF at mean (scalar and array)
    p0 = CMND.pdf(0.0)
    p0_arr = CMND.pdf([0.0])
    assert p0 > 0 and np.isfinite(p0)
    assert np.isclose(p0, p0_arr)

    p2 = CMND.pdf(2.0)
    assert p2 > 0 and np.isfinite(p2)

    # RVS shape (Nsam x 1)
    np.random.seed(42)
    data = CMND.rvs(500)
    assert data.shape == (500, 1)
    sample_mean = data[:, 0].mean()
    assert 0.5 < sample_mean < 1.5, "Sample mean should be near 1.0 (0.5*0 + 0.5*2)"

    # Init from ngauss/nvars
    CMND2 = mn.ComposedMultiVariateNormal(ngauss=2, nvars=1)
    assert CMND2.nvars == 1 and CMND2.ngauss == 2
    CMND2.set_sigmas([[[1.0]], [[1.0]]])
    assert CMND2.params is not None and len(CMND2.params) == 2 * (
        1 + 1 + 1
    )  # 2*(w+mu+var)

    # Params roundtrip
    p = CMND.params.copy()
    CMND3 = mn.ComposedMultiVariateNormal(params=p, nvars=1)
    np.testing.assert_allclose(CMND3.weights, CMND.weights)
    np.testing.assert_allclose(CMND3.mus, CMND.mus)
    np.testing.assert_allclose(CMND3.Sigmas, CMND.Sigmas)

    # Univariate 1-D API: mus and Sigmas as 1-D arrays
    mus_1d = np.array([0.0, 2.0])
    Sigmas_1d = np.array([1.0, 0.25])  # variances
    CMND_1d = mn.ComposedMultiVariateNormal(
        mus=mus_1d, weights=[0.5, 0.5], Sigmas=Sigmas_1d
    )
    assert CMND_1d.nvars == 1 and CMND_1d.ngauss == 2
    np.testing.assert_allclose(CMND_1d.mus.flatten(), mus_1d)
    np.testing.assert_allclose(CMND_1d.Sigmas[:, 0, 0], Sigmas_1d)
    np.testing.assert_allclose(CMND_1d.pdf(0.0), CMND.pdf(0.0))


def test_cmnd_univariate_fit():
    """Test FitCMND with ngauss=1, nvars=1 (univariate single Gaussian)."""
    np.random.seed(1)
    true_mu = 0.0
    true_var = 1.5
    CMND = mn.ComposedMultiVariateNormal(
        mus=np.array([true_mu]), weights=[1.0], Sigmas=np.array([true_var])
    )
    sample = CMND.rvs(500)
    F = mn.FitCMND(ngauss=1, nvars=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(sample, advance=False)
    assert np.all(np.isfinite(F.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F.log_l(sample)), "Log likelihood should be finite"
    fitted_mu = F.cmnd.mus.ravel()[0]
    fitted_var = F.cmnd.sigmas.ravel()[0] ** 2
    np.testing.assert_allclose(fitted_mu, true_mu, atol=0.3)
    np.testing.assert_allclose(fitted_var, true_var, atol=0.4)


def test_cmnd_univariate_fit_mixture():
    """Test FitCMND with ngauss=2, nvars=1 (univariate mixture of two Gaussians)."""
    np.random.seed(42)
    # Two univariate Gaussians: means 0 and 3, weights 0.5/0.5, variances 1 and 0.5
    mus = np.array([0.0, 3.0])
    weights = [0.5, 0.5]
    Sigmas = np.array([1.0, 0.5])  # variances
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
    sample = CMND.rvs(800)
    F = mn.FitCMND(ngauss=2, nvars=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(sample, advance=False)
    assert np.all(np.isfinite(F.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F.log_l(sample)), "Log likelihood should be finite"
    # Fitted weights (order may swap)
    fitted_weights = np.sort(F.cmnd.weights)
    np.testing.assert_allclose(fitted_weights, [0.5, 0.5], atol=0.15)
    # Fitted means and variances (match to true in some order)
    fitted_mus = np.sort(F.cmnd.mus.ravel())
    fitted_vars = np.sort(F.cmnd.sigmas.ravel() ** 2)
    np.testing.assert_allclose(fitted_mus, [0.0, 3.0], atol=0.4)
    np.testing.assert_allclose(fitted_vars, [0.5, 1.0], atol=0.35)


def test_cmnd_fitting(cmnd_params):
    """Test fitting a CMND with 2 Gaussians to generated data."""
    np.random.seed(1)
    weights, mus, sigmas, angles = cmnd_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    # Correctly use weights based on updated signature
    CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)

    # Generate data
    n_samples = 2000  # Smaller sample for faster test
    data = CMND.rvs(n_samples)

    # Initialize fitter with correct number of gaussians (2)
    F = mn.FitCMND(ngauss=2, nvars=3)

    # Run fit
    # Suppress divide by zero warning that may occur during optimization steps
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(data, advance=True)

    # Basic verification
    # Log likelihood should be reasonable (not infinite or NaN)
    # Also suppress warning for this check just in case
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        val_log_l = F.log_l(data)

    assert np.isfinite(val_log_l), "Log likelihood should be finite"

    # Check if fitted weights are roughly [0.5, 0.5]
    fitted_weights = F.cmnd.weights
    # Sort to handle component swapping
    fitted_weights = np.sort(fitted_weights)
    expected_weights = np.sort(weights)

    # Tolerance 0.2 because fit might not be perfect with 2000 samples and random initialization
    np.testing.assert_allclose(
        fitted_weights,
        expected_weights,
        atol=0.2,
        err_msg="Fitted weights deviation too large",
    )


def test_truncated_1d_fit():
    """Test FitCMND with domain=[[0, 1]] (1D truncated), inspired by multimin_truncated_tutorial."""
    np.random.seed(42)
    # Two Gaussians on [0, 1]: means 0.2 and 0.8, equal weights, small variance
    CMND_1d = mn.ComposedMultiVariateNormal(
        mus=[0.2, 0.8],
        weights=[0.5, 0.5],
        Sigmas=[0.02, 0.02],
        domain=[[0, 1]],
    )
    data_1d = CMND_1d.rvs(5000)
    F_1d = mn.FitCMND(ngauss=2, nvars=1, domain=[[0, 1]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F_1d.fit_data(data_1d, advance=True)
    assert np.all(np.isfinite(F_1d.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F_1d.log_l(data_1d)), "Log likelihood should be finite"
    # Fitted means should be in [0, 1] and close to 0.2 and 0.8
    fitted_mus = np.sort(F_1d.cmnd.mus.ravel())
    assert 0 <= fitted_mus[0] <= 1 and 0 <= fitted_mus[1] <= 1
    np.testing.assert_allclose(fitted_mus, [0.2, 0.8], atol=0.12)
    fitted_weights = np.sort(F_1d.cmnd.weights)
    np.testing.assert_allclose(fitted_weights, [0.5, 0.5], atol=0.12)


def test_truncated_3d_fit():
    """Test FitCMND with domain=[None, [0, 1], None] (3D, one variable truncated), inspired by multimin_truncated_tutorial."""
    np.random.seed(123)
    weights = [0.5, 0.5]
    mus = [[0.0, 0.3, 0.0], [0.0, 0.7, 0.0]]
    sigmas = [[0.6, 0.15, 0.6], [0.6, 0.15, 0.6]]
    Sigmas = [np.diag(np.array(s) ** 2) for s in sigmas]
    CMND_3d = mn.ComposedMultiVariateNormal(
        mus=mus,
        weights=weights,
        Sigmas=Sigmas,
        domain=[None, [0, 1], None],
    )
    data_3d = CMND_3d.rvs(5000)
    F_3d = mn.FitCMND(ngauss=2, nvars=3, domain=[None, [0, 1], None])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F_3d.fit_data(data_3d, advance=True)
    assert np.all(np.isfinite(F_3d.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F_3d.log_l(data_3d)), "Log likelihood should be finite"
    # Bounded variable (index 1) means must lie in [0, 1]
    mus_y = F_3d.cmnd.mus[:, 1]
    assert np.all((mus_y >= 0) & (mus_y <= 1)), (
        "Fitted means for bounded variable must be in [0, 1]"
    )
    # Fitted means for y should be close to 0.3 and 0.7 (order may swap)


def test_univariate_function_fit():
    """
    Test FitFunctionCMND for univariate function fitting.

    Fits an exponential decay with several Gaussians, as in
    examples/multimin_functions_tutorial.ipynb (non-gaussian function section).
    """
    np.random.seed(42)
    xs = np.linspace(0, 10, 100)
    ys = np.exp(-xs)
    F = mn.FitFunctionCMND(data=(xs, ys), ngauss=3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero")
        F.fit_data(advance=5, tol=1e-6, options=dict(maxiter=500), verbose=False)
    assert F.normalization is not None, "fit_data should set normalization"
    out = F.quality_of_fit(verbose=False)
    assert np.isfinite(out["R2"]), "R² should be finite"
    assert out["R2"] > 0.8, (
        "Exponential approximated by 3 Gaussians should yield R² > 0.8"
    )
    assert F.X.size == 100, "Grid should have 100 points"
    fig = F.plot_fit(figsize=(4, 3))
    assert fig is not None, "plot_fit should return a figure"
    plt.close(fig)


def test_univariate_function_fit_multimodal_single_peak():
    """Regression: multimodal mode must work for a single detected peak.

    Previously, when only one peak was detected, the multimodal initializer used
    parameter offsets that assumed mixture weights were present, corrupting the
    sigma slot and producing NaN loss at the first evaluation.
    """
    np.random.seed(42)
    xs = np.linspace(600.0, 900.0, 1000)
    mu0 = 750.0
    sigma0 = 40.0
    ys = np.exp(-0.5 * ((xs - mu0) / sigma0) ** 2)

    F = mn.FitFunctionCMND(data=(xs, ys), ngauss=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero")
        F.fit_data(mode="multimodal", options=dict(maxiter=200), verbose=False)

    assert F.normalization is not None, "fit_data should set normalization"
    out = F.quality_of_fit(verbose=False)
    assert np.isfinite(out["R2"]), "R² should be finite"


def test_set_initial_params_syncs_cmnd_fitcmnd():
    """set_initial_params should keep F.cmnd in sync (FitCMND)."""
    F = mn.FitCMND(ngauss=1, nvars=2)
    mus = [1.0, -2.0]
    sigmas = [0.4, 0.8]
    rhos = [0.25]
    F.set_initial_params(mus=mus, sigmas=sigmas, rhos=rhos)

    np.testing.assert_allclose(F.cmnd.mus.ravel(), np.asarray(mus, dtype=float))
    np.testing.assert_allclose(
        F.cmnd.sigmas.ravel(), np.asarray(sigmas, dtype=float), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        F.cmnd.rhos.ravel(), np.asarray(rhos, dtype=float), rtol=0, atol=1e-12
    )

def test_cmnd_update_params_updates_mus_only():
    """update_params updates mus without altering Sigmas."""
    cmnd = mn.ComposedMultiVariateNormal(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    old_sigmas = cmnd.sigmas.copy()
    old_rhos = cmnd.rhos.copy()
    old_Sigmas = cmnd.Sigmas.copy()

    cmnd.update_params(mus=[2.0, 3.0])

    assert np.allclose(cmnd.mus, np.array([[2.0, 3.0], [2.0, 3.0]]))
    assert np.allclose(cmnd.sigmas, old_sigmas)
    assert np.allclose(cmnd.rhos, old_rhos)
    assert np.allclose(cmnd.Sigmas, old_Sigmas)

def test_cmnd_update_params_updates_sigmas_and_rhos():
    """update_params updates sigmas/rhos and recomputes covariance matrices."""
    cmnd = mn.ComposedMultiVariateNormal(
        weights=[0.2, 0.8],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    cmnd.update_params(sigmas=[2.0, 3.0], rhos=[0.25])
    assert np.allclose(cmnd.sigmas, np.array([[2.0, 3.0], [2.0, 3.0]]))
    assert np.allclose(cmnd.rhos, np.array([[0.25], [0.25]]))

    expected = mn.Stats.calc_covariance_from_correlations(cmnd.sigmas, cmnd.rhos)
    assert np.allclose(cmnd.Sigmas, expected)

def test_cmnd_update_params_broadcasting_and_shape_errors():
    """update_params enforces FitCMND-like shapes."""
    cmnd = mn.ComposedMultiVariateNormal(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    with pytest.raises(ValueError):
        cmnd.update_params(mus=[1.0])

    with pytest.raises(ValueError):
        cmnd.update_params(sigmas=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        cmnd.update_params(rhos=[0.1, 0.2])


def test_cmnd_update_params_updates_weights_and_normalizes():
    """update_params updates weights and normalizes when normalize_weights=True."""
    cmnd = mn.ComposedMultiVariateNormal(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    cmnd.update_params(weights=[2.0, 1.0])
    assert np.allclose(cmnd.weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert np.isclose(cmnd.weights.sum(), 1.0)


def test_cmnd_update_params_updates_weights_without_normalization():
    """update_params keeps weights scale when normalize_weights=False."""
    cmnd = mn.ComposedMultiVariateNormal(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
        normalize_weights=False,
    )

    cmnd.update_params(weights=[2.0, 1.0])
    assert np.allclose(cmnd.weights, np.array([2.0, 1.0]))


def test_cmnd_plot_pdf_univariate_runs():
    """plot_pdf should run for univariate CMND and return a DensityPlot."""
    cmnd = mn.ComposedMultiVariateNormal(
        mus=[0.0, 2.5],
        Sigmas=[1.0, 0.25],
        weights=[0.5, 0.5],
    )
    G = cmnd.plot_pdf(properties=["x"], figsize=2)
    assert hasattr(G, "fig")
    assert hasattr(G, "axs")


def test_cmnd_plot_pdf_bivariate_runs():
    """plot_pdf should run for 2D CMND and return a DensityPlot."""
    cmnd = mn.ComposedMultiVariateNormal(
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        weights=[0.5, 0.5],
    )
    props = dict(
        x=dict(label=r"$x$", range=[-3, 3]),
        y=dict(label=r"$y$", range=[-3, 3]),
    )
    G = cmnd.plot_pdf(properties=props, figsize=2, grid_size=30)
    assert hasattr(G, "fig")
    assert hasattr(G, "axs")


def test_set_initial_params_syncs_cmnd_fitfunctioncmnd():
    """set_initial_params should keep F.cmnd in sync (FitFunctionCMND)."""
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.exp(-xs)
    F = mn.FitFunctionCMND(data=(xs, ys), ngauss=1)
    mu0 = 0.3
    sigma0 = 0.2
    F.set_initial_params(mus=mu0, sigmas=sigma0)

    np.testing.assert_allclose(F.cmnd.mus.ravel(), np.asarray([mu0], dtype=float))
    np.testing.assert_allclose(
        F.cmnd.sigmas.ravel(), np.asarray([sigma0], dtype=float), rtol=0, atol=1e-12
    )


def test_save_load_fit():
    """Test saving and loading a FitCMND object."""
    import tempfile
    import shutil

    # Create simple data
    np.random.seed(42)
    mean = [2, 3]
    cov = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean, cov, size=100)

    # Create and fit
    F = mn.FitCMND(data, ngauss=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(data, verbose=0, options={"maxiter": 20})

    # Save to temp file
    tmp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(tmp_dir, "test_fit.pkl")
        F.save_fit(save_path, useprefix=False)

        # Load back
        F_loaded = mn.FitCMND(save_path)

        # Verify
        assert F_loaded.ngauss == F.ngauss
        assert F_loaded.nvars == F.nvars
        np.testing.assert_allclose(F_loaded.minparams, F.minparams)

        # Verify functionality
        val_orig = F.log_l(data)
        val_loaded = F_loaded.log_l(data)
        assert np.isclose(val_orig, val_loaded)

    finally:
        shutil.rmtree(tmp_dir)
