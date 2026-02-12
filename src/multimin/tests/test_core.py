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
def mog_params():
    """Return common parameters for MoG setup used across tests."""
    weights = [0.5, 0.5]
    mus = np.array([[1.0, 0.5, -0.5], [1.0, -0.5, +0.5]])
    sigmas = [[1, 1.2, 2.3], [0.8, 0.2, 3.3]]
    angles = [
        [10 * deg, 30 * deg, 20 * deg],
        [-20 * deg, 0 * deg, 30 * deg],
    ]
    return weights, mus, sigmas, angles


def test_mog_evaluation(mog_params):
    """Test calculation of the MoG function value."""
    weights, mus, sigmas, angles = mog_params

    # Verify Stats module availability and function
    assert hasattr(mn, "Stats"), "Stats module not found in multimin"
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)

    # Initialize MoG
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)

    # Evaluate PDF at one of the mean positions
    val_mean = MoG.pdf(mus[0])
    assert val_mean > 0, "PDF value at mean should be positive"
    assert np.isfinite(val_mean), "PDF value should be finite"

    # Evaluate PDF at origin
    val_origin = MoG.pdf([0, 0, 0])
    assert val_origin > 0, "PDF value at origin should be positive"
    assert np.isfinite(val_origin), "PDF value should be finite"


def test_sample_generation(mog_params):
    """Test random sample generation and verify basic statistics."""
    np.random.seed(1)
    weights, mus, sigmas, angles = mog_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)

    n_samples = 5000
    data = MoG.rvs(n_samples)

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


def test_plotting(setup_gallery, mog_params):
    """Test generation of graphics saving to gallery without opening windows."""
    np.random.seed(1)
    weights, mus, sigmas, angles = mog_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)
    data = MoG.rvs(1000)

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


def test_mog_univariate():
    """Test MixtureOfGaussians with nvars=1 (univariate Gaussians)."""
    # 2 univariate Gaussians: means 0 and 2, weights 0.5/0.5, variances 1 and 0.25
    mus = np.array([[0.0], [2.0]])
    weights = [0.5, 0.5]
    Sigmas = np.array([[[1.0]], [[0.25]]])  # std 1 and 0.5
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)

    assert MoG.nvars == 1 and MoG.ngauss == 2

    # PDF at mean (scalar and array)
    p0 = MoG.pdf(0.0)
    p0_arr = MoG.pdf([0.0])
    assert p0 > 0 and np.isfinite(p0)
    assert np.isclose(p0, p0_arr)

    p2 = MoG.pdf(2.0)
    assert p2 > 0 and np.isfinite(p2)

    # RVS shape (Nsam x 1)
    np.random.seed(42)
    data = MoG.rvs(500)
    assert data.shape == (500, 1)
    sample_mean = data[:, 0].mean()
    assert 0.5 < sample_mean < 1.5, "Sample mean should be near 1.0 (0.5*0 + 0.5*2)"

    # Init from ngauss/nvars
    MoG2 = mn.MixtureOfGaussians(ngauss=2, nvars=1)
    assert MoG2.nvars == 1 and MoG2.ngauss == 2
    MoG2.set_sigmas([[[1.0]], [[1.0]]])
    assert MoG2.params is not None and len(MoG2.params) == 2 * (
        1 + 1 + 1
    )  # 2*(w+mu+var)

    # Params roundtrip
    p = MoG.params.copy()
    MoG3 = mn.MixtureOfGaussians(params=p, nvars=1)
    np.testing.assert_allclose(MoG3.weights, MoG.weights)
    np.testing.assert_allclose(MoG3.mus, MoG.mus)
    np.testing.assert_allclose(MoG3.Sigmas, MoG.Sigmas)

    # Univariate 1-D API: mus and Sigmas as 1-D arrays
    mus_1d = np.array([0.0, 2.0])
    Sigmas_1d = np.array([1.0, 0.25])  # variances
    MoG_1d = mn.MixtureOfGaussians(mus=mus_1d, weights=[0.5, 0.5], Sigmas=Sigmas_1d)
    assert MoG_1d.nvars == 1 and MoG_1d.ngauss == 2
    np.testing.assert_allclose(MoG_1d.mus.flatten(), mus_1d)
    np.testing.assert_allclose(MoG_1d.Sigmas[:, 0, 0], Sigmas_1d)
    np.testing.assert_allclose(MoG_1d.pdf(0.0), MoG.pdf(0.0))


def test_mog_univariate_fit():
    """Test FitMoG with ngauss=1, nvars=1 (univariate single Gaussian)."""
    np.random.seed(1)
    true_mu = 0.0
    true_var = 1.5
    MoG = mn.MixtureOfGaussians(
        mus=np.array([true_mu]), weights=[1.0], Sigmas=np.array([true_var])
    )
    sample = MoG.rvs(500)
    F = mn.FitMoG(ngauss=1, nvars=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(sample, advance=False)
    assert np.all(np.isfinite(F.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F.log_l(sample)), "Log likelihood should be finite"
    fitted_mu = F.mog.mus.ravel()[0]
    fitted_var = F.mog.sigmas.ravel()[0] ** 2
    np.testing.assert_allclose(fitted_mu, true_mu, atol=0.3)
    np.testing.assert_allclose(fitted_var, true_var, atol=0.4)


def test_mog_univariate_fit_mixture():
    """Test FitMoG with ngauss=2, nvars=1 (univariate mixture of two Gaussians)."""
    np.random.seed(42)
    # Two univariate Gaussians: means 0 and 3, weights 0.5/0.5, variances 1 and 0.5
    mus = np.array([0.0, 3.0])
    weights = [0.5, 0.5]
    Sigmas = np.array([1.0, 0.5])  # variances
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)
    sample = MoG.rvs(800)
    F = mn.FitMoG(ngauss=2, nvars=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(sample, advance=False)
    assert np.all(np.isfinite(F.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F.log_l(sample)), "Log likelihood should be finite"
    # Fitted weights (order may swap)
    fitted_weights = np.sort(F.mog.weights)
    np.testing.assert_allclose(fitted_weights, [0.5, 0.5], atol=0.15)
    # Fitted means and variances (match to true in some order)
    fitted_mus = np.sort(F.mog.mus.ravel())
    fitted_vars = np.sort(F.mog.sigmas.ravel() ** 2)
    np.testing.assert_allclose(fitted_mus, [0.0, 3.0], atol=0.4)
    np.testing.assert_allclose(fitted_vars, [0.5, 1.0], atol=0.35)


def test_mog_fitting(mog_params):
    """Test fitting a MoG with 2 Gaussians to generated data."""
    np.random.seed(1)
    weights, mus, sigmas, angles = mog_params
    Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)
    # Correctly use weights based on updated signature
    MoG = mn.MixtureOfGaussians(mus=mus, weights=weights, Sigmas=Sigmas)

    # Generate data
    n_samples = 2000  # Smaller sample for faster test
    data = MoG.rvs(n_samples)

    # Initialize fitter with correct number of gaussians (2)
    F = mn.FitMoG(ngauss=2, nvars=3)

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
    fitted_weights = F.mog.weights
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
    """Test FitMoG with domain=[[0, 1]] (1D truncated), inspired by multimin_truncated_tutorial."""
    np.random.seed(42)
    # Two Gaussians on [0, 1]: means 0.2 and 0.8, equal weights, small variance
    MoG_1d = mn.MixtureOfGaussians(
        mus=[0.2, 0.8],
        weights=[0.5, 0.5],
        Sigmas=[0.02, 0.02],
        domain=[[0, 1]],
    )
    data_1d = MoG_1d.rvs(5000)
    F_1d = mn.FitMoG(ngauss=2, nvars=1, domain=[[0, 1]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F_1d.fit_data(data_1d, advance=True)
    assert np.all(np.isfinite(F_1d.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F_1d.log_l(data_1d)), "Log likelihood should be finite"
    # Fitted means should be in [0, 1] and close to 0.2 and 0.8
    fitted_mus = np.sort(F_1d.mog.mus.ravel())
    assert 0 <= fitted_mus[0] <= 1 and 0 <= fitted_mus[1] <= 1
    np.testing.assert_allclose(fitted_mus, [0.2, 0.8], atol=0.12)
    fitted_weights = np.sort(F_1d.mog.weights)
    np.testing.assert_allclose(fitted_weights, [0.5, 0.5], atol=0.12)


def test_truncated_3d_fit():
    """Test FitMoG with domain=[None, [0, 1], None] (3D, one variable truncated), inspired by multimin_truncated_tutorial."""
    np.random.seed(123)
    weights = [0.5, 0.5]
    mus = [[0.0, 0.3, 0.0], [0.0, 0.7, 0.0]]
    sigmas = [[0.6, 0.15, 0.6], [0.6, 0.15, 0.6]]
    Sigmas = [np.diag(np.array(s) ** 2) for s in sigmas]
    MoG_3d = mn.MixtureOfGaussians(
        mus=mus,
        weights=weights,
        Sigmas=Sigmas,
        domain=[None, [0, 1], None],
    )
    data_3d = MoG_3d.rvs(5000)
    F_3d = mn.FitMoG(ngauss=2, nvars=3, domain=[None, [0, 1], None])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F_3d.fit_data(data_3d, advance=True)
    assert np.all(np.isfinite(F_3d.solution.x)), "Optimizer solution should be finite"
    assert np.isfinite(F_3d.log_l(data_3d)), "Log likelihood should be finite"
    # Bounded variable (index 1) means must lie in [0, 1]
    mus_y = F_3d.mog.mus[:, 1]
    assert np.all((mus_y >= 0) & (mus_y <= 1)), (
        "Fitted means for bounded variable must be in [0, 1]"
    )
    # Fitted means for y should be close to 0.3 and 0.7 (order may swap)


def test_univariate_function_fit():
    """
    Test FitFunctionMoG for univariate function fitting.

    Fits an exponential decay with several Gaussians, as in
    examples/multimin_functions_tutorial.ipynb (non-gaussian function section).
    """
    np.random.seed(42)
    xs = np.linspace(0, 10, 100)
    ys = np.exp(-xs)
    F = mn.FitFunctionMoG(data=(xs, ys), ngauss=3)
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

    F = mn.FitFunctionMoG(data=(xs, ys), ngauss=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero")
        F.fit_data(mode="multimodal", options=dict(maxiter=200), verbose=False)

    assert F.normalization is not None, "fit_data should set normalization"
    out = F.quality_of_fit(verbose=False)
    assert np.isfinite(out["R2"]), "R² should be finite"


def test_set_initial_params_syncs_mog_fitmog():
    """set_initial_params should keep F.mog in sync (FitMoG)."""
    F = mn.FitMoG(ngauss=1, nvars=2)
    mus = [1.0, -2.0]
    sigmas = [0.4, 0.8]
    rhos = [0.25]
    F.set_initial_params(mus=mus, sigmas=sigmas, rhos=rhos)

    np.testing.assert_allclose(F.mog.mus.ravel(), np.asarray(mus, dtype=float))
    np.testing.assert_allclose(
        F.mog.sigmas.ravel(), np.asarray(sigmas, dtype=float), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        F.mog.rhos.ravel(), np.asarray(rhos, dtype=float), rtol=0, atol=1e-12
    )


def test_mog_update_params_updates_mus_only():
    """update_params updates mus without altering Sigmas."""
    mog = mn.MixtureOfGaussians(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    old_sigmas = mog.sigmas.copy()
    old_rhos = mog.rhos.copy()
    old_Sigmas = mog.Sigmas.copy()

    mog.update_params(mus=[2.0, 3.0])

    assert np.allclose(mog.mus, np.array([[2.0, 3.0], [2.0, 3.0]]))
    assert np.allclose(mog.sigmas, old_sigmas)
    assert np.allclose(mog.rhos, old_rhos)
    assert np.allclose(mog.Sigmas, old_Sigmas)


def test_mog_update_params_updates_sigmas_and_rhos():
    """update_params updates sigmas/rhos and recomputes covariance matrices."""
    mog = mn.MixtureOfGaussians(
        weights=[0.2, 0.8],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    mog.update_params(sigmas=[2.0, 3.0], rhos=[0.25])
    assert np.allclose(mog.sigmas, np.array([[2.0, 3.0], [2.0, 3.0]]))
    assert np.allclose(mog.rhos, np.array([[0.25], [0.25]]))

    expected = mn.Stats.calc_covariance_from_correlations(mog.sigmas, mog.rhos)
    assert np.allclose(mog.Sigmas, expected)


def test_mog_update_params_broadcasting_and_shape_errors():
    """update_params enforces FitMoG-like shapes."""
    mog = mn.MixtureOfGaussians(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    with pytest.raises(ValueError):
        mog.update_params(mus=[1.0])

    with pytest.raises(ValueError):
        mog.update_params(sigmas=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        mog.update_params(rhos=[0.1, 0.2])


def test_mog_update_params_updates_weights_and_normalizes():
    """update_params updates weights and normalizes when normalize_weights=True."""
    mog = mn.MixtureOfGaussians(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    mog.update_params(weights=[2.0, 1.0])
    assert np.allclose(mog.weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert np.isclose(mog.weights.sum(), 1.0)


def test_mog_update_params_updates_weights_without_normalization():
    """update_params keeps weights scale when normalize_weights=False."""
    mog = mn.MixtureOfGaussians(
        weights=[0.5, 0.5],
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        domain=[[-10.0, 10.0], [-10.0, 10.0]],
        normalize_weights=False,
    )

    mog.update_params(weights=[2.0, 1.0])
    assert np.allclose(mog.weights, np.array([2.0, 1.0]))


def test_mog_plot_pdf_univariate_runs():
    """plot_pdf should run for univariate MoG and return a DensityPlot."""
    mog = mn.MixtureOfGaussians(
        mus=[0.0, 2.5],
        Sigmas=[1.0, 0.25],
        weights=[0.5, 0.5],
    )
    G = mog.plot_pdf(properties=["x"], figsize=2)
    assert hasattr(G, "fig")
    assert hasattr(G, "axs")


def test_mog_plot_pdf_bivariate_runs():
    """plot_pdf should run for 2D MoG and return a DensityPlot."""
    mog = mn.MixtureOfGaussians(
        mus=[[0.0, 0.0], [1.0, 1.0]],
        Sigmas=[np.eye(2), np.eye(2)],
        weights=[0.5, 0.5],
    )
    props = dict(
        x=dict(label=r"$x$", range=[-3, 3]),
        y=dict(label=r"$y$", range=[-3, 3]),
    )
    G = mog.plot_pdf(properties=props, figsize=2, grid_size=30)
    assert hasattr(G, "fig")
    assert hasattr(G, "axs")


def test_set_initial_params_syncs_mog_fitfunctionmog():
    """set_initial_params should keep F.mog in sync (FitFunctionMoG)."""
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.exp(-xs)
    F = mn.FitFunctionMoG(data=(xs, ys), ngauss=1)
    mu0 = 0.3
    sigma0 = 0.2
    F.set_initial_params(mus=mu0, sigmas=sigma0)

    np.testing.assert_allclose(F.mog.mus.ravel(), np.asarray([mu0], dtype=float))
    np.testing.assert_allclose(
        F.mog.sigmas.ravel(), np.asarray([sigma0], dtype=float), rtol=0, atol=1e-12
    )


def test_save_load_fit():
    """Test saving and loading a FitMoG object."""
    import tempfile
    import shutil

    # Create simple data
    np.random.seed(42)
    mean = [2, 3]
    cov = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean, cov, size=100)

    # Create and fit
    F = mn.FitMoG(data, ngauss=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        F.fit_data(data, verbose=0, options={"maxiter": 20})

    # Save to temp file
    tmp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(tmp_dir, "test_fit.pkl")
        F.save_fit(save_path, useprefix=False)

        # Load back
        F_loaded = mn.FitMoG(save_path)

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
