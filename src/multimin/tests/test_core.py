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
        G = mn.CornerPlot(properties, figsize=3)

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
