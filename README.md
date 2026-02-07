
<p></p>
<div align="center">
  <a href="https://multimin.readthedocs.io/">
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/docs/multimin-logo-white.webp" alt="MultiMin Logo" width="600"/>
  </a>
</div>
<p></p>

[![version](https://img.shields.io/pypi/v/multimin?color=blue)](https://pypi.org/project/multimin/)
[![license](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://github.com/seap-udea/multimin/blob/master/LICENSE)
[![pythonver](https://img.shields.io/pypi/pyversions/multimin)](https://pypi.org/project/multimin/)
[![downloads](https://img.shields.io/pypi/dw/multimin)](https://pypi.org/project/multimin/)
[![Powered by SciPy](https://img.shields.io/badge/Powered%20by-SciPy-blue)](https://scipy.org/)

<div align="center">
<p></p>
</div>

## Introducing `MultiMin`

`MultiMin` is a `Python` package designed to provide numerical tools for **fitting composed multivariate distributions** to data. It is particularly useful for modelling complex multimodal distributions in N-dimensions.

These are the main features of `MultiMin`:

- **Multivariate Fitting**: Tools for fitting composed multivariate normal distributions (CMND).
- **Visualization**: Corner plots and specific visualization utilities.
- **Statistical Analysis**: Tools for handling covariance matrices and correlations.

## Documentation

Full API documentation is available at [https://multimin.readthedocs.io](https://multimin.readthedocs.io).

## Installation

### From PyPI

`MultiMin` will be available on PyPI at https://pypi.org/project/multimin/. Once published, you can install it with:

```bash
pip install -U multimin
```

### From Sources

You can also install from the [GitHub repository](https://github.com/seap-udea/multimin):

```bash
git clone https://github.com/seap-udea/multimin
cd multimin
pip install .
```

For development, use an editable installation:

```bash
cd multimin
pip install -e .
```

### In Google Colab

If you use Google Colab, you can install `MultiMin` by executing:

```python
!pip install -U multimin
```

## Quick Start

Getting started with `MultiMin` is straightforward. Import the package:

```python
import multimin as mn
```

> **NOTE**: If you are working in Google Colab, load the matplotlib backend before producing plots:
>
> ```python
> %matplotlib inline
> ```

Here is a basic example of how to use `MultiMin` to fit a 3D distribution composed of 2 Multivariate Normals.

### 1. Define a true distribution

First, we define a distribution from which we will generate synthetic data. We use a **Composed Multivariate Normal Distribution (CMND)** with 2 Gaussian components (`Ngauss=2`) in 3 dimensions (`Nvars=3`).

```python
import numpy as np
import multimin as mn

# Define parameters for 2 Gaussian components
weights = [0.5, 0.5]
mus = [[1.0, 0.5, -0.5], [1.0, -0.5, +0.5]]
sigmas = [[1, 1.2, 2.3], [0.8, 0.2, 3.3]]
deg = np.pi/180
angles = [
    [10*deg, 30*deg, 20*deg],
    [-20*deg, 0*deg, 30*deg],
] 

# Calculate covariance matrices from rotation angles
Sigmas = mn.Stats.calc_covariance_from_rotation(sigmas, angles)

# Create the CMND object
CMND = mn.ComposedMultiVariateNormal(mus=mus, weights=weights, Sigmas=Sigmas)
```

### 2. Generate sample data

We generate 5000 random samples from this distribution to serve as our "observed" data.

```python
np.random.seed(1)
data = CMND.rvs(5000)
```

### 3. Visualize the data

We can check the distribution of the generated data using `CornerPlot`.

```python
import matplotlib.pyplot as plt

# Define properties labels
properties = dict(
    x=dict(label=r"$x$", range=None),
    y=dict(label=r"$y$", range=None),
    z=dict(label=r"$z$", range=None),
)

# Plot the corner plot
G = mn.CornerPlot(properties, figsize=3)
hargs = dict(bins=30, cmap='Spectral_r')
sargs = dict(s=1.2, edgecolor='None', color='r')
hist = G.scatter_plot(data, **sargs)
```

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/examples/gallery/multimin_quickstart_1.png" alt="Data Scatter Plot" width="600"/>
</div>

### 4. Initialize the Fitter and Run the Fit

We initialize the `FitCMND` handler with the expected number of Gaussians (2) and variables (3). We then run the fitting procedure.

```python
# Initialize the fitter
F = mn.FitCMND(Ngauss=2, Nvars=3)

# Run the fit (using advance=True for better convergence on complex models)
F.fit_data(data, advance=True)
```

### 5. Check and Plot Results

Finally, we visualize the fitted distribution compared to the data.

```python
# Plot the fit result
G = F.plot_fit(
    props=["x", "y", "z"],
    hargs=dict(bins=30, cmap='YlGn'),
    sargs=dict(s=0.2, edgecolor='None', color='r'),
    figsize=3
)
```

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/examples/gallery/multimin_quickstart_3.png" alt="Fit Result" width="600"/>
</div>


## Citation

The numerical tools and codes provided in this package have been developed and tested over several years of scientific research.

If you use `MultiMin` in your research, please cite:

```bibtex
@software{multimin2026,
  author = {Zuluaga, Jorge I.},
  title = {MultiMin: Multivariate Gaussian fitting},
  year = {2026},
  url = {https://github.com/seap-udea/multimin}
}
```

## What's New

For a detailed list of changes and new features, see [WHATSNEW.md](WHATSNEW.md).

## Authors and Licensing

This project is developed by the Solar, Earth and Planetary Physics Group (SEAP) at Universidad de Antioquia, Medellín, Colombia. The main developers are:

- **Jorge I. Zuluaga** - jorge.zuluaga@udea.edu.co


This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! If you're interested in contributing to MultiMin, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.
