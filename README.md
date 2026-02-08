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
- **Visualization**: Density plots and specific visualization utilities.
- **Statistical Analysis**: Tools for handling covariance matrices and correlations.

## Resources

- Documentation including examples and full API documentation: [https://multimin.readthedocs.io](https://multimin.readthedocs.io).
- PyPI project page: [https://pypi.org/project/multimin/](https://pypi.org/project/multimin/).
- Github repo: [https://github.com/seap-udea/multimin](https://github.com/seap-udea/multimin)

## Installation

### From PyPI

`MultiMin` is available on PyPI at [https://pypi.org/project/multimin/](https://pypi.org/project/multimin/). You can install it with:

```bash
pip install -U multimin
```

If you prefer, you can install the latest version of the developers taking it from the github repo:

```bash
pip install -U git+https://github.com/seap-udea/multimin
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
!pip install -Uq multimin
```

or

```bash
pip install -Uq git+https://github.com/seap-udea/multimin
```

## Theoretical Background

The core of `MultiMin` is the **Composed Multivariate Normal Distribution (CMND)**. The theory behind it posits that any multivariate distribution function $p(\tilde U):\Re^{N}\rightarrow\Re$, where $\tilde U:(u_1,u_2,u_3,\ldots,u_N)$ are random variables, can be approximated with arbitrary precision by a normalized linear combination of $M$ Multivariate Normal Distributions (MND):

$$
p(\tilde U) \approx \mathcal{C}_M(\tilde U; \{w_k\}_M, \{\mu_k\}_M, \{\Sigma_k\}_M) \equiv \sum_{i=1}^{M} w_i\mathcal{N}(\tilde U; \tilde \mu_i, \Sigma_i)
$$

where the multivariate normal $\mathcal{N}(\tilde U; \tilde \mu, \Sigma)$ with mean vector $\tilde \mu$ and covariance matrix $\Sigma$ is given by:

$$
\mathcal{N}(\tilde U; \tilde \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^{k} \det \Sigma}} \exp\left[-\frac{1}{2}(\tilde U - \tilde \mu)^{\rm T} \Sigma^{-1} (\tilde U - \tilde \mu)\right]
$$

The covariance matrix $\Sigma$ elements are defined as $\Sigma_{ij} = \rho_{ij}\sigma_{i}\sigma_{j}$, where $\sigma_i$ is the standard deviation of $u_i$ and $\rho_{ij}$ is the correlation coefficient between variable $u_i$ and $u_j$ ($-1<\rho_{ij}<1$, $\rho_{ii}=1$).

The normalization condition on $p(\tilde U)$ implies that the set of weights $\{w_k\}_M$ are also normalized, i.e., $\sum_i w_i=1$.

### Fitting procedure

To estimate the parameters of the CMND that best describe a given dataset ,
we use the **Likelihood Statistics** method.

Given a dataset of $S$ objects with state vectors $\{\tilde U_k\}_{k=1}^S$, the likelihood $\mathcal{L}$ of the
CMND parameters is defined as the product of the probability densities evaluated at each data point:

$$
\mathcal{L} = \prod_{i=1}^{S} \mathcal{C}_M(\tilde U_i)
$$

The goal is to find the set of parameters (weights, means, and covariances) that maximize this likelihood.
In practice, it is numerically more stable to minimize the **negative normalized log-likelihood**:

$$
-\frac{\log \mathcal{L}}{S} = -\frac{1}{S} \sum_{i=1}^{S} \log \mathcal{C}_M(\tilde U_i)
$$

This approach allows us to fit the distribution without making strong assumptions about the underlying
normality of the data, effectively treating the CMND as a series expansion of the true probability density function.

In `MultiMin`, we use the `scipy.optimize.minimize` function to find the set of parameters that minimize the negative normalized log-likelihood.

## Quickstart

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

First, we define a distribution from which we will generate synthetic data. We use a **Composed Multivariate Normal Distribution (CMND)** with 2 Gaussian components (`ngauss=2`) in 3 dimensions (`nvars=3`).

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
sample = CMND.rvs(5000)
```

### 3. Visualize the data

We can check the distribution of the generated data using `DensityPlot`.

```python
import matplotlib.pyplot as plt

# Define properties labels
properties = dict(
    x=dict(label=r"$x$", range=None),
    y=dict(label=r"$y$", range=None),
    z=dict(label=r"$z$", range=None),
)

# Plot the density plot
G = mn.DensityPlot(properties, figsize=3)

hargs=dict(bins=30,cmap='Spectral_r')
histogram=G.plot_hist(sample,**hargs)

sargs=dict(s=0.5,edgecolor='None',color='r')
scatter=G.scatter_plot(sample,**sargs)
```

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/examples/gallery/cmnd_data_density_scatter.png" alt="Data Scatter Plot" width="600"/>
</div>

### 4. Initialize the Fitter and Run the Fit

We initialize the `FitCMND` handler with the expected number of Gaussians (2) and variables (3). We then run the fitting procedure.

```python
# Initialize the fitter
F = mn.FitCMND(ngauss=2, nvars=3)

# Run the fit (using advance=True for better convergence on complex models)
F.fit_data(sample, advance=True)
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
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/examples/gallery/cmnd_fit_result_3d.png" alt="Fit Result" width="600"/>
</div>

### 6. Inspect Parameters and Get Explicit PDF Function

You can tabulate the fitted parameters and obtain an explicit Python function that evaluates the fitted PDF. Below, each step is shown with its output.

**Stage 1: Tabulate the fitted CMND**

```python
F.cmnd.tabulate(sort_by='weight')
```

Output:

```
                  w      mu_1      mu_2      mu_3   sigma_1   sigma_2   sigma_3    rho_12    rho_13    rho_23
component                                                                                                    
2          0.509108  1.019245 -0.480997  0.618821  0.794906  0.245786  3.327537  0.539417 -0.008936 -0.017769
1          0.490892  0.957687  0.517584 -0.463392  1.039489  1.538029  2.116544 -0.209695  0.121184 -0.527142
```

**Stage 2: Get the source code and a callable function**

```python
code, cmnd = F.cmnd.get_function()
```

Output (the printed code, which you can copy):

```
from multimin import nmd

def cmnd(X):

    mu1_1 = 0.957687
    mu1_2 = 0.517584
    mu1_3 = -0.463392
    mu1 = [mu1_1, mu1_2, mu1_3]
    Sigma1 = [[1.080538, -0.335252, 0.266619], [-0.335252, 2.365532, -1.716008], [0.266619, -1.716008, 4.479757]]
    n1 = nmd(X, mu1, Sigma1)

    mu2_1 = 1.019245
    mu2_2 = -0.480997
    mu2_3 = 0.618821
    mu2 = [mu2_1, mu2_2, mu2_3]
    Sigma2 = [[0.631876, 0.10539, -0.023637], [0.10539, 0.060411, -0.014533], [-0.023637, -0.014533, 11.072504]]
    n2 = nmd(X, mu2, Sigma2)

    w1 = 0.490892
    w2 = 0.509108

    return (
        w1*n1
        + w2*n2
    )
```

**Stage 3: Evaluate the PDF at a point**

```python
cmnd([1.0, 0.5, -0.5])
```

Output:

```
0.011073778538439395
```

**Stage 4: LaTeX output for papers**

You can get the fitted PDF as a LaTeX string (suitable for inclusion in papers) with parameter values and the definition of the normal distribution:

```python
latex_str, _ = F.cmnd.get_function(print_code=False, type='latex', decimals=4)
print(latex_str)
```

Output:

$$f(\mathbf{x}) = w_1 \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_1, \mathbf{\Sigma}_1) + w_2 \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_2, \mathbf{\Sigma}_2)$$

where

$$w_1 = 0.4909$$
$$\boldsymbol{\mu}_1 = \left( \begin{array}{c} 0.9577 \\ 0.5176 \\ -0.4634 \end{array}\right)$$
$$\mathbf{\Sigma}_1 = \left( \begin{array}{ccc} 1.0805 & -0.3353 & 0.2666 \\ -0.3353 & 2.3655 & -1.716 \\ 0.2666 & -1.716 & 4.4798 \end{array}\right)$$

$$w_2 = 0.5091$$
$$\boldsymbol{\mu}_2 = \left( \begin{array}{c} 1.0192 \\ -0.481 \\ 0.6188 \end{array}\right)$$
$$\mathbf{\Sigma}_2 = \left( \begin{array}{ccc} 0.6319 & 0.1054 & -0.0236 \\ 0.1054 & 0.0604 & -0.0145 \\ -0.0236 & -0.0145 & 11.0725 \end{array}\right)$$

Here the normal distribution is defined as:

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \mathbf{\Sigma}) = \frac{1}{\sqrt{(2\pi)^{{k}} \det \mathbf{\Sigma}}} \exp\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top} \mathbf{\Sigma}^{{-1}} (\mathbf{x}-\boldsymbol{\mu})\right]$$

A parameter table in LaTeX is also available via ``F.cmnd.tabulate(sort_by='weight', type='latex')``.

## Truncated multivariate distributions.

In real problems the domain of the variables is not infinite but bounded into a semi-finite region. 

If we start from the unbounded multivariate normal distribution:

$$
\mathcal{N}_k(\tilde U; \tilde \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^{k}\det \Sigma}} \exp\left[ -\frac{1}{2}(\tilde U - \tilde \mu)^{\rm T}\Sigma^{-1}(\tilde U - \tilde \mu) \right]
$$

Let $T\subset\{l,\dots,m\}$, where $l\leq k$ and $m\leq k$ be the set of indices of the truncated variables, and let $a_i<b_i$ be the truncation bounds for $i\in S$. Define the truncation region:

$$
A_S : \{\tilde U\in\mathbb{R}^k:\ a_i \le \tilde U_i \le b_i \ \ \forall\, i\in T \}
$$

with the remaining coordinates $i\notin T$ unbounded. The partially-truncated multivariate normal distribution is defined by

$$
\mathcal{TN}_T(\tilde U;\tilde\mu,\Sigma,\mathbf{a}_T,\mathbf{b}_T) = \frac{\mathcal{N}_k(\tilde U;\tilde\mu,\Sigma)\,\mathbf{1}_{A_T}(\tilde U)}{Z_ (\tilde\mu,\Sigma,\mathbf{a}_T,\mathbf{b}_T)},
$$

where $\mathbf{1}_{A_T}$ is the indicator function of $A_T$ and the normalization constant is

$$
Z_T(\tilde\mu,\Sigma,\mathbf{a}_T,\mathbf{b}_T)= \int_{A_T}\mathcal{N}_k(\tilde T;\tilde\mu,\Sigma)\,d\tilde T = \mathbb{P}_{\tilde T\sim\mathcal{N}_k(\tilde\mu,\Sigma)}\left(\tilde T\in A_T\right).
$$

### Example: univariate truncated mixture

Define a mixture of two Gaussians on the interval $[0, 1]$ with the **domain** parameter, generate data, and fit with `FitCMND(..., domain=[[0, 1]])`:

```python
import numpy as np
import multimin as mn

# Truncated mixture of 2 Gaussians on [0, 1]
CMND_1d = mn.ComposedMultiVariateNormal(
    mus=[0.2, 0.8],
    weights=[0.5, 0.5],
    Sigmas=[0.01, 0.03],
    domain=[[0, 1]],
)
np.random.seed(1)
data_1d = CMND_1d.rvs(5000)

# Fit with same domain so likelihood and means respect [0, 1]
F_1d = mn.FitCMND(ngauss=2, nvars=1, domain=[[0, 1]])
F_1d.fit_data(data_1d, advance=True)
G = F_1d.plot_fit(hargs=dict(bins=40), sargs=dict(s=0.5, alpha=0.6))
```

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/multimin/master/examples/gallery/truncated_1d_fit.png" alt="Truncated 1D fit" width="500"/>
</div>

See [examples/multimin_truncated_tutorial.ipynb](examples/multimin_truncated_tutorial.ipynb) for 3D truncated examples and more detail.

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

This project is developed by the Solar, Earth and Planetary Physics Group (SEAP) at Universidad de Antioquia, Medellín, Colombia. The main developer is Prof. **Jorge I. Zuluaga** - jorge.zuluaga@udea.edu.co. 

Other beta testers and contributions from:

- **Juanita A. Agudelo** - juanita.agudelo@udea.edu.co. Testing of the initial versions of the package in the context of NEAs research.

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! If you're interested in contributing to MultiMin, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.
