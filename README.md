
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
<!-- [![downloads](https://img.shields.io/pypi/dw/multimin)](https://pypi.org/project/multimin/) -->

<div align="center">
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

## Examples

Working examples and tutorials will be added as the package develops.

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
