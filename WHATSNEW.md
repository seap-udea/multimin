# MultiMin

## What's New

`MultiMin` is under active development. Here you will find a list of features introduced in each version of the package.

## Version History

- **Version 0.11.x**:
  - Added `marginals` parameter to plotting functions (`plot_pdf`, `plot_sample`, `plot_fit`) for displaying marginal distributions on diagonal panels.
  - Introduced `margs` dictionary parameter in `MultiPlot` methods (`sample_scatter`, `sample_hist`, `mog_pdf`, `mog_contour`) for fine-grained control of marginal plot properties.
  - Fixed issue with duplicate watermarks when calling multiple plotting methods on the same `MultiPlot` instance.
  - Fixed marginal distributions vertical offset - now properly aligned at zero.
  - Added 50- and 20-component unbounded MoG example fits to the gallery.
  - Updated example notebooks with improved execution outputs and documentation.

- **Version 0.10.x**:
  - **Major refactoring**: Renamed `CMND` (Composed Multinormal Distribution) to `MoG` (Mixture of Gaussians) throughout the package for better clarity and consistency with statistical literature.
  - **Package modularization**: Split monolithic code into separate modules (`base.py`, `util.py`, `plotting.py`, `mog.py`, `fitting.py`) for better maintainability.
  - **Performance improvements**: Introduced C-optimized routines for distribution calculations using GSL library, significantly improving computation speed.
  - **Enhanced plotting capabilities**:
    - Refactored `MultiPlot` to directly support `mog_pdf` and `mog_contour` methods.
    - Added `decomp` parameter to `mog_contour` for visualizing individual Gaussian components with detailed legends.
    - Added data storage and dynamic range resetting to `MultiPlot`.
  - **New fitting features**:
    - MoG component dropping capability for removing low-weight components.
    - Fitting initialization from existing MoG objects.
    - Improved `FitFunctionMoG` class with multiple fitting modes (`lsq`, `multimodal`, `noisy`, `noisy_multimodal`, `adaptive`).
  - **Comparison with scikit-learn**: Added comprehensive comparison between MultiMin and scikit-learn's GaussianMixture in documentation and examples.
  - Added new CLI (Command Line Interface) for package utilities.
  - Comprehensive updates to tutorials and example notebooks.

- **Versions 0.9.x**:
  - Beta versions.
  - All features working.
  - Unifying option props in plotting routines.

- **Versions 0.6.x**:
  - Include univariate distributions.
  - You can ask the CMND objects to show and return a simplified version of the function.
  - CMND objects now have a method to show in a table its parameters.
  - LaTeX support, ie. the CMND class is able to produce LaTeX code tabulating parameters and function.
  - A full example showing how to apply the tools in the package to the distribution in orbital elements is developed.
  - Initial value of the fitting parameters can now be set.

- **Versions 0.5.x**:

  - Official release of `MultiMin`.
  - Major refactoring of the codebase.
  - Improved documentation.
  - Conservative naming conventions.

### Planned Features

Future versions will include:

- **Versions 0.6.x**:

  - Richer density plots

---

*This document will be updated with each new release.*
