# TODO — Projects and ideas for future development

List of projects, improvements and ideas for future MultiMin development. No implied order of priority.

---

## Functionality and API

- [ ] **Optimization and fitters**
  - Try other optimizers besides `scipy.optimize.minimize` (e.g. emcee, dynesty) for uncertainty estimation.
  - Option to use L-BFGS-B or gradient-free methods for large problems.
  - Configurable stopping criterion (tolerance on log-likelihood or on parameters).

- [ ] **Domains and truncation**
  - Explicit support for semi-infinite domains (e.g. \([a, +\infty)\)) in addition to finite intervals.
  - Documented, reusable utilities for mapping variables to unbounded domains (log, logit).

- [ ] **Initialization and fitting**
  - Automatic initialization of means/widths from K-means or initial mixture by components.
  - Automatic search for number of components (BIC/AIC or cross-validation).
  - In `savefit`: rename argument `objfile` to `prefix` for clarity.

- [ ] **Orbital elements**
  - Decide whether to introduce an in-house `Orbit` class or integrate an external package (rebound, poliastro, etc.).
  - Allow arguments `W`, `w`, `M` (and other parametrizations) for the elements PDF.
  - Compute elements before 4 orbital periods using rebound (or alternative).

- [ ] **Units and constants**
  - Define policy for physical units and constants (astronomical, SI) in data and outputs.

---

## Visualization (DensityPlot and related)

- [ ] **Richer plots** (in line with “Richer density plots” in WHATSNEW)
  - Density contours overlaid on 2D panels.
  - Option for shared or per-panel colorbar in 2D histograms.
  - Consistent legends and labels across plot types (univariate already improved).

- [ ] **Output and gallery**
  - Save multiple figures to `gallery` programmatically (e.g. helper or script).
  - Create `gallery` directory automatically on install or on first use of routines that save figures.

- [ ] **Elements PDF density**
  - Define and document a method (e.g. in notebook or example) to visualize the orbital elements PDF (grid, contours or projections).

---

## Documentation and examples

- [ ] **Docs**
  - Expand API on Read the Docs with more examples in docstrings.
  - Short “best practices” guide (choice of `ngauss`, domain, transformations, convergence).

- [ ] **Notebooks and tutorials**
  - Performance tutorial (fit times, sample size, number of components).
  - Full pipeline example: raw data → transform → fit → export (function/LaTeX).

- [ ] **Tests**
  - Routines to measure execution times and track performance regressions.

---

## Infrastructure and quality

- [ ] **Data and storage**
  - Improve data handling (I/O, formats, validation).
  - Option for external repository (e.g. Google Drive) to store and share CMND fits.

- [ ] **UX and feedback**
  - Integrate `tqdm` (or similar) in long-running tasks (fit, sampling, save).
  - Optional progress messages or logging in `fit_data` and heavy functions.

- [ ] **Packaging and CI**
  - Review and update `MANIFEST.in` / `pyproject.toml` for files that must be distributed.
  - CI for tests, lint and (optionally) docs build on each push/PR.

---

## Science and applications

- [ ] **NEOs / asteroids**
  - Extend the asteroids example (notebook and/or module) with systematic model–data comparison.
  - Integrate with catalogues or NEA data update pipelines if a stable API is defined.

- [ ] **Publication and reproducibility**
  - Scripts or notebooks that reproduce figures and tables from papers (e.g. manuscript-neoflux).
  - Export results in standard format (JSON/YAML) to share fits and metadata.

---

## Open ideas

- [ ] Support for unnormalized weights (softmax or other parametrization) in the user interface.
- [ ] High-level “fit from dataframe” interface (columns → variables, default options).
- [ ] Post-fit diagnostic tools (residuals, QQ, histogram vs PDF comparison).
- [ ] Option to use JAX or vectorized NumPy for log-likelihood and gradients in large problems.

---

*This file is updated with new ideas and when items are completed. For very concrete tasks or bugs, see also the repository issues.*
