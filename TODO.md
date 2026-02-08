# TODO — Projects and ideas for future development

List of projects, improvements and ideas for future MultiMin development. No implied order of priority.

---

## Functionality and API

- [] **Fitting functions**
  - Instead of fitting data, fit functions.
  - Fitting functions in 1- and N-dimensions.

- [ ] **Optimization and fitters**
  - Try other optimizers besides `scipy.optimize.minimize` (e.g. emcee, dynesty) for uncertainty estimation.
  - Option to use L-BFGS-B or gradient-free methods for large problems.
  - Configurable stopping criterion (tolerance on log-likelihood or on parameters).

- [ ] **Domains and truncation**
  - Explicit support for semi-infinite domains (e.g. \([a, +\infty)\)) in addition to finite intervals.
  - Documented, reusable utilities for mapping variables to unbounded domains (log, logit).

- [ ] **Initialization and fitting**
  - Automatic initialization of means/widths from K-means or initial mixture by components.
  
---

## Visualization (DensityPlot and related)

- [ ] **Richer plots** (in line with “Richer density plots” in WHATSNEW)
  - Density contours overlaid on 2D panels.
  - Option for shared or per-panel colorbar in 2D histograms.
  
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

---

## Infrastructure and quality

- [ ] **UX and feedback**
  - Integrate `tqdm` (or similar) in long-running tasks (fit, sampling, save).
  - Optional progress messages or logging in `fit_data` and heavy functions.

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

- [ ] High-level “fit from dataframe” interface (columns → variables, default options).
- [ ] Post-fit diagnostic tools (residuals, QQ, histogram vs PDF comparison).
- [ ] Option to use JAX or vectorized NumPy for log-likelihood and gradients in large problems.

---

*This file is updated with new ideas and when items are completed. For very concrete tasks or bugs, see also the repository issues.*
