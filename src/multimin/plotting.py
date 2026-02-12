##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                        #
#                                                                #
# Authors: Jorge I. Zuluaga                                      #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)       #
##################################################################

"""
Visualization and plotting utilities for MultiMin package.

Contains:
- MultiPlot: Grid plotting for N-dimensional data projections
- multimin_watermark: Add watermark to plots
"""

import warnings
import numpy as np
from matplotlib import pyplot as plt

# Import from package modules
from .base import MultiMinBase
from .util import Util
from .version import __version__


# =============================================================================
# VISUALIZATION
# =============================================================================
def multimin_watermark(ax, frac=1 / 4, alpha=1):
    """Add a water mark to a 2d or 3d plot.

    Parameters:

        ax: Class axes:
            Axe where the pryngles mark will be placed.
    """
    # Import show_watermark from main module at runtime
    import multimin as mn

    if not mn.show_watermark:
        return None
    # Get the height of axe
    axh = (
        ax.get_window_extent()
        .transformed(ax.get_figure().dpi_scale_trans.inverted())
        .height
    )
    fig_factor = frac * axh

    # Options of the water mark
    args = dict(
        rotation=270,
        ha="left",
        va="top",
        transform=ax.transAxes,
        color="pink",
        fontsize=8 * fig_factor,
        zorder=100,
        alpha=alpha,
    )

    # Text of the water mark
    mark = f"MultiMin {__version__}"

    # Choose the according to the fact it is a 2d or 3d plot
    try:
        ax.add_collection3d
        plt_text = ax.text2D
    except:
        plt_text = ax.text

    text = plt_text(1, 1, mark, **args)
    return text


class MultiPlot(MultiMinBase):
    """
    Create a grid of plots showing the projection of a N-dimensional data.

    Parameters
    ----------
    properties : dict
        List of properties to be shown, dictionary of dictionaries (N entries).
        Keys are label of attribute, ex. "q".
        Dictionary values:

        * label: label used in axis, string
        * range: range for property, tuple (2)
    figsize : int, optional
        Base size for panels (the size of figure will be M x figsize), default 3.
    fontsize : int, optional
        Base fontsize, default 10.
    direction : str, optional
        Direction of ticks in panels, default 'out'.

    Attributes
    ----------
    N : int
        Number of properties.
    M : int
        Size of grid matrix (M=N-1).
    fw : int
        Figsize.
    fs : int
        Fontsize.
    fig : matplotlib.figure.Figure
        Figure handle.
    axs : numpy.ndarray
        Matrix with subplots, axes handles (MxM).
    axp : dict
        Matrix with subplots, dictionary of dictionaries.
    properties : list
        List of properties labels, list of strings (N).

    Methods
    -------
    tight_layout()
        Tight layout if no constrained_layout was used.
    set_labels(**args)
        Set labels parameters.
    set_ranges()
        Set ranges in panels according to ranges defined in dparameters.
    set_tick_params(**args)
        Set tick parameters.
    sample_hist(data, colorbar=False, **args)
        Create a 2d-histograms of data on all panels of the MultiPlot.
    sample_scatter(data, **args)
        Scatter plot on all panels of the MultiPlot.
    mog_pdf(mog, **args)
        Plot the PDF of a MoG on all panels of the MultiPlot.
    mog_contour(mog, **args)
        Plot the contours of a MoG on all panels of the MultiPlot.

    """

    def __init__(self, properties, figsize=3, fontsize=10, direction="out"):

        # Basic attributes
        self.dproperties = properties
        self.properties = list(properties.keys())
        self.data = None

        # Secondary attributes
        self.N = len(properties)
        self.M = max(1, self.N - 1)  # 1 when univariate so we have one panel
        self._univariate = self.N == 1

        # Optional properties
        self.fw = figsize
        self.fs = fontsize

        # Univariate: single 1D panel
        if self._univariate:
            from matplotlib import pyplot as plt

            self.fig, ax = plt.subplots(
                1, 1, constrained_layout=True, figsize=(self.fw * 1.5, self.fw)
            )
            self.axs = np.array([[ax]])
            self.constrained = True
            self.single = True
            self.axp = dict()
            prop0 = self.properties[0]
            self.axp[prop0] = {prop0: ax}
            ax.set_xlabel(self.dproperties[prop0]["label"], fontsize=fontsize)
            self.tight_layout()
            return

        # Create figure and axes: it works
        try:
            from matplotlib import pyplot as plt

            self.fig, self.axs = plt.subplots(
                self.M,
                self.M,
                constrained_layout=True,
                figsize=(self.M * self.fw, self.M * self.fw),
                sharex="col",
                sharey="row",
            )
            self.constrained = True
        except:
            self.fig, self.axs = plt.subplots(
                self.M,
                self.M,
                figsize=(self.M * self.fw, self.M * self.fw),
                sharex="col",
                sharey="row",
            )
            self.constrained = False

        if not isinstance(self.axs, np.ndarray):
            self.axs = np.array([[self.axs]])
            self.single = True
        else:
            self.single = False

        # Create named axis
        self.axp = dict()
        for j in range(self.N):
            propj = self.properties[j]
            if propj not in self.axp.keys():
                self.axp[propj] = dict()
            for i in range(self.N):
                propi = self.properties[i]
                if i == j:
                    continue
                if propi not in self.axp.keys():
                    self.axp[propi] = dict()
                if i < j:
                    self.axp[propj][propi] = self.axp[propi][propj]
                    continue
                self.axp[propj][propi] = self.axs[i - 1][j]

        # Deactivate unused panels
        for i in range(self.M):
            for j in range(i + 1, self.M):
                self.axs[i][j].axis("off")

        # Place ticks
        for i in range(self.M):
            for j in range(i + 1):
                if not self.single:
                    self.axs[i, j].tick_params(axis="both", direction=direction)
                else:
                    self.axs[i, i].tick_params(axis="both", direction=direction)
        for i in range(self.M):
            self.axs[i, 0].tick_params(axis="y", direction="out")
            self.axs[self.M - 1, i].tick_params(axis="x", direction="out")

        # Set properties of panels
        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()

    def tight_layout(self):
        """
        Tight layout if no constrained_layout was used.


        """
        if self.constrained == False:
            self.fig.subplots_adjust(wspace=self.fw / 100.0, hspace=self.fw / 100.0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The figure layout has changed to tight"
            )
            self.fig.tight_layout()

    def set_tick_params(self, **args):
        """
        Set tick parameters.
        Ex. set_tick_params(labelsize=10)

        Parameters
        ----------
        **args : dict
            Same arguments as tick_params method.


        """
        opts = dict(axis="both", which="major", labelsize=0.8 * self.fs)
        opts.update(args)
        for i in range(self.M):
            for j in range(self.M):
                self.axs[i][j].tick_params(**opts)

    def set_ranges(self):
        """
        Set ranges in panels according to ranges defined in dparameters.


        """
        if getattr(self, "_univariate", False):
            prop = self.properties[0]
            if self.dproperties[prop]["range"] is not None:
                self.axs[0][0].set_xlim(self.dproperties[prop]["range"])
            return
        for i, propi in enumerate(self.properties):
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue
                if self.dproperties[propi]["range"] is not None:
                    self.axp[propi][propj].set_xlim(self.dproperties[propi]["range"])
                if self.dproperties[propj]["range"] is not None:
                    self.axp[propi][propj].set_ylim(self.dproperties[propj]["range"])

    def reset_ranges(self):
        """
        Reset ranges to match the data limits.
        """
        if self.data is not None:
            for i, prop in enumerate(self.properties):
                dmin, dmax = self.data[:, i].min(), self.data[:, i].max()
                # Force range to data limits (overriding default 4-sigma extents of PDF)
                self.dproperties[prop]["range"] = [dmin, dmax]
        self.set_ranges()

    def set_labels(self, **args):
        """
        Set labels parameters.
        Ex. set_labels(fontsize=12)

        Parameters
        ----------
        **args : dict
            Common arguments of set_xlabel, set_ylabel and text.


        """
        opts = dict(fontsize=self.fs)
        opts.update(args)
        for i, prop in enumerate(self.properties[:-1]):
            label = self.dproperties[prop]["label"]
            self.axs[self.M - 1][i].set_xlabel(label, **opts)
        for i, prop in enumerate(self.properties[1:]):
            label = self.dproperties[prop]["label"]
            self.axs[i][0].set_ylabel(label, rotation=90, labelpad=10, **opts)

        for i in range(1, self.M):
            label = self.dproperties[self.properties[i]]["label"]
            self.axs[i - 1][i].text(
                0.5,
                0.0,
                label,
                ha="center",
                transform=self.axs[i - 1][i].transAxes,
                **opts,
            )
            # 270 if you want rotation
            self.axs[i - 1][i].text(
                0.0,
                0.5,
                label,
                rotation=270,
                va="center",
                transform=self.axs[i - 1][i].transAxes,
                **opts,
            )

        label = self.dproperties[self.properties[0]]["label"]
        if not self.single:
            self.axs[0][1].text(
                0.0,
                1.0,
                label,
                rotation=0,
                ha="left",
                va="top",
                transform=self.axs[0][1].transAxes,
                **opts,
            )

        label = self.dproperties[self.properties[-1]]["label"]
        # 270 if you want rotation
        self.axs[-1][-1].text(
            1.05,
            0.5,
            label,
            rotation=270,
            ha="left",
            va="center",
            transform=self.axs[-1][-1].transAxes,
            **opts,
        )

        self.tight_layout()

    def sample_hist(self, data, colorbar=False, **args):
        """
        Create a 2d-histograms of data on all panels of the MultiPlot.
        Ex. G.sample_hist(data, bins=100, cmap='viridis')

        Parameters
        ----------
        data : numpy.ndarray
            Data to be histogramed (n=len(data)), numpy array (nxN).
        colorbar : bool, optional
            Include a colorbar? (default False).
        **args : dict
            All arguments of hist2d method.

        Returns
        -------
        hist : list
            List of histogram instances.

        Examples
        --------
        >>> properties = {
        ...     'Q': {'label': r"$Q$", 'range': None},
        ...     'E': {'label': r"$C$", 'range': None},
        ...     'I': {'label': r"$I$", 'range': None},
        ... }
        >>> G = mm.MultiPlot(properties, figsize=3)
        >>> hargs = dict(bins=100, cmap='viridis')
        >>> hist = G.sample_hist(udata, **hargs)


        """
        self.data = data
        opts = dict()
        opts.update(args)

        # Default zorder for histogram (background)
        if "zorder" not in opts:
            opts["zorder"] = -100

        # Univariate: 1D histogram (same style as plot_sample)
        if getattr(self, "_univariate", False):
            ax = self.axs[0][0]
            hargs_1d = {k: v for k, v in opts.items() if k != "cmap"}
            if "bins" not in hargs_1d:
                hargs_1d["bins"] = min(50, max(10, len(data) // 20))
            if "density" not in hargs_1d:
                hargs_1d["density"] = True
            hargs_1d.setdefault("label", "sample histogram")
            ax.hist(data[:, 0], **hargs_1d)
            ax.yaxis.set_label_position("left")
            ax.set_ylabel("density")
            # Legend (univariate): if no twin yet, add legend for histogram only
            handles, labels = ax.get_legend_handles_labels()
            if handles and getattr(self, "_ax_twin", None) is None:
                ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles),
                    frameon=False,
                )
                self.fig.subplots_adjust(top=0.88)
            self.set_ranges()
            self.set_tick_params()
            self.tight_layout()
            if not getattr(self, "_watermark_added", False):
                multimin_watermark(
                    ax, frac=0.5
                )  # larger frac for single panel (match 2-panel size)
                self._watermark_added = True
            return []

        hist = []
        for i, propi in enumerate(self.properties):
            if self.dproperties[propi]["range"] is not None:
                xmin, xmax = self.dproperties[propi]["range"]
            else:
                xmin = data[:, i].min()
                xmax = data[:, i].max()
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue

                if self.dproperties[propj]["range"] is not None:
                    ymin, ymax = self.dproperties[propj]["range"]
                else:
                    ymin = data[:, j].min()
                    ymax = data[:, j].max()

                opts["range"] = [[xmin, xmax], [ymin, ymax]]
                h, xe, ye, im = self.axp[propi][propj].hist2d(
                    data[:, i], data[:, j], **opts
                )

                hist += [im]
                if colorbar:
                    # Create color bar
                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    divider = make_axes_locatable(self.axp[propi][propj])
                    cax = divider.append_axes("top", size="9%", pad=0.1)
                    self.fig.add_axes(cax)
                    cticks = np.linspace(h.min(), h.max(), 10)[2:-1]
                    self.fig.colorbar(
                        im,
                        ax=self.axp[propi][propj],
                        cax=cax,
                        orientation="horizontal",
                        ticks=cticks,
                    )
                    cax.xaxis.set_tick_params(
                        labelsize=0.5 * self.fs, direction="in", pad=-0.8 * self.fs
                    )
                    xt = cax.get_xticks()
                    xm = xt.mean()
                    m, e = Util.mantisa_exp(xm)
                    xtl = []
                    for x in xt:
                        xtl += ["%.1f" % (x / 10**e)]
                    cax.set_xticklabels(xtl)
                    cax.text(
                        0,
                        0.5,
                        r"$\times 10^{%d}$" % e,
                        ha="left",
                        va="center",
                        transform=cax.transAxes,
                        fontsize=6,
                        color="w",
                    )

        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
        return hist

    def sample_scatter(self, data, **args):
        """
        Scatter plot on all panels of the MultiPlot.
        Ex. G.sample_scatter(data, s=0.2, color='r')

        Parameters
        ----------
        data : numpy.ndarray
            Data to be histogramed (n=len(data)), numpy array (nxN).
        **args : dict
            All arguments of scatter method.

        Returns
        -------
        scatter : list
            List of scatter instances.

        Examples
        --------
        >>> sargs = dict(s=0.2, edgecolor='None', color='r')
        >>> hist = G.sample_scatter(udata, **sargs)


        """
        self.data = data
        # Univariate: scatter on a twin y-axis so data range is independent of PDF/density
        if getattr(self, "_univariate", False):
            ax = self.axs[0][0]
            ax_twin = ax.twinx()
            x = data[:, 0]
            y_jitter = np.random.uniform(0, 1, size=len(x))
            sargs_1d = dict(args)
            sargs_1d.setdefault("label", "sample")
            # Default zorder for scatter (foreground)
            sargs_1d.setdefault("zorder", 100)
            sc = ax_twin.scatter(x, y_jitter, **sargs_1d)
            ax_twin.set_ylim(0, 1)
            ax_twin.set_yticks([])
            prop_name = self.properties[0]
            ax_twin.set_ylabel(
                "sample " + self.dproperties[prop_name]["label"], fontsize=self.fs
            )
            self._ax_twin = ax_twin  # store for reference
            # Legend: combine primary ax (e.g. histogram) and twin (sample scatter)
            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles),
                    frameon=False,
                )
                self.fig.subplots_adjust(top=0.88)  # room for legend above
            self.set_ranges()
            self.set_tick_params()
            self.tight_layout()
            if not getattr(self, "_watermark_added", False):
                multimin_watermark(
                    ax, frac=0.5
                )  # larger frac for single panel (match 2-panel size)
                self._watermark_added = True
            return [sc]

        scatter = []
        # Default zorder for scatter (foreground)
        if "zorder" not in args:
            args["zorder"] = 100

        for i, propi in enumerate(self.properties):
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue
                scatter += [
                    self.axp[propi][propj].scatter(data[:, i], data[:, j], **args)
                ]

        self.set_labels()
        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
        return scatter

    def mog_pdf(self, mog, grid_size=200, **args):
        """
        Plot the PDF of a MoG on all panels of the MultiPlot.
        Ex. G.mog_pdf(mog, color='k', lw=2)

        Parameters
        ----------
        mog : MixtureOfGaussians
            MoG object to plot.
        grid_size : int, optional
            Number of points for the grid (default 200).
        **args : dict
            Arguments for the plot function (e.g. color, linewidth).
        """
        opts = dict(color="k", lw=2)
        opts.update(args)

        # Default zorder for PDF (background)
        # Note: User requested zorder=100 for background and -100 for foreground,
        # but standard is low=back, high=front. We use -100 for background.
        if "zorder" not in opts:
            opts["zorder"] = -100

        if getattr(self, "_univariate", False):
            # Filter out arguments not supported by ax.plot
            # (e.g. cmap/colorbar are for pcolormesh/images)
            plot_opts = opts.copy()
            for key in ["cmap", "colorbar"]:
                plot_opts.pop(key, None)

            ax = self.axs[0][0]
            if "label" not in plot_opts:
                plot_opts["label"] = "PDF"

            if self.dproperties[self.properties[0]]["range"] is not None:
                xmin, xmax = self.dproperties[self.properties[0]]["range"]
            else:
                bounds = getattr(mog, "_domain_bounds", None)
                if (
                    bounds is not None
                    and np.isfinite(bounds[0][0])
                    and np.isfinite(bounds[0][1])
                ):
                    xmin, xmax = bounds[0]
                else:
                    # Robust auto-range based on mus/sigmas
                    mu_min = np.min(mog.mus[:, 0])
                    mu_max = np.max(mog.mus[:, 0])
                    sig_max = np.max(mog.sigmas[:, 0])
                    nsig = 4.0
                    xmin = mu_min - nsig * sig_max
                    xmax = mu_max + nsig * sig_max
                    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
                        xmin, xmax = mu_min - 1.0, mu_max + 1.0

            x = np.linspace(xmin, xmax, int(grid_size))
            y = mog.pdf(x.reshape(-1, 1))
            ax.plot(x, y, **plot_opts)

            # Update y-limits if needed
            if y.size > 0:
                current_ylim = ax.get_ylim()
                new_ymax = max(current_ylim[1], float(np.max(y)) * 1.05)
                ax.set_ylim(0, new_ymax)

            self.set_ranges()
            self.set_tick_params()
            self.tight_layout()
            if not getattr(self, "_watermark_added", False):
                multimin_watermark(ax, frac=0.5)
                self._watermark_added = True
            return

        # Multivariate case
        w = np.asarray(mog.weights, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(w) / max(1, w.size)
        else:
            w = w / w_sum
        base_point = np.average(mog.mus, axis=0, weights=w)

        # Helper to get range for a variable (index k, name prop, axis ax)
        def _get_range(k, prop, ax, axis_idx=0):  # axis_idx 0 for x, 1 for y
            # 1. User specified range in properties
            if self.dproperties[prop]["range"] is not None:
                return self.dproperties[prop]["range"]

            # 2. Existing axis limits (if data is present)
            # Check if axis has data that might have set limits
            has_data = (
                ax.has_data()
                or len(ax.collections) > 0
                or len(ax.images) > 0
                or len(ax.lines) > 0
            )
            if has_data:
                if axis_idx == 0:
                    return ax.get_xlim()
                else:
                    return ax.get_ylim()

            # 3. MoG bounds
            bounds = getattr(mog, "_domain_bounds", None)
            if bounds is not None:
                lo, hi = bounds[k]
                if np.isfinite(lo) and np.isfinite(hi):
                    return [float(lo), float(hi)]

            # 4. Auto-range based on MoG parameters
            mu_min = float(np.min(mog.mus[:, k]))
            mu_max = float(np.max(mog.mus[:, k]))
            sig_max = float(np.max(mog.sigmas[:, k]))
            nsig = 4.0
            lo = mu_min - nsig * sig_max
            hi = mu_max + nsig * sig_max
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = mu_min - 1.0, mu_max + 1.0
            return [lo, hi]

        first_im = None
        cmap = args.get("cmap", "Spectral_r")  # Extract cmap from args or default

        for i, propi in enumerate(self.properties):
            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue

                ax = self.axp[propi][propj]

                x_min, x_max = _get_range(i, propi, ax, 0)
                y_min, y_max = _get_range(j, propj, ax, 1)

                xs = np.linspace(float(x_min), float(x_max), int(grid_size))
                ys = np.linspace(float(y_min), float(y_max), int(grid_size))

                # Careful with meshgrid indexing for pcolormesh
                xx, yy = np.meshgrid(xs, ys, indexing="xy")
                pts = np.column_stack([xx.ravel(), yy.ravel()])

                X_full = np.tile(base_point, (pts.shape[0], 1))
                X_full[:, i] = pts[:, 0]
                X_full[:, j] = pts[:, 1]

                zz = np.asarray(mog.pdf(X_full), dtype=float).reshape(xx.shape)

                # pcolormesh
                # Use zorder from args if present, else default to -100
                zorder = args.get("zorder", -100)
                im = ax.pcolormesh(xx, yy, zz, shading="auto", cmap=cmap, zorder=zorder)
                if first_im is None:
                    first_im = (ax, im)

        # Handle colorbar if requested (logic from original plot_pdf)
        # Note: colorbar arg was not explicitly in mog_pdf signature in previous snippet
        # but usage in plot_pdf(..., colorbar=False) suggests it might be passed in **args or needed.
        # The user's snippet for mog_pdf(self, mog, grid_size=200, **args)
        # If colorbar is needed, we should check args.
        if args.get("colorbar", False) and first_im is not None:
            ax0, im0 = first_im
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("top", size="9%", pad=0.1)
            self.fig.add_axes(cax)
            vmin = float(np.nanmin(im0.get_array()))
            vmax = float(np.nanmax(im0.get_array()))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                cticks = np.linspace(vmin, vmax, 8)[1:-1]
            else:
                cticks = None
            self.fig.colorbar(
                im0, ax=ax0, cax=cax, orientation="horizontal", ticks=cticks
            )
            cax.xaxis.set_tick_params(
                labelsize=0.5 * self.fs, direction="in", pad=-0.8 * self.fs
            )

        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        if not getattr(self, "_watermark_added", False):
            multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
            self._watermark_added = True

    def mog_contour(self, mog, grid_size=200, **args):
        """
        Plot the contours of a MoG on all panels of the MultiPlot.
        Ex. G.mog_contour(mog, levels=5, cmap='Reds')

        Parameters
        ----------
        mog : MixtureOfGaussians
            MoG object to plot.
        grid_size : int, optional
            Number of points for the grid (default 200).
        **args : dict
            Arguments for contour function.
        """
        opts = dict(levels=5, cmap="Reds", legend=True)
        opts.update(args)

        if getattr(self, "_univariate", False):
            # Contours don't make sense in 1D, maybe strict validation or ignore?
            return

        # Decomposition handling
        decomp = args.pop("decomp", False)

        # We need to access MixtureOfGaussians to create components if decomp=True
        from .mog import MixtureOfGaussians

        # Collect legend handles if decomp=True
        legend_handles = []
        legend_labels = []

        for i, propi in enumerate(self.properties):
            if self.dproperties[propi]["range"] is not None:
                xmin, xmax = self.dproperties[propi]["range"]
            else:
                xmin, xmax = (
                    self.axp[propi][self.properties[i + 1]].get_xlim()
                    if i + 1 < self.N
                    else (0, 1)
                )

            for j, propj in enumerate(self.properties):
                if j <= i:
                    continue

                if self.dproperties[propj]["range"] is not None:
                    ymin, ymax = self.dproperties[propj]["range"]
                else:
                    ymin, ymax = self.axp[propi][propj].get_ylim()

                # Evaluation grid
                xi = np.linspace(xmin, xmax, grid_size)
                yi = np.linspace(ymin, ymax, grid_size)
                Xi, Yi = np.meshgrid(xi, yi)

                # Helper to plot a specific MoG (full or component)
                def plot_mog_instance(sub_mog, style_opts):
                    # Full vector X
                    X_full = np.zeros((grid_size * grid_size, sub_mog.nvars))
                    mean_vec = np.average(sub_mog.mus, axis=0, weights=sub_mog.weights)
                    X_full[:] = mean_vec
                    X_full[:, i] = Xi.ravel()
                    X_full[:, j] = Yi.ravel()

                    Z = sub_mog.pdf(X_full).reshape(grid_size, grid_size)

                    # Adjust levels to avoid white frame
                    current_opts = style_opts.copy()
                    if isinstance(style_opts.get("levels"), int):
                        nlevels = style_opts["levels"]
                        zmax = Z.max()
                        current_opts["levels"] = np.linspace(
                            0.1 * zmax, 0.95 * zmax, nlevels
                        )

                    cntr = self.axp[propi][propj].contour(Xi, Yi, Z, **current_opts)
                    return cntr

                if not decomp:
                    plot_mog_instance(mog, opts)
                else:
                    # Decomposition: plot each component
                    for k in range(mog.ngauss):
                        # Extract component
                        mu_k = mog.mus[k : k + 1]
                        sigma_k = mog.Sigmas[k : k + 1]
                        rho_k = mog.rhos[k : k + 1] if mog.rhos is not None else None

                        # Create component MoG
                        # Create component MoG
                        # rhos are implicit in Sigmas, so we don't pass them to init
                        comp_mog = MixtureOfGaussians(
                            mus=mu_k,
                            Sigmas=sigma_k,
                            weights=[1.0],
                            domain=getattr(mog, "domain", None),
                            normalize_weights=False,
                        )

                        # Style for component
                        comp_opts = opts.copy()
                        # Cycle colors: C0, C1...
                        color = f"C{k % 10}"
                        comp_opts["colors"] = color
                        comp_opts.pop("cmap", None)  # Remove cmap to use colors

                        plot_mog_instance(comp_mog, comp_opts)

                        # Collect legend info (only need to do this once for the first 2D panel found)
                        if len(legend_handles) < mog.ngauss:
                            # Create a dummy line for legend
                            from matplotlib.lines import Line2D

                            line = Line2D([0], [0], color=color, lw=2)
                            legend_handles.append(line)

                            mu_i = mu_k[0, i]
                            mu_j = mu_k[0, j]
                            # Safe sigma calculation
                            var_i = sigma_k[0, i, i]
                            var_j = sigma_k[0, j, j]
                            sig_i = np.sqrt(max(0, var_i))
                            sig_j = np.sqrt(max(0, var_j))

                            # Calculate rho from covariance matrix
                            cov_ij = sigma_k[0, i, j]
                            if sig_i > 0 and sig_j > 0:
                                rho_val = cov_ij / (sig_i * sig_j)
                            else:
                                rho_val = 0.0

                            # Safe sigma calculation
                            var_i = sigma_k[0, i, i]
                            var_j = sigma_k[0, j, j]
                            sig_i = np.sqrt(max(0, var_i))
                            sig_j = np.sqrt(max(0, var_j))

                            # Calculate rho from covariance matrix
                            cov_ij = sigma_k[0, i, j]
                            if sig_i > 0 and sig_j > 0:
                                rho_val = cov_ij / (sig_i * sig_j)
                            else:
                                rho_val = 0.0

                            label = rf"Comp {k + 1}: $\mu$=({mu_i:.2f}, {mu_j:.2f}), $\sigma$=({sig_i:.2f}, {sig_j:.2f}), $\rho$={rho_val:.2f}"
                            legend_labels.append(label)

        if decomp and legend_handles and opts["legend"]:
            # Add legend to the right of G.axs[0][0]
            # We anchor it to axs[0][0] (top-left panel)
            ax_ref = self.axs[0][0]
            ax_ref.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                frameon=True,
                fontsize=6,
            )

        self.set_ranges()
        self.set_tick_params()
        self.tight_layout()
        multimin_watermark(self.axs[0][0], frac=1 / 4 * self.axs.shape[0])
