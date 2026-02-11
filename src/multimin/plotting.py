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
- DensityPlot: Grid plotting for N-dimensional data projections
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


class DensityPlot(MultiMinBase):
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
    plot_hist(data, colorbar=False, **args)
        Create a 2d-histograms of data on all panels of the DensityPlot.
    scatter_plot(data, **args)
        Scatter plot on all panels of the DensityPlot.

    """

    def __init__(self, properties, figsize=3, fontsize=10, direction="out"):

        # Basic attributes
        self.dproperties = properties
        self.properties = list(properties.keys())

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
            warnings.filterwarnings('ignore', message='The figure layout has changed to tight')
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

    def plot_hist(self, data, colorbar=False, **args):
        """
        Create a 2d-histograms of data on all panels of the DensityPlot.
        Ex. G.plot_hist(data, bins=100, cmap='viridis')

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
        >>> G = mm.DensityPlot(properties, figsize=3)
        >>> hargs = dict(bins=100, cmap='viridis')
        >>> hist = G.plot_hist(udata, **hargs)


        """
        opts = dict()
        opts.update(args)

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

    def scatter_plot(self, data, **args):
        """
        Scatter plot on all panels of the DensityPlot.
        Ex. G.scatter_plot(data, s=0.2, color='r')

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
        >>> hist = G.scatter_plot(udata, **sargs)


        """
        # Univariate: scatter on a twin y-axis so data range is independent of PDF/density
        if getattr(self, "_univariate", False):
            ax = self.axs[0][0]
            ax_twin = ax.twinx()
            x = data[:, 0]
            y_jitter = np.random.uniform(0, 1, size=len(x))
            sargs_1d = dict(args)
            sargs_1d.setdefault("label", "sample")
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
