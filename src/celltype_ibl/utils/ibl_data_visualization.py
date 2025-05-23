from reproducible_ephys_functions import (
    filter_recordings,
    labs,
    BRAIN_REGIONS,
    query,
    get_insertions,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fig_ephysfeatures.ephysfeatures_functions import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from iblatlas.regions import BrainRegions
from fig_ephysfeatures.ephysfeatures_load_data import load_dataframe
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from permutation_test import permut_test, distribution_dist_approx_max
from statsmodels.stats.multitest import multipletests
import pdb

import figrid as fg

import matplotlib.pyplot as plt
import pickle
from reproducible_ephys_functions import (
    figure_style,
    filter_recordings,
    # save_figure_path,
    labs,
)
from deploy.iblsdsc import OneSdsc as ONE
import numpy as np

br = BrainRegions()
# lab_number_map, institution_map, lab_colors = labs()


def panel_probe_neurons(
    fig,
    ax,
    df_filt,
    lab_colors,
    plot_feature="dim0",
    boundary_align="DG-TH",
    ylim=[-2000, 2000],
):
    # scatter plot of neurons on the probe
    df_chns = load_dataframe(df_name="chns")
    df_clust = load_dataframe(df_name="clust")
    cmap = plt.get_cmap("tab20")
    indices = np.arange(0, 20, 2).tolist() + np.arange(1, 20, 2).tolist()

    # colors = np.array(['dodgerblue',
    #       'greenyellow', 
    #       'hotpink',#'palevioletred',
    #       'gold',
    #       'darkorange',
    #       'darkviolet',
    #       'orangered',
    #       'darkturquoise',
    #       'forestgreen',
    #       'tan',
    #       'violet',
    #       ])
    colors = np.array(['violet',
          'greenyellow', 
          'hotpink',#'palevioletred',
          'dodgerblue',
          'darkorange',
          'gold',
          'orangered',
          'darkturquoise',
          'forestgreen',
          'tan',
          'darkviolet',
          ])
    # colors = np.array([cmap(i) for i in indices[:11]])


    for iR, data in df_filt.iterrows():

        df_ch = df_chns[df_chns["pid"] == data["pid"]]
        df_clu = df_clust[df_clust["pid"] == data["pid"]]

        if len(df_ch) == 0:
            continue

        la = {}
        la["id"] = df_ch["region_id"].values
        z = df_ch["z"].values * 1e6
        boundaries, colours, regions = get_brain_boundaries(la, z)
        # pdb.set_trace()

        if boundary_align is not None:
            z_subtract = boundaries[
                np.where(np.array(regions) == boundary_align)[0][0] + 1
            ]
            z = z - z_subtract
        else:
            z_subtract = 0

        levels = [-10, 10]
        im = ax[iR].scatter(
            np.random.uniform(low=0.25, high=0.75, size=df_clu.shape[0]),
            df_clu["depths_aligned"] - z_subtract,
            c=colors[df_clu[plot_feature].values],#,:],
            edgecolors='black',
            linewidths=0.1,
            s=2,
            # cmap="tab20",#"hot",
            # vmin=levels[0],
            # vmax=levels[1],
            zorder=2,
        )
        # ax[iR].add_image(im)
        ax[iR].set_xlim(0, 1)

        # First for all regions
        region_info = br.get(df_ch["region_id"].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_colours = region_info.rgb[boundaries[1:]]

        width = ax[iR].get_xlim()[1]
        for reg, col in zip(regions, region_colours):
            height = np.abs(reg[1] - reg[0])
            color = col / 255
            ax[iR].bar(
                x=width / 2,
                height=height,
                width=width,
                color="grey",
                bottom=reg[0],
                edgecolor="w",
                alpha=0,
                zorder=0,
            )

        # Now for rep site
        region_info = br.get(df_ch["region_id_rep"].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_labels = np.c_[
            np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]
        ]
        region_labels[region_labels[:, 1] == "VISa", 1] = "PPC"
        region_colours = region_info.rgb[boundaries[1:]]
        reg_idx = np.where(np.isin(region_labels[:, 1], BRAIN_REGIONS))[0]
        #
        for i, (reg, col, lab) in enumerate(
            zip(regions, region_colours, region_labels)
        ):
            height = np.abs(reg[1] - reg[0])
            # if np.isin(i, reg_idx):
            alpha = 1
            color = col / 255
            # else:
            #     alpha = 0
            #     color = "grey"
            ax[iR].bar(
                x=width / 2,
                height=height,
                width=width,
                color=color,
                bottom=reg[0],
                edgecolor="k",
                alpha=alpha,
                zorder=1,
            )

        ax[iR].set_title(data["recording"] + 1, color=lab_colors[data["institute"]])

        if iR == 0:
            ax[iR].set(
                yticks=np.arange(ylim[0], ylim[1] + 1, 500),
                yticklabels=np.arange(ylim[0], ylim[1] + 1, 500) / 1000,
                xticks=[],
            )
            ax[iR].tick_params(axis="y")
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel("Depth relative to DG-Thalamus (mm)")
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)

    #         # Add squigly line if probe plot is cut off
    #         if np.min(z) < np.min(ylim):
    #             ax[iR].text(
    #                 ax[iR].get_xlim()[1] / 2, ylim[0] - 180, "~", fontsize=10, ha="center"
    #             )

    # Add brain regions
    # width = ax[-1].get_xlim()[1]
    # ax[-1].set(ylim=ylim)
    # ax[-1].bar(
    #     x=width / 2,
    #     height=750,
    #     width=width,
    #     color=np.array([0, 159, 172]) / 255,
    #     bottom=1250,
    #     edgecolor="k",
    #     linewidth=0,
    # )
    # ax[-1].bar(
    #     x=width / 2,
    #     height=500,
    #     width=width,
    #     color=np.array([126, 208, 75]) / 255,
    #     bottom=650,
    #     edgecolor="k",
    #     linewidth=0,
    # )
    # ax[-1].bar(
    #     x=width / 2,
    #     height=500,
    #     width=width,
    #     color=np.array([126, 208, 75]) / 255,
    #     bottom=50,
    #     edgecolor="k",
    #     linewidth=0,
    # )
    # ax[-1].bar(
    #     x=width / 2,
    #     height=900,
    #     width=width,
    #     color=np.array([255, 144, 159]) / 255,
    #     bottom=-950,
    #     edgecolor="k",
    #     linewidth=0,
    # )
    # ax[-1].bar(
    #     x=width / 2,
    #     height=950,
    #     width=width,
    #     color=np.array([255, 144, 159]) / 255,
    #     bottom=-2000,
    #     edgecolor="k",
    #     linewidth=0,
    # )
    # ax[-1].text(
    #     width / 2 + 0.1,
    #     1600,
    #     "VISa/am",
    #     rotation=90,
    #     va="center",
    #     color="w",
    #     fontweight="bold",
    #     ha="center",
    #     size=5,
    # )
    # ax[-1].text(
    #     width / 2 + 0.1,
    #     900,
    #     "CA1",
    #     rotation=90,
    #     va="center",
    #     color="w",
    #     fontweight="bold",
    #     ha="center",
    #     size=5,
    # )
    # ax[-1].text(
    #     width / 2 + 0.1,
    #     300,
    #     "DG",
    #     rotation=90,
    #     va="center",
    #     color="w",
    #     fontweight="bold",
    #     ha="center",
    #     size=5,
    # )
    # ax[-1].text(
    #     width / 2 + 0.1,
    #     -500,
    #     "LP",
    #     rotation=90,
    #     va="center",
    #     color="w",
    #     fontweight="bold",
    #     ha="center",
    #     size=5,
    # )
    # ax[-1].text(
    #     width / 2 + 0.1,
    #     -1500,
    #     "PO",
    #     rotation=90,
    #     va="center",
    #     color="w",
    #     fontweight="bold",
    #     ha="center",
    #     size=5,
    # )
    # ax[-1].set_axis_off()

    # # Add colorbar
    # axin = inset_axes(
    #     ax[-1],
    #     width="50%",
    #     height="80%",
    #     loc="lower right",
    #     borderpad=0,
    #     bbox_to_anchor=(1, 0.1, 1, 1),
    #     bbox_transform=ax[-1].transAxes,
    # )
    # cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    # cbar.ax.set_yticklabels([f"{levels[0]}", f"{levels[1]}"])
    # cbar.set_label("Firing rate (spks/s)", rotation=270, labelpad=-2)

    # # Return the list of pids used in this figure
    # return np.unique(df_filt["pid"])
