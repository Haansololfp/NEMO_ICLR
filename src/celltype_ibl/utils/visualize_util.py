from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel

import numpy as np
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
import colorcet as cc
from celltype_ibl.utils.c4_data_utils import get_c4_labeled_dataset
from celltype_ibl.utils.ibl_data_util import get_ibl_wvf_acg_pairs
from celltype_ibl.utils.cell_explorer_data_util import (
    get_allen_labeled_dataset,
    get_cell_explorer_dataset,
)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from sklearn import preprocessing
import os
from PIL import Image
from scipy import stats
import pandas as pd
import pdb


def umap_visualization(
    dataset: str = "c4_labelled",
    standardize: bool = True,
    features: str = "representation",
    embedding_model_path: str | None = None,
    initialise: bool = True,
    latent_dim: int = 10,
    dot_size: int = 3,
    model: str = "contrastive",
    acg_normalization: bool = False,
    batch_norm: bool = False,
    fold_plot: list[int] = [i for i in range(10)],
    split_id: int = 1,
    training: bool = True,
    adjust_to_ce: bool = False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset == "c4_labelled":
        wvf_labeled, acg_labeled, label_idx, _, _, correspondence = (
            get_c4_labeled_dataset()
        )
        test_dataset = np.concatenate(
            (acg_labeled.reshape(-1, 1010), wvf_labeled), axis=1
        ).astype("float32")

        acg = (
            torch.from_numpy(test_dataset[:, :1010].reshape(-1, 101, 10)).to(device)
            * 10
        )
        wvf = torch.from_numpy(test_dataset[:, 1010:]).to(device)

        labels = np.array([correspondence[i] for i in label_idx])
        cm = plt.cm.tab10
    elif dataset == "ibl":
        wvf, acg, cosmos_region, fold_idx = get_ibl_wvf_acg_pairs(
            return_region="cosmos"
        )
        unique_labels, label_idx = np.unique(cosmos_region, return_inverse=True)

        keep_idx = []
        for i in fold_plot:
            keep_idx.append(np.where(fold_idx == i)[0])
        keep_idx = np.concatenate(keep_idx)

        wvf = wvf[keep_idx]
        acg = acg[keep_idx]
        cosmos_region = cosmos_region[keep_idx]

        acg = torch.from_numpy(acg.astype("float32")).to(device) * 10
        wvf = torch.from_numpy(wvf.astype("float32")).to(device)

        correspondence = {i: l for i, l in enumerate(unique_labels)}
        labels = cosmos_region
        cm = plt.cm.tab20
    elif dataset == "allen":
        wvf, acg, labels = get_allen_labeled_dataset()
        acg = torch.from_numpy(acg.astype("float32")).to(device) * 10
        wvf = torch.from_numpy(wvf.astype("float32")).to(device)
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        correspondence = {i: l for i, l in enumerate(unique_labels)}
        cm = plt.cm.tab10
    elif dataset == "cell_explorer":
        wvf, acg, labels = get_cell_explorer_dataset(padding=not adjust_to_ce)
        acg = torch.from_numpy(acg.astype("float32")).to(device) * 10
        wvf = torch.from_numpy(wvf.astype("float32")).to(device)
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        correspondence = {i: l for i, l in enumerate(unique_labels)}
        cm = plt.cm.tab10
    else:
        raise ValueError("Dataset not supported")

    if acg_normalization:
        # Calculate the maximum values along dimension 1
        max_values, _ = torch.max(acg.reshape(-1, 1010), 1)
        max_values = max_values[:, None, None]  # Reshape for broadcasting

        # Normalize acg by its maximum value and scale
        acg = acg / max_values * 10

    # if dataset == "cell_explorer":
    #     adjust_to_ce = True
    # else:
    #     adjust_to_ce = False

    if model == "contrastive":
        encode_model = BimodalEmbeddingModel(
            layer_norm=False,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
            batch_norm=batch_norm,
            adjust_to_ce=adjust_to_ce,
        )

    if initialise:
        assert embedding_model_path != None
        if os.path.exists(embedding_model_path):
            checkpoint = torch.load(embedding_model_path)
            encode_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("Model path does not exist")
    encode_model = encode_model.to(device)

    encode_model.eval()
    if features == "representation":
        wvf_rep, acg_rep = encode_model.representation(
            wvf,
            acg.reshape(-1, 1, 10, 101),
        )
        wvf_rep = wvf_rep.cpu().detach().numpy()
        acg_rep = acg_rep.cpu().detach().numpy()
    elif features == "embedding":
        wvf_rep, acg_rep = encode_model.embed(
            wvf,
            acg.reshape(-1, 1, 10, 101),
        )
        wvf_rep = wvf_rep.cpu().detach().numpy()
        acg_rep = acg_rep.cpu().detach().numpy()
    elif features == "raw":
        wvf_rep = wvf
        acg_rep = acg.reshape(-1, 1010)
    else:
        raise ValueError("Feature not supported")

    n_class = len(correspondence)

    fig, legend = umap_from_embedding(
        wvf_rep, acg_rep, labels, n_class, standardize, dot_size, features
    )

    return fig, legend


def umap_from_embedding(
    wvf_rep: np.ndarray,
    acg_rep: np.ndarray,
    labels: np.ndarray,
    n_class: int,
    standardize: bool = True,
    dot_size: int = 3,
    features: str = "representation",
) -> plt.Figure:
    if n_class < 10:
        cm = plt.cm.tab10
    else:
        cm = plt.cm.tab20
    fig, axs = plt.subplots(1, 3, figsize=[15, 5])
    reducer = umap.UMAP(random_state=42)
    if standardize:
        acg_scaler = preprocessing.StandardScaler().fit(acg_rep)
        wvf_scaler = preprocessing.StandardScaler().fit(wvf_rep)
        wvf_embedding = reducer.fit_transform(wvf_scaler.transform(wvf_rep))
        acg_embedding = reducer.fit_transform(acg_scaler.transform(acg_rep))
        acg_wvf_embedding = reducer.fit_transform(
            np.concatenate(
                (acg_scaler.transform(acg_rep), wvf_scaler.transform(wvf_rep)), axis=1
            )
        )
    else:
        wvf_embedding = reducer.fit_transform(wvf_rep)
        acg_embedding = reducer.fit_transform(acg_rep)
    # Mapping each unique region to a color
    unique_labels = np.unique(labels)

    region_to_color = dict(zip(unique_labels, cm(np.arange(len(unique_labels)))))

    for label in unique_labels:
        # Select data points belonging to the current region
        idx = labels == label
        axs[0].scatter(
            wvf_embedding[idx, 0],
            wvf_embedding[idx, 1],
            c=[region_to_color[label]],
            label=label,
            s=dot_size,
        )
        axs[1].scatter(
            acg_embedding[idx, 0],
            acg_embedding[idx, 1],
            c=[region_to_color[label]],
            label=label,
            s=dot_size,
        )
        axs[2].scatter(
            acg_wvf_embedding[idx, 0],
            acg_wvf_embedding[idx, 1],
            c=[region_to_color[label]],
            label=label,
            s=dot_size,
        )

    axs[0].set_title("wvf CLIP " + features)
    # axs[0].legend(title="label")

    axs[1].set_title("acg CLIP " + features)
    # axs[1].legend(title="label")

    axs[2].set_title("acg + wvf CLIP " + features)
    legend = axs[2].legend(title="label", bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.show()

    return fig, legend


def resize_to_square(img, size=100):
    """Resize an image to square, stretching it to fit."""
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def interpretable_2D_embedding(
    embeddings: np.ndarray,
    data: np.ndarray,
    modality: str = "wvf",
    max_samples: int = 500,
    labels: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "tab20",
    title: str | None = None,
) -> None:

    assert embeddings.shape[1] == 2, "Embeddings must be 2D"
    assert modality in ["wvf", "acg3d"], "Modality must be either wvf or acg3d"
    num_points = embeddings.shape[0]
    assert num_points == data.shape[0], "Number of embeddings must match data"
    if modality == "wvf":
        assert data.ndim == 2, "wvf must be 2D"
    else:
        assert data.ndim == 3, "Acg must be 3D"

    # Normalize embeddings to [0, 1]
    original_min = embeddings.min(axis=0)
    original_ptp = embeddings.ptp(
        axis=0
    )  # Peak to peak range (max-min) for each dimension
    embeddings = (embeddings - original_min) / original_ptp

    if labels is not None:
        _, class_labels = np.unique(labels, return_inverse=True)
    else:
        class_labels = np.zeros(num_points)

    if num_points > max_samples:
        # set random seed
        np.random.seed(42)
        indices = np.random.choice(num_points, max_samples, replace=False)
        embeddings = embeddings[indices]
        data = data[indices]
        class_labels = class_labels[indices]
        num_points = max_samples

    xmin, xmax = np.percentile(embeddings[:, 0], [1, 99])
    ymin, ymax = np.percentile(embeddings[:, 1], [1, 99])

    if cmap == "tab20":
        cmap = plt.cm.tab20
    elif cmap == "tab10":
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.get_cmap("viridis", len(np.unique(class_labels)))

    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Set the axis limits to the embedding range
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.axis("off")  # Turn off the axis

    # Add a scale bar
    # Determine an appropriate scale bar size as a proportion of the original data's units
    scale_length = 0.05  # Length of scale bar in normalized units
    real_world_length = scale_length * original_ptp[0]  # Convert back to original units
    scalebar_label = (
        f"{real_world_length:.2f}"  # Label scale bar with real world length
    )
    scalebar = AnchoredSizeBar(
        ax.transData,
        scale_length,
        scalebar_label,
        "lower right",
        pad=0.1,
        color="black",
        frameon=False,
        size_vertical=0.01,
    )
    ax.add_artist(scalebar)
    ax.add_artist(scalebar)

    # Title and labels
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Add inset plots at predefined coordinates
    for i in range(num_points):
        color = cmap(class_labels[i])

        if modality == "wvf":
            assert data[i].ndim == 1, "Waveform data must be 1D"
            # Plot waveform
            if (
                (embeddings[i, 0] < xmin)
                | (embeddings[i, 0] > xmax)
                | (embeddings[i, 1] < ymin)
                | (embeddings[i, 1] > ymax)
            ):
                continue

            inset_ax = ax.inset_axes(
                [embeddings[i, 0], embeddings[i, 1], 0.01, 0.01], transform=ax.transData
            )
            inset_ax.plot(data[i], color=color, linewidth=1)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            for spine in inset_ax.spines.values():
                spine.set_visible(False)
            inset_ax.patch.set_alpha(0)  # Make background transparent
        else:
            # Resize image data to square format by stretching
            img = resize_to_square(data[i], size=100)  # You can adjust 'size' as needed
            # Stretch the image to fit a square bounding box in the display
            image = OffsetImage(
                img, zoom=0.1
            )  # Use appropriate zoom level to fit the box
            # Ensure frames are visible by setting a distinct edge color and width
            ab = AnnotationBbox(
                image,
                (embeddings[i, 0], embeddings[i, 1]),
                frameon=True,
                boxcoords="data",
                pad=0.0,
                bboxprops={
                    "edgecolor": color,
                    "linewidth": 1,
                    "boxstyle": "square,pad=0.1",
                },
            )
            ax.add_artist(ab)
    return fig
    # # Show the final plot
    # plt.show()


# Function to calculate and plot significance matrix
def plot_significance_matrix(lr_df, unique_models, metric, ax, title):
    unique_combination = []
    for i in range(len(unique_models)):
        for j in range(i, len(unique_models)):
            unique_combination.append((unique_models[i], unique_models[j]))

    models1, models2, t_stats, p_values = [], [], [], []
    for model1, model2 in unique_combination:
        group_A = lr_df[lr_df["Model"] == model1][metric]
        group_B = lr_df[lr_df["Model"] == model2][metric]

        # pdb.set_trace()
        try:
            t, p = stats.ttest_rel(group_A, group_B)
        except:
            pdb.set_trace()

        t_stats.append(t)
        p_values.append(p)
        models1.append(model1)
        models2.append(model2)

    stats_df = pd.DataFrame(
        {"model1": models1, "model2": models2, "t_stats": t_stats, "p_values": p_values}
    )
    # print(stats_df)

    one_side_reject = []
    for i in range(len(stats_df)):
        p = stats_df["p_values"][i]
        t = stats_df["t_stats"][i]
        if p > 0.1:
            one_side_reject.append(0)
        else:
            if t > 0:
                one_side_reject.append(1)
            else:
                one_side_reject.append(-1)
    stats_df["reject"] = one_side_reject

    m = len(unique_models)
    significant_matrix = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            reject_val = stats_df.loc[
                (stats_df["model1"] == unique_models[i])
                & (stats_df["model2"] == unique_models[j]),
                "reject",
            ].values
            if reject_val:
                significant_matrix[i][j] = reject_val[0]
            else:
                significant_matrix[i][j] = 0

    # Plot significance matrix
    data = np.array(significant_matrix)
    lower_mask = np.tri(data.shape[0], data.shape[1], k=0)
    masked_data_lower = np.ma.array(data, mask=lower_mask)
    ax.imshow(masked_data_lower, cmap="RdBu_r", vmin=-2, vmax=2)

    # Set x and y tick labels
    ax.set_xticks(np.arange(len(unique_models)))
    ax.set_xticklabels(unique_models, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(unique_models)))
    ax.set_yticklabels(unique_models)
    ax.set_title(title)


def add_scale_bars(
    ax,
    x_length=2,
    y_length=2,
    x_loc=0.05,
    y_loc=0.05,
    bar_color="black",
    bar_thickness=0.02,
    label_fontsize=12,
):
    """
    Add x and y scale bars to a plot, showing a fixed length in data coordinates.

    Parameters:
    - ax: The axis to add the scale bars to.
    - x_length: The length of the x-axis scale bar (in data units, e.g., 2 units).
    - y_length: The length of the y-axis scale bar (in data units, e.g., 2 units).
    - x_loc: The relative x location for the scale bars (in axis coordinates, 0 to 1).
    - y_loc: The relative y location for the scale bars (in axis coordinates, 0 to 1).
    - bar_color: The color of the scale bars.
    - bar_thickness: The thickness of the scale bars in data units.
    - label_fontsize: Font size for the scale bar labels.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Place the scale bars in the lower left corner in data coordinates
    x_start = x_min + (x_max - x_min) * x_loc
    y_start = y_min + (y_max - y_min) * y_loc

    # Add x-axis scale bar
    ax.add_patch(
        Rectangle(
            (x_start, y_start),
            x_length,
            bar_thickness * (y_max - y_min),
            color=bar_color,
            clip_on=False,
        )
    )

    # Add y-axis scale bar
    ax.add_patch(
        Rectangle(
            (x_start, y_start),
            bar_thickness * (x_max - x_min),
            y_length,
            color=bar_color,
            clip_on=False,
        )
    )

    # Label the x and y scale bars with the value (e.g., "2 units")
    ax.text(
        x_start + x_length / 2,
        y_start - bar_thickness * (y_max - y_min) * 4,  # Slightly below the x-bar
        f"{x_length} a.u.",
        ha="center",
        va="bottom",
        fontsize=label_fontsize,
    )

    ax.text(
        x_start - bar_thickness * (x_max - x_min) * 4,
        y_start + y_length / 2,  # Slightly left of the y-bar
        f"{y_length} a.u.",
        ha="left",
        va="center",
        fontsize=label_fontsize,
        rotation="vertical",
    )
