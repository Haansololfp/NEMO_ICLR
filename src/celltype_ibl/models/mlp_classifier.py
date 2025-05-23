import contextlib
import os
import argparse
import gc
from typing import Optional, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


with contextlib.suppress(ImportError):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data as data

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with contextlib.suppress(ImportError):
    from laplace import BaseLaplace, Laplace
    from laplace.utils import KronDecomposed
with contextlib.suppress(ImportError):
    from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

import statsmodels.api as sm

# import npyx.datasets as datasets
import npyx.plot as npyx_plot_utils
from npyx.ml import set_seed

# from npyx.plot import to_hex
from npyx.c4.plots_functions import plot_collapsed_densities


# from npyx.c4 import plots_functions as pf
from npyx.c4.dataset_init import (
    BIN_SIZE,
    WIN_SIZE,
    save_results,
)
from npyx.c4.dl_utils import (
    load_waveform_encoder,
)
from npyx.c4.misc import require_advanced_deps
from npyx.c4.plots_functions import plot_confusion_from_proba

from celltype_ibl.utils.c4_vae_util import VAEEncoder, vae_encode_model, load_acg_vae
from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.params.config import (
    WVF_ENCODER_ARGS_SINGLE,
    C4_COLORS,
    REGION_COLORS,
    V1_COLORS,
)


class embedding_MLP_classifier(nn.Module):
    def __init__(
        self,
        encode_model: BimodalEmbeddingModel | vae_encode_model,
        embedding_model: str = "contrastive",
        n_classes: int = 5,
        freeze_encoder_weights: bool = True,
        use_final_embed: bool = False,
        modality: str = "both",
        additional_input_size: int = 0,
    ) -> None:
        super().__init__()
        self.encode_model = encode_model
        self.embedding_model = embedding_model
        if freeze_encoder_weights:
            for param in self.encode_model.parameters():
                param.requires_grad = False
        assert modality in ["both", "acg", "wvf"]
        self.modality = modality
        # train MLP with dropout (or not?)
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.LazyLinear(n_classes)
        self.use_final_embed = use_final_embed

        if modality == "both":
            self.batch_norm = (
                nn.BatchNorm1d(20 + additional_input_size)
                if self.use_final_embed
                else nn.BatchNorm1d(500 + additional_input_size)
            )
        elif modality == "acg":
            self.batch_norm = (
                nn.BatchNorm1d(10 + additional_input_size)
                if self.use_final_embed
                else nn.BatchNorm1d(200 + additional_input_size)
            )
        else:
            self.batch_norm = (
                nn.BatchNorm1d(10 + additional_input_size)
                if self.use_final_embed
                else nn.BatchNorm1d(300 + additional_input_size)
            )
        self.additional_input_size = additional_input_size

    def forward(
        self,
        x: torch.FloatTensor | torch.cuda.FloatTensor,
    ) -> torch.FloatTensor | torch.cuda.FloatTensor:
        acg = x[:, :1010]
        if self.additional_input_size > 0:
            additional_input = x[:, -self.additional_input_size :]
            wvf = x[:, 1010 : -self.additional_input_size]
        else:
            wvf = x[:, 1010:]
            additional_input = None
        if self.embedding_model == "contrastive":
            if self.use_final_embed:
                wvf_rep, acg_rep = self.encode_model.embed(
                    wvf, acg.reshape(-1, 1, 10, 101)
                )
            else:
                wvf_rep, acg_rep = self.encode_model.representation(
                    wvf, acg.reshape(-1, 1, 10, 101)
                )
        elif self.embedding_model == "VAE":
            wvf_rep, acg_rep = self.encode_model.embed(
                wvf,
                acg.reshape(-1, 1, 10, 101),
                return_pre_projection=not self.use_final_embed,
            )
        else:
            raise ValueError("Unknown embedding model")
        if self.modality == "both":
            x = torch.cat([wvf_rep, acg_rep], dim=1)
        elif self.modality == "acg":
            x = acg_rep
        else:
            x = wvf_rep
        if additional_input is not None:
            x = torch.cat(
                [x, additional_input.reshape((len(additional_input), -1))], dim=1
            )
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))

        x = (
            self.fc2(
                torch.cat(
                    (x, additional_input.reshape((len(additional_input), -1))), dim=1
                )
            )
            if additional_input is not None
            else self.fc2(x)
        )
        # x = self.fc2(x)
        return x


class CustomDataset(data.Dataset):
    """Dataset of waveforms and 3D acgs. Every batch will have shape:
    (batch_size, WAVEFORM_SAMPLES * ACG_3D_BINS * ACG_3D_LEN))"""

    def __init__(
        self,
        data,
        targets,
        multi_chan_wave=False,
    ):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            targets (string): Array of labels for the provided data
        """
        self.data = data
        self.targets = targets
        self.multi_chan_wave = multi_chan_wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = self.data[idx, :].astype("float32")
        target = self.targets[idx].astype("int")

        # acg = data_point[:1010]
        # waveform = data_point[1010:]
        # data_point = np.concatenate((acg.ravel(), waveform)).astype("float32")
        return data_point, target


def plot_training_curves(train_losses, f1_train, epochs, save_folder=None):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    axes[0].plot(train_losses.mean(0), label="Mean training loss")
    axes[0].fill_between(
        range(epochs),
        train_losses.mean(0) + train_losses.std(0),
        train_losses.mean(0) - train_losses.std(0),
        facecolor="blue",
        alpha=0.2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper left")

    axes[1].plot(f1_train.mean(0), label="Mean training F1")
    axes[1].fill_between(
        range(epochs),
        f1_train.mean(0) + f1_train.std(0),
        f1_train.mean(0) - f1_train.std(0),
        facecolor="blue",
        alpha=0.2,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].legend(loc="upper left")

    fig.savefig(os.path.join(save_folder, "training_curves.png"))
    plt.close("all")


@require_advanced_deps("torch", "torchvision", "laplace")
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0]


@require_advanced_deps("torch", "torchvision", "laplace")
def train(
    model,
    iterator,
    optimizer,
    criterion,
    device,
):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    for batch in iterator:
        x = batch[0].to(device)
        y = batch[1].to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        # measure the loss
        loss = criterion(y_pred, y)

        # calculate the backward pass
        loss.backward()

        # updates the weights based on the backward pass
        optimizer.step()

        # compute performance
        with torch.no_grad():
            f1 = f1_score(
                y.cpu(),
                y_pred.cpu().argmax(1),
                labels=np.unique(y.cpu().numpy()),
                average="macro",
                zero_division=1,
            )

            acc = calculate_accuracy(y_pred, y)

        # store performance for this minibatch
        epoch_loss += loss.item()
        epoch_f1 += f1.item()
        epoch_acc += acc.item()

    return (
        epoch_loss / len(iterator),
        epoch_f1 / len(iterator),
        epoch_acc / len(iterator),
    )


@require_advanced_deps("torch", "torchvision", "laplace")
def get_kronecker_hessian_attributes(*kronecker_hessians: KronDecomposed):
    hessians = []
    for h in kronecker_hessians:
        hess_dict = {
            "eigenvalues": h.eigenvalues,
            "eigenvectors": h.eigenvectors,
            "deltas": h.deltas,
            "damping": h.damping,
        }
        hessians.append(hess_dict)
    return hessians


@require_advanced_deps("torch", "torchvision", "laplace")
def get_model_probabilities(
    model: torch.nn.Module,
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    device: torch.device = torch.device("cpu"),
    laplace: bool = True,
    enforce_layer: bool = False,
    labelling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[BaseLaplace]]:
    """
    Computes the probabilities of a given model for a test dataset, with or without Laplace approximation calibration.

    Args:
    - model: a PyTorch model.
    - train_loader: a PyTorch DataLoader for the training dataset.
    - test_loader: a PyTorch DataLoader for the test dataset.
    - device: a PyTorch device to run the computations on.
    - laplace: a boolean indicating whether to use Laplace approximation calibration or not. Default is True.
    - enforce_layer: a boolean indicating whether to enforce layer correction or not. Default is False.
    - labelling: a dictionary containing the labels for each cell type. Required if enforce_layer is True. Default is None.

    Returns:
    - probs_normal: a PyTorch tensor containing the uncalibrated probabilities for the test dataset.
    - probs_laplace: a PyTorch tensor containing the calibrated probabilities for the test dataset, if laplace is True. Otherwise, it is the same as probs_normal.
    - la: a Laplace object containing the fitted Laplace approximation, if laplace is True. Otherwise, it is None.
    """

    if enforce_layer:
        assert labelling is not None, "Labelling must be provided if enforcing layer"

    model.eval().to(device)
    # First get uncalibrated probabilities
    probs_normal = []
    with torch.no_grad():
        for x, _ in test_loader:
            model_uncalibrated_probabilities = torch.softmax(
                model(x.float().to(device)), dim=-1
            )
            # if enforce_layer:
            #     model_uncalibrated_probabilities = layer_correction(
            #         model_uncalibrated_probabilities, x[:, -4:], labelling
            #     )
            probs_normal.append(model_uncalibrated_probabilities)
    if not laplace:
        return torch.cat(probs_normal).cpu(), torch.cat(probs_normal).cpu(), None

    # Then fit Laplace approximation
    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")

    # Finally get adjusted probabilities
    probs_laplace = []
    with torch.no_grad():
        for x, _ in test_loader:
            model_calibrated_probabilities = la(x.float().to(device))
            probs_laplace.append(model_calibrated_probabilities)

    return torch.cat(probs_normal).cpu(), torch.cat(probs_laplace).cpu(), la


@require_advanced_deps("torch", "torchvision", "laplace")
def cross_validate(
    dataset: NDArray[np.float_],
    targets: NDArray[np.int_],
    embedding_model: str = "contrastive",
    embedding_model_path: str | None = None,
    additional_input: None | NDArray[np.float_] = None,
    acg_vae_path: str | None = None,
    wvf_vae_path: str | None = None,
    pool_type: str = "avg",
    loo: bool = False,
    n_runs: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    save_folder: str | None = None,
    use_final_embed: bool = False,
    layer_norm: bool = True,
    latent_dim: int = 10,
    l2_norm: bool = True,
    activation: str = "gelu",
    use_linear_projector: bool = True,
    split_representation: bool = False,
    freeze_encoder_weights: bool = True,
    adjust_to_ce: bool = False,
    adjust_to_ultra: bool = True,
    modality: str = "both",
    initialise: bool = True,
):  ###change the code to accomodate to the extra input
    n_splits = len(dataset) if loo else 5
    n_classes = len(np.unique(targets))

    train_losses = np.zeros((n_splits * n_runs, epochs))
    f1_train = np.zeros((n_splits * n_runs, epochs))
    acc_train = np.zeros((n_splits * n_runs, epochs))

    all_runs_f1_scores = []
    all_runs_targets = []
    all_runs_predictions = []
    all_runs_probabilities = []
    folds_stddev = []
    unit_idxes = []

    total_runs = 0

    set_seed(SEED, seed_torch=True)

    if additional_input is not None:
        dataset = np.concatenate(
            (dataset, additional_input.reshape((len(additional_input), -1))), axis=1
        )

    for _ in tqdm(range(n_runs), desc="Cross-validation run", position=0, leave=True):
        run_true_targets = []
        run_model_pred = []
        run_probabilities = []
        run_unit_idxes = []
        folds_f1 = []

        cross_seed = 42 + np.random.randint(0, 100)  # SEED + np.random.randint(0, 100)
        kfold = (
            LeaveOneOut()
            if loo
            else StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=cross_seed
            )
        )

        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(dataset, targets)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=kfold.get_n_splits(dataset),
        ):
            dataset_train = dataset[train_idx]
            y_train = targets[train_idx]

            dataset_val = dataset[val_idx]
            y_val = targets[val_idx]

            oversample = RandomOverSampler(random_state=cross_seed)
            resample_idx, _ = oversample.fit_resample(
                np.arange(len(dataset_train)).reshape(-1, 1), y_train
            )
            resample_idx = resample_idx.ravel()
            dataset_train = dataset_train[resample_idx]
            y_train = y_train[resample_idx]

            train_iterator = data.DataLoader(
                CustomDataset(
                    dataset_train,
                    y_train,
                    multi_chan_wave=False,
                ),
                shuffle=True,
                batch_size=batch_size,
                num_workers=1,
                # num_workers=4,
            )

            val_iterator = data.DataLoader(
                CustomDataset(
                    dataset_val,
                    y_val,
                    multi_chan_wave=False,
                ),
                batch_size=len(dataset_val),
            )

            if embedding_model == "VAE":
                assert acg_vae_path is not None
                assert wvf_vae_path is not None
                if not os.path.exists(acg_vae_path):
                    # If the path does not exist, raise an error
                    raise FileNotFoundError(
                        f"The required path '{acg_vae_path}' does not exist."
                    )
                if not os.path.exists(wvf_vae_path):
                    # If the path does not exist, raise an error
                    raise FileNotFoundError(
                        f"The required path '{wvf_vae_path}' does not exist."
                    )

                acg_vae = load_acg_vae(
                    acg_vae_path,
                    WIN_SIZE // 2,
                    BIN_SIZE,
                    # initialise=False,
                    initialise=initialise,
                    pool=pool_type,
                    activation=activation,
                )
                acg_vae.encoder
                acg_head = VAEEncoder(acg_vae.encoder, 10)  # maybe change this?

                if adjust_to_ce:
                    in_features = 41
                elif adjust_to_ultra:
                    in_features = 82
                else:
                    in_features = 90

                wvf_vae = load_waveform_encoder(
                    WVF_ENCODER_ARGS_SINGLE,
                    wvf_vae_path,
                    in_features=in_features,
                    initialise=initialise,
                    # initialise=False,
                )
                wvf_head = VAEEncoder(
                    wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"]
                )

                encode_model = vae_encode_model(wvf_head, acg_head)

            elif embedding_model == "contrastive":
                assert embedding_model_path is not None
                if not os.path.exists(embedding_model_path):
                    # If the path does not exist, raise an error
                    raise FileNotFoundError(
                        f"The required path '{embedding_model_path}' does not exist."
                    )

                encode_model = BimodalEmbeddingModel(
                    layer_norm=layer_norm,
                    latent_dim=latent_dim,
                    l2_norm=l2_norm,
                    activation=activation,
                    adjust_to_ultra=adjust_to_ultra,
                )
                checkpoint = torch.load(embedding_model_path)
                encode_model.load_state_dict(checkpoint["model_state_dict"])

            model = embedding_MLP_classifier(
                encode_model,
                embedding_model,
                n_classes,
                freeze_encoder_weights,
                use_final_embed,
                modality,
                additional_input_size=(
                    additional_input.shape[1] if additional_input is not None else 0
                ),
            )

            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, epochs, 1, last_epoch=-1
            )

            criterion = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            criterion = criterion.to(device)

            for epoch in tqdm(range(epochs), position=2, leave=False, desc="Epochs"):
                train_loss, train_f1, train_acc = train(
                    model,
                    train_iterator,
                    optimizer,
                    criterion,
                    device,
                )

                train_losses[total_runs, epoch] = train_loss
                acc_train[total_runs, epoch] = train_acc
                f1_train[total_runs, epoch] = train_f1
                scheduler.step()

            total_runs += 1

            # Append results
            _, prob_calibrated, model_calibrated = get_model_probabilities(
                model,
                train_iterator,
                val_iterator,
                torch.device("cpu"),
                laplace=True,
            )
            run_true_targets.append(y_val)
            run_model_pred.append(prob_calibrated.argmax(1))
            run_probabilities.append(prob_calibrated)

            fold_f1 = f1_score(y_val, prob_calibrated.argmax(1), average="macro")
            folds_f1.append(fold_f1)
            unit_idxes.append(val_idx)
            run_unit_idxes.append(val_idx)

            del model
            del train_iterator
            del val_iterator
            del model_calibrated
            torch.cuda.empty_cache()
            gc.collect()

        run_unit_idxes = np.concatenate(run_unit_idxes).squeeze()

        # sort arrays using run_unit_idxes to restore original order
        sort_idx = np.argsort(run_unit_idxes)

        run_model_pred = np.concatenate(run_model_pred).squeeze()[sort_idx]
        run_true_targets = np.concatenate(run_true_targets).squeeze()[sort_idx]

        run_f1 = f1_score(run_true_targets, run_model_pred, average="macro")
        all_runs_f1_scores.append(run_f1)
        all_runs_targets.append(np.array(run_true_targets))
        all_runs_predictions.append(np.array(run_model_pred))
        all_runs_probabilities.append(
            np.concatenate(run_probabilities, axis=0)[sort_idx]
        )
        folds_stddev.append(np.array(folds_f1).std())

    plot_training_curves(train_losses, f1_train, epochs, save_folder=save_folder)

    all_targets = np.concatenate(all_runs_targets).squeeze()
    raw_probabilities = np.stack(all_runs_probabilities, axis=2)

    if save_folder is not None:
        np.save(
            os.path.join(
                save_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"
            ),
            raw_probabilities,
        )

    all_probabilities = np.concatenate(all_runs_probabilities).squeeze()

    return {
        "f1_scores": all_runs_f1_scores,
        # "train_accuracies": run_train_accuracies,
        "true_targets": all_targets,
        "predicted_probability": all_probabilities,
        "folds_stddev": np.array(folds_stddev),
        "indexes": np.concatenate(unit_idxes).squeeze(),
    }


def plot_confusion_matrices(
    results_dict,
    save_folder,
    model_name,
    labelling,
    correspondence,
    plots_prefix="",
    modality="both",
):
    if -1 in correspondence.keys():
        del correspondence[-1]
    if modality == "both":
        features_name = "3D ACGs and waveforms"
    elif modality == "wvf":
        features_name = "waveforms"
    else:
        features_name = "3D ACGs"
    prefix = ""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_results(results_dict, save_folder, modality)

    # if loo:
    n_models = len(results_dict["f1_scores"])
    n_classes = results_dict["predicted_probability"].shape[1]
    n_observations = results_dict["predicted_probability"].shape[0] // n_models
    predictions_matrix = results_dict["predicted_probability"].reshape(
        (n_models, n_observations, n_classes)
    )

    predictions_matrix = predictions_matrix.transpose(1, 2, 0)
    predicted_probabilities = predictions_matrix.mean(axis=2)
    true_labels = results_dict["true_targets"][:n_observations]
    # else:
    #     true_labels = results_dict["true_targets"]
    #     predicted_probabilities = results_dict["predicted_probability"]

    if "MLI_A" in labelling.keys():
        shuffle_matrix = [4, 5, 3, 1, 2, 0]
    elif "MLI" in labelling.keys():
        shuffle_matrix = [3, 4, 1, 0, 2]
    else:
        shuffle_matrix = None

    for threshold in tqdm(
        list(np.arange(0.4, 1, 0.1)) + [0.0], desc="Saving results figures"
    ):
        threshold = round(threshold, 2)
        fig = plot_results_from_threshold(
            true_labels,
            predicted_probabilities,
            correspondence,
            threshold,
            f"{' '.join(model_name.split('_')).title()} {plots_prefix}({features_name})",
            collapse_classes=False,
            _shuffle_matrix=shuffle_matrix,
            f1_scores=(
                results_dict["f1_scores"] if "f1_scores" in results_dict else None
            ),
            _folds_stddev=(
                results_dict["folds_stddev"] if "folds_stddev" in results_dict else None
            ),
        )
        npyx_plot_utils.save_mpl_fig(
            fig, f"{prefix}{model_name}_at_threshold_{threshold}", save_folder, "pdf"
        )
        plt.close()


def plot_results_from_threshold(
    true_targets: np.ndarray,
    predicted_proba: np.ndarray,
    correspondence: dict,
    threshold: float = 0.0,
    model_name: str = "model",
    kde_bandwidth: float = 0.02,
    collapse_classes: bool = False,
    f1_scores: np.ndarray = None,
    _shuffle_matrix: list = None,
    _folds_stddev: np.ndarray = None,
):
    fig, ax = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    if "MLI" in correspondence.values():
        colors_dict = C4_COLORS
    elif "HPF" in correspondence.values():
        colors_dict = REGION_COLORS
    elif "PV" in correspondence.values():
        colors_dict = V1_COLORS

    if collapse_classes:
        all_true_positives = []
        all_false_positives = []

    for label in range(len(correspondence.keys())):
        cell_type = correspondence[label]
        col = npyx_plot_utils.to_hex(colors_dict[cell_type])
        predictions = np.argmax(predicted_proba, axis=1)

        predicted_label_mask = predictions == label
        true_label_mask = true_targets == label

        true_positive_p = predicted_proba[predicted_label_mask & true_label_mask, label]
        false_positive_p = predicted_proba[
            predicted_label_mask & (~true_label_mask), label
        ]

        if collapse_classes:
            all_true_positives.append(true_positive_p)
            all_false_positives.append(false_positive_p)
            continue

        skip_tp, skip_fp = False, False
        try:
            density_correct_factor_tp = (
                len(true_positive_p) + len(false_positive_p)
            ) / len(true_positive_p)
        except ZeroDivisionError:
            skip_tp = True

        try:
            density_correct_factor_fp = (
                len(true_positive_p) + len(false_positive_p)
            ) / len(false_positive_p)
        except ZeroDivisionError:
            skip_fp = True

        if not skip_tp:
            kde = sm.nonparametric.KDEUnivariate(true_positive_p)
            kde.fit(bw=kde_bandwidth)  # Estimate the densities
            ax[0].fill_between(
                kde.support,
                kde.density * 0,
                kde.density / 100 / density_correct_factor_tp,
                label=f"Pr(f(x)={cell_type}|{cell_type})",
                facecolor=col,
                lw=2,
                alpha=0.5,
            )

        if not skip_fp:
            kde = sm.nonparametric.KDEUnivariate(false_positive_p)
            kde.fit(bw=kde_bandwidth)  # Estimate the densities
            ax[0].plot(
                kde.support,
                kde.density / 100 / density_correct_factor_fp,
                label=f"Pr(f(x)={cell_type}|¬{cell_type})",
                color=col,
                lw=2,
                alpha=0.8,
            )

    if collapse_classes:
        plot_collapsed_densities(
            all_true_positives, all_false_positives, kde_bandwidth, ax
        )
    ax[0].set_xlim([0.2, 1])
    ax[0].set_ylim([0, 0.1])
    yl = ax[0].get_ylim()
    ax[0].plot([threshold, threshold], yl, color="red", lw=3, ls="-")
    ax[0].legend(loc="upper left", fontsize=12)
    ax[0].set_xlabel("Predicted probability", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Density", fontsize=14, fontweight="bold")

    ax[1] = plot_confusion_from_proba(
        true_targets,
        predicted_proba,
        correspondence,
        threshold,
        model_name,
        axis=ax[1],
        _shuffle_matrix=_shuffle_matrix,
    )

    # ax[1] = plot_confusion_matrix(
    #     predicted_proba,
    #     true_targets,
    #     correspondence,
    #     confidence_threshold=threshold,
    #     label_order=_shuffle_matrix,
    #     normalize=True,
    #     axis=ax[1],
    #     saveDir=None,
    #     saveFig=False,
    # )

    if f1_scores is not None:
        if type(f1_scores) not in [np.ndarray, list]:
            f1_string = f"F1 score: {f1_scores:.3f}"
        else:
            f1_string = f"Mean F1 score across {len(f1_scores)} runs: {np.mean(f1_scores):.3f}, std: {np.std(f1_scores):.3f}"
        if _folds_stddev is not None:
            f1_string += f"\n Stdev across folds: {_folds_stddev.mean():.3f}"
    else:
        f1_string = ""

    ax[1].set_xlabel("Predicted label", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("True label", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"\nResults for {model_name}, threshold = {threshold}\n" + f1_string,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


@require_advanced_deps("torch", "torchvision", "laplace")
def main(
    dataset: NDArray[np.float_],
    targets: NDArray[np.int_],
    LABELLING: dict,
    CORRESPONDENCE: dict,
    output_folder: str,
    additional_input: NDArray[np.float_] | None = None,
    embedding_model_path: str | None = None,
    acg_vae_path: str | None = None,
    wvf_vae_path: str | None = None,
    loo: bool = True,
    embedding_model: str = "contrastive",
    use_final_embed: bool = False,
    layer_norm: bool = False,
    latent_dim: int = 10,
    l2_norm: bool = True,
    activation: str = "gelu",
    use_linear_projector: bool = True,
    split_representation: bool = False,
    freeze_encoder_weights=True,
    adjust_to_ce: bool = False,
    adjust_to_ultra: bool = True,
    modality: str = "both",
    seed=42,
    initialise: bool = True,
) -> None:
    """
    Runs a deep semi-supervised classifier on the C4 ground-truth datasets,
    training it on mouse opto-tagged data and testing it on expert-labelled monkey neurons.

    Args:
        data_folder: The path to the folder containing the datasets.
        freeze_vae_weights: Whether to freeze the pretrained weights of the VAE.
        VAE_random_init: Whether to randomly initialize the VAE weights that were pretrained.
        augment_acg: Whether to augment the ACGs.
        augment_wvf: Whether to augment the waveforms.
        mli_clustering: Whether to cluster the MLI cells.
        use_layer: Whether to use layer information.
        loo: Whether to use leave-one-out cross-validation.
        multi_chan_wave: Whether to use multi-channel waveforms.

    Returns:
        None
    """
    global SEED
    SEED = seed
    global N_CHANNELS
    N_CHANNELS = 10
    pool_type = "avg"

    np.random.seed(SEED)

    # Prepare model name to save results
    suffix = ""
    model_suffix = ""
    cv_string = (
        "_loo_cv" if loo else "_5fold_cv"
    )  # leave-one-out or 5-fold cross-validation

    model_suffix += "_" + embedding_model
    model_suffix += "_" + activation
    model_suffix += "_femb" + str(use_final_embed)
    model_suffix += "_frozen" if freeze_encoder_weights else ""
    model_name = f"mlp{model_suffix}{suffix}"
    features_suffix = ""

    if not os.path.exists(output_folder):
        # If the save_folder does not exist, create it
        os.makedirs(output_folder)
        print(f"'{output_folder}' did not exist and was created.")
    else:
        print(f"'{output_folder}' already exists.")

    save_folder = os.path.join(
        output_folder,
        # model_name,
        f"mouse_results{cv_string}{model_suffix}{modality}",
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    results_dict = cross_validate(
        dataset,
        targets,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        additional_input=additional_input,
        acg_vae_path=acg_vae_path,
        wvf_vae_path=wvf_vae_path,
        pool_type=pool_type,
        loo=loo,
        save_folder=save_folder,
        use_final_embed=use_final_embed,
        layer_norm=layer_norm,
        latent_dim=latent_dim,
        l2_norm=l2_norm,
        activation=activation,
        use_linear_projector=use_linear_projector,
        split_representation=split_representation,
        freeze_encoder_weights=freeze_encoder_weights,
        adjust_to_ce=adjust_to_ce,
        adjust_to_ultra=adjust_to_ultra,
        modality=modality,
        initialise=initialise,
    )

    plot_confusion_matrices(
        results_dict,
        save_folder,
        model_name,
        labelling=LABELLING,
        correspondence=CORRESPONDENCE,
        modality=modality,
    )
