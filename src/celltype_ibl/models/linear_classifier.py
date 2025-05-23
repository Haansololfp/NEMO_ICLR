from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import confusion_matrix
from celltype_ibl.utils.cross_session_validation import CrossSessionLeaveOneOut
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import gc
import os
from datetime import datetime
import npyx.plot as npyx_plot
from npyx.ml import set_seed
import npyx.c4.plots_functions as pf
from npyx.c4.dataset_init import BIN_SIZE, WIN_SIZE, save_results
from npyx.c4.dl_utils import (
    load_waveform_encoder,
)
from celltype_ibl.utils.c4_vae_util import VAEEncoder, vae_encode_model, load_acg_vae
from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.params.config import WVF_ENCODER_ARGS_SINGLE
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn


class vae_encode_model(nn.Module):
    def __init__(self, wvf_encoder, acg_encoder):
        super().__init__()
        self.wvf_encoder = wvf_encoder
        self.acg_encoder = acg_encoder

    def embed(self, wvf, acg, return_pre_projection=False):
        wvf_rep = self.wvf_encoder(wvf, return_pre_projection=return_pre_projection)
        acg_rep = self.acg_encoder(
            acg, return_pre_projection=return_pre_projection, isacg=True
        )
        return wvf_rep, acg_rep


def linear_classify(
    dataset: NDArray[np.float_],
    target: NDArray[np.int_],
    model: str = "logistic",
    random_state: int = 42,
) -> SGDClassifier:
    """
    linear classifier on wvf and acg embedding
    """
    if model == "logistic":
        loss = "log_loss"
    # elif model == "svm":
    #     loss = "hinge"
    else:
        raise ValueError(f"model {model} not supported")

    np.random.seed(random_state)  # can't remove this ! otherwise give different results
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, tol=1e-5, C=0.02),
    )

    clf.fit(dataset, target)

    return clf


class embedding_linear_classifier:
    def __init__(
        self,
        encode_model: BimodalEmbeddingModel | vae_encode_model,
        embedding_model: str = "contrastive",
        model: str = "logistic",
        use_final_embed: bool = False,
        device: str = "cpu",
        random_state: int = 42,
        modality: str = "both",
    ) -> None:
        self.encode_model = encode_model
        self.model = model
        self.use_final_embed = use_final_embed
        self.embedding_model = embedding_model
        self.device = device
        self.random_state = random_state
        assert modality in ["both", "acg", "wvf"]
        self.modality = modality

    def prepare_representation(self, x: np.ndarray) -> np.ndarray:
        # get representations for waveforms and acgs in higher dimension
        acg = torch.from_numpy(x[:, :1010]).to(self.device)
        wvf = torch.from_numpy(x[:, 1010:]).to(self.device)
        self.encode_model.eval()
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

        if self.modality == "both":
            return np.concatenate(
                [wvf_rep.cpu().detach().numpy(), acg_rep.cpu().detach().numpy()],
                axis=1,
            )
        elif self.modality == "acg":
            return acg_rep.cpu().detach().numpy()
        elif self.modality == "wvf":
            return wvf_rep.cpu().detach().numpy()

    def train(
        self,
        x: np.ndarray,
        target: np.ndarray,
        additional_input: None | np.ndarray = None,
    ):
        x_rep = self.prepare_representation(x)
        if additional_input is not None:
            x_rep = np.concatenate(
                [x_rep, additional_input.reshape((len(additional_input), -1))], axis=1
            )
        self.clf = linear_classify(
            x_rep, target, model=self.model, random_state=self.random_state
        )

    def predict(self, x, additional_input=None):
        x_rep = self.prepare_representation(x)
        if additional_input is not None:
            x_rep = np.concatenate(
                [x_rep, additional_input.reshape((len(additional_input), -1))], axis=1
            )
        return self.clf.predict(x_rep)


def unpack_process_fold_args(args):
    return process_fold(*args)


def process_fold(
    train_idx,
    val_idx,
    cross_seed,
    dataset,
    targets,
    additional_input,
    encode_model,
    embedding_model,
    linear_model,
    use_final_embed,
    modality,
    undersample=False,
    use_smote=False,
):
    dataset_train = dataset[train_idx]
    y_train = targets[train_idx]
    dataset_val = dataset[val_idx]
    y_val = targets[val_idx]
    if additional_input is not None:
        additional_input_train = additional_input[train_idx]
        additional_input_val = additional_input[val_idx]
    else:
        additional_input_train = None
        additional_input_val = None

    if not use_smote:
        # Data resampling
        if not undersample:
            resample = RandomOverSampler(random_state=cross_seed)
        else:
            resample = RandomUnderSampler(random_state=cross_seed)

        resample_idx, _ = resample.fit_resample(
            np.arange(len(dataset_train)).reshape(-1, 1), y_train
        )
        resample_idx = resample_idx.ravel()
        dataset_train = dataset_train[resample_idx]
        y_train = y_train[resample_idx]
        if additional_input is not None:
            additional_input_train = additional_input_train[resample_idx]
    else:
        resample = SMOTE(random_state=cross_seed)
        if additional_input is not None:
            dataset_train, y_train = resample.fit_resample(
                np.concatenate([dataset_train, additional_input_train], axis=1), y_train
            )
            dataset_train = dataset_train[:, : dataset.shape[1]]
            additional_input_train = dataset_train[:, dataset.shape[1] :]
        else:
            dataset_train, y_train = resample.fit_resample(dataset_train, y_train)

    # Model training and prediction (adapt according to your setup)
    model = embedding_linear_classifier(
        encode_model,
        embedding_model=embedding_model,
        model=linear_model,
        use_final_embed=use_final_embed,
        device="cpu",
        random_state=cross_seed,
        modality=modality,
    )
    model.train(dataset_train, y_train, additional_input_train)
    pred_target = model.predict(dataset_val, additional_input_val)

    fold_f1 = f1_score(y_val, pred_target, average="macro")
    return y_val, pred_target, fold_f1, val_idx


def cross_validate(
    dataset: NDArray[np.float_],
    targets: NDArray[np.int_],
    linear_model: str = "logistic",
    embedding_model: str = "contrastive",
    embedding_model_path: str | None = None,
    additional_input: None | NDArray[np.float_] = None,
    acg_vae_path: str | None = None,
    wvf_vae_path: str | None = None,
    pool_type: str = "avg",
    loo: bool = False,
    by_session: bool = False,
    n_runs: int = 10,
    session_idx: NDArray[np.int_] | None = None,
    save_folder: str | None = None,
    device: str = "cuda:0",
    use_final_embed: bool = False,
    layer_norm: bool = True,
    latent_dim: int = 10,
    l2_norm: bool = True,
    batch_norm: bool = False,
    activation: str = "gelu",
    use_linear_projector: bool = True,
    split_representation: bool = False,
    modality: str = "both",
    undersample: bool = False,
    use_smote: bool = False,
    seed: None | int = None,
):
    if seed is not None:
        SEED = seed
    assert embedding_model in ["VAE", "contrastive"]
    # assert loo and by_session not both True
    assert not (loo and by_session)
    if by_session:
        assert session_idx is not None

    adjust_to_ce = dataset.shape[1] == (1010 + 41)
    adjust_to_ultra = dataset.shape[1] == (1010 + 82)

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

        if adjust_to_ce:
            in_features = 41
        elif adjust_to_ultra:
            in_features = 82
        else:
            in_features = 90

        acg_vae = load_acg_vae(
            acg_vae_path,
            WIN_SIZE // 2,
            BIN_SIZE,
            initialise=True,
            pool=pool_type,
            activation=activation,
        )
        acg_head = VAEEncoder(acg_vae.encoder, 10)  # maybe change this?

        wvf_vae = load_waveform_encoder(
            WVF_ENCODER_ARGS_SINGLE,
            wvf_vae_path,
            in_features=in_features,
            initialise=True,
        )
        wvf_head = VAEEncoder(wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"])

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
            batch_norm=batch_norm,
            adjust_to_ce=adjust_to_ce,
            adjust_to_ultra=adjust_to_ultra,
        )
        checkpoint = torch.load(embedding_model_path)
        encode_model.load_state_dict(checkpoint["model_state_dict"])

    encode_model = encode_model.to(device)

    dataset = dataset.astype("float32")
    targets = targets.astype("int")

    #!!! check whether this make sense

    n_splits = len(dataset) if loo else 5

    all_runs_f1_scores = []
    all_runs_targets = []
    all_runs_predictions = []
    folds_stddev = []
    unit_idxes = []

    # Create a list to collect all fold tasks across all runs
    all_tasks = []
    for run_id in range(n_runs):
        cross_seed = SEED + np.random.randint(0, 100)
        kfold = (
            CrossSessionLeaveOneOut(session_idx)
            if by_session
            else (
                LeaveOneOut()
                if loo
                else StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=cross_seed
                )
            )
        )

        fold_args = [
            (
                train_idx,
                val_idx,
                cross_seed,
                dataset,
                targets,
                additional_input,
                encode_model,
                embedding_model,
                linear_model,
                use_final_embed,
                modality,
                undersample,
                use_smote,
            )
            for train_idx, val_idx in kfold.split(dataset, targets)
        ]
        all_tasks.extend(fold_args)

    # Process all tasks in parallel
    if len(all_tasks) > 10:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(
                tqdm(
                    executor.map(unpack_process_fold_args, all_tasks, chunksize=10),
                    total=len(all_tasks),
                )
            )
    else:
        results = [process_fold(*arg) for arg in all_tasks]

    if by_session:
        n_splits = kfold.n_splits

    # Post-process to organize results by runs
    result_index = 0
    for _ in range(n_runs):
        run_results = results[result_index : result_index + n_splits]
        result_index += n_splits
        for y_val, pred_target, fold_f1, val_idx in run_results:
            all_runs_targets.extend(y_val)
            all_runs_predictions.extend(pred_target)
            folds_stddev.append(fold_f1)
            unit_idxes.extend(val_idx)
        # all_runs_f1_scores.append(np.mean([f[2] for f in run_results]))
        all_runs_f1_scores.append([f[2] for f in run_results])

    all_runs_f1_scores = np.array(all_runs_f1_scores)
    all_targets = np.array(all_runs_targets)
    all_predictions = np.array(all_runs_predictions)

    if save_folder is not None:
        save_path = os.path.join(
            save_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"
        )
        np.save(
            save_path,
            all_predictions,
        )

    return {
        "f1_scores": all_runs_f1_scores,
        "true_targets": all_targets,
        "predicted_classes": all_predictions,
        "folds_stddev": np.array(folds_stddev),
        "indexes": np.array(unit_idxes),
    }


def plot_confusion_matrices(
    results_dict,
    save_folder,
    model_name,
    labelling,
    correspondence,
    modality,
):
    if -1 in correspondence.keys():
        del correspondence[-1]
    features_name = "3D ACGs and waveforms"
    prefix = ""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_results(results_dict, save_folder, modality)

    n_models = len(results_dict["f1_scores"])
    n_classes = len(np.unique(results_dict["predicted_classes"]))
    n_observations = results_dict["predicted_classes"].shape[0] // n_models
    predictions_matrix = np.zeros([n_models, n_observations, n_classes])
    idx_models = np.repeat(np.arange(n_models)[:, None], n_observations, axis=1)
    idx_observations = np.repeat(np.arange(n_observations)[None, :], n_models, axis=0)
    idx_models = idx_models.reshape(-1)
    idx_observations = idx_observations.reshape(-1)

    predictions_matrix[
        idx_models, idx_observations, results_dict["predicted_classes"].reshape(-1)
    ] = 1

    predictions_matrix = predictions_matrix.transpose(1, 2, 0)
    predicted_probabilities = predictions_matrix.mean(axis=2)
    true_labels = results_dict["true_targets"][:n_observations]

    if "MLI_A" in labelling.keys():
        shuffle_matrix = [4, 5, 3, 1, 2, 0]
    elif "MLI" in labelling.keys():
        shuffle_matrix = [3, 4, 1, 0, 2]
    else:
        shuffle_matrix = None

    # if "f1_scores" in results_dict:
    #     f1_scores = results_dict["f1_scores"]
    # else:
    #     f1_scores = None

    # if "folds_stddev" in results_dict:
    #     folds_stddev = results_dict["folds_stddev"]
    # else:
    #     folds_stddev = None

    # threshold = 0.0

    # ax = pf.plot_confusion_from_proba(
    #     true_labels,
    #     predicted_probabilities,
    #     correspondence,
    #     threshold=threshold,
    #     model_name=model_name,
    #     _shuffle_matrix=shuffle_matrix,
    #     axis=None,
    # )
    mean_confusion = confusion_matrix(
        results_dict["true_targets"],
        results_dict["predicted_classes"],
        normalize="true",
    )

    ax = sns.heatmap(
        mean_confusion * 100,
        annot=mean_confusion * 100,
        cmap="viridis",
        cbar=True,
        linewidths=1,
        linecolor="black",
        fmt=".3g",
        square=True,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )

    x_labels = [
        int(ax.get_xticklabels()[i].get_text())
        for i in range(len((ax.get_xticklabels())))
    ]
    y_labels = [
        int(ax.get_yticklabels()[i].get_text())
        for i in range(len(ax.get_yticklabels()))
    ]

    ax.set_xticklabels(
        pd.Series(x_labels).replace(to_replace=correspondence).to_numpy(),
        fontsize=12,
    )
    ax.set_yticklabels(
        pd.Series(y_labels).replace(to_replace=correspondence).to_numpy(),
        fontsize=12,
    )

    yl = ax.get_ylim()
    ax.plot([len(correspondence), len(correspondence)], yl, color="white", lw=4)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)

    fig = plt.gcf()

    f1_scores = f1_score(
        results_dict["true_targets"], results_dict["predicted_classes"], average="macro"
    )

    if f1_scores is not None:
        if type(f1_scores) not in [np.ndarray, list]:
            f1_string = f"F1 score: {f1_scores:.3f}"
        else:
            f1_string = f"Mean F1 score across {len(f1_scores)} runs: {np.mean(f1_scores):.3f}, std: {np.std(f1_scores):.3f}"
    # _folds_stddev = folds_stddev
    # if _folds_stddev is not None:
    #     f1_string += f"\n Stdev across folds: {_folds_stddev.mean():.3f}"
    # else:
    #     f1_string = ""

    ax.set_xlabel("Predicted label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True label", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"\nResults for {model_name} modality {modality}" + f1_string,
        fontsize=15,
        fontweight="bold",
    )

    npyx_plot.save_mpl_fig(
        fig,
        f"{prefix}{model_name}_{modality}",
        save_folder,
        "png",
    )
    plt.close()

    return os.path.join(save_folder, f"{prefix}{model_name}_{modality}.png")


def main(
    dataset: NDArray[np.float_],
    targets: NDArray[np.int_],
    LABELLING: dict,
    CORRESPONDENCE: dict,
    output_folder: str,
    additional_input: NDArray[np.float_] | None = None,
    session_idx: NDArray[np.int_] | None = None,
    embedding_model_path: str | None = None,
    acg_vae_path: str | None = None,
    wvf_vae_path: str | None = None,
    loo: bool = True,
    by_session: bool = False,
    n_runs: int = 10,
    linear_model: str = "logistic",
    embedding_model: str = "contrastive",
    use_final_embed: bool = False,
    layer_norm: bool = False,
    latent_dim: int = 10,
    l2_norm: bool = True,
    activation: str = "gelu",
    batch_norm: bool = False,
    seed=42,
    use_linear_projector: bool = True,
    split_representation: bool = False,
    modality: str = "both",
    undersample: bool = False,
    use_smote: bool = False,
) -> tuple[str, str, str]:
    global SEED
    SEED = seed
    np.random.seed(SEED)
    global N_CHANNELS
    N_CHANNELS = 10
    pool_type = "avg"
    if by_session:
        assert session_idx is not None

    # Prepare model name to save results
    suffix = ""
    cv_string = (
        "_loo_cv" if loo else "_5fold_cv"
    )  # leave-one-out or 5-fold cross-validation

    model_suffix = "_" + linear_model
    model_suffix += "_" + embedding_model
    model_suffix += "_" + activation
    model_suffix += "_femb" + str(use_final_embed)
    model_name = f"linear{model_suffix}{suffix}"

    if not os.path.exists(output_folder):
        # If the save_folder does not exist, create it
        os.makedirs(output_folder)
        print(f"'{output_folder}' did not exist and was created.")
    else:
        print(f"'{output_folder}' already exists.")

    save_folder = os.path.join(
        output_folder,
        # model_name,
        f"mouse_results{cv_string}{model_suffix}",
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # if torch.cuda.is_available():
    #     device = "cuda:0"
    device = "cpu"  # using cuda violates multiprocessing

    results_dict = cross_validate(
        dataset,
        targets,
        linear_model=linear_model,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        additional_input=additional_input,
        acg_vae_path=acg_vae_path,
        wvf_vae_path=wvf_vae_path,
        pool_type=pool_type,
        loo=loo,
        by_session=by_session,
        session_idx=session_idx,
        n_runs=n_runs,
        save_folder=save_folder,
        device=device,
        use_final_embed=use_final_embed,
        layer_norm=layer_norm,
        latent_dim=latent_dim,
        l2_norm=l2_norm,
        batch_norm=batch_norm,
        activation=activation,
        use_linear_projector=use_linear_projector,
        split_representation=split_representation,
        modality=modality,
        undersample=undersample,
        use_smote=use_smote,
        seed=SEED,
    )

    img_path = plot_confusion_matrices(
        results_dict, save_folder, model_name, LABELLING, CORRESPONDENCE, modality
    )

    today = datetime.now().strftime("%d_%b")
    # find pkl file in save_folder
    cv_path = os.path.join(save_folder, f"results_{today}_{modality}.pkl")
    return cv_path, img_path, save_folder
