import os
import glob
import re
from pathlib import Path
import pickle
import numpy as np
import pdb
from celltype_ibl.params.config import DATASETS_DIRECTORY
from torch.utils.data import Dataset
import torch

SAVE_DIR = DATASETS_DIRECTORY


def find_files(directory: str, filename: str):
    """
    Find files with a specific filename pattern in a directory.
    Args:
        directory (str): The directory to search in.
        filename (str): The filename pattern to search for.
    """
    # Use os.path.join to make the search path OS independent
    # Glob recursively searches for the filename pattern in all subdirectories
    search_pattern = os.path.join(directory, "**", filename)
    # The recursive=True parameter allows searching through all subdirectories
    found_files = glob.glob(search_pattern, recursive=True)
    return found_files


def extract_id(file_path: str):
    """
    Extract the PID from the file path.
    """
    match = re.search(r"temps([a-f0-9\-]+)/", file_path)
    return match.group(1) if match else None


def get_ibl_wvf_acg_pairs(
    data_dir: str = SAVE_DIR,
    return_region: str | None = None,
    with_root: bool = False,
    repeated_sites: bool = False,
    normalised: bool = True,
    return_uuids: bool = False,
    adjust_to_ce: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the IBL waveforms and ACG pairs from the pickle file.
    Args:
        data_dir (str): The directory where the data is stored.
        return_region (str): The region to return the data for. Must be one of 'allen', 'cosmos', 'beryl' or None.
        with_root (bool): Whether to include the root data.
        repeated_sites (bool): Whether to include repeated sites only.
        normalised (bool): Whether to return the normalised templates.
        return_uuids (bool): Whether to return the UUIDs.
    """
    # Define the path to the file you want to load
    save_dir = Path(data_dir)
    if not repeated_sites:
        if with_root:
            save_file = save_dir / "curated_ibl_wvf_acg_pair.pkl"
        else:
            save_file = (
                save_dir / "curated_ibl_wvf_acg_pair_wo_root_with_depth_2024-04-30.pkl"
            )
    else:
        save_file = (
            save_dir / "curated_ibl_wvf_acg_pair_wo_root_repeated_sites_2024-04-18.pkl"
        )

    # Load the data from the pickle file
    with open(save_file, "rb") as f:
        data = pickle.load(f)

    if normalised:
        curated_templates = data["curated_normalized_templates"]
    else:
        curated_templates = data["curated_templates"]

    if return_uuids:
        curated_uuids = data["curated_uuids"]

    curated_acg3d = data["curated_acg3d"]
    if not repeated_sites:
        fold_idx = np.array(data["fold_idx"])
    else:
        fold_idx = None

    curated_acg3d = curated_acg3d.reshape(-1, 10, 201)
    curated_acg3d = curated_acg3d[:, :, 100:]

    if not adjust_to_ce:
        curated_templates = curated_templates[:, 12:102]
    else:
        curated_templates = curated_templates[:, 22:63]

    if return_region is None:
        return (curated_templates, curated_acg3d, fold_idx) + (
            (curated_uuids,) if return_uuids else ()
        )
    elif return_region in ["allen", "cosmos", "beryl"]:
        curated_cluster = data.get(f"curated_cluster_{return_region}")
        return (curated_templates, curated_acg3d, curated_cluster, fold_idx) + (
            (curated_uuids,) if return_uuids else ()
        )
    else:
        raise ValueError(
            "Invalid return_region. Must be one of 'allen', 'cosmos', 'beryl' or None."
        )


def get_ibl_wvf_acg_per_depth(
    data_dir: str = SAVE_DIR, return_region: str = "Cosmos", return_depth: bool = False
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Load the IBL waveforms and ACG pairs from the pickle file.
    Args:
        data_dir (str): The directory where the data is stored.
        return_region (str): The region to return the data for. Must be one of 'allen', 'cosmos', 'beryl' or None.
        with_root (bool): Whether to include the root data.
        repeated_sites (bool): Whether to include repeated sites only.
        normalised (bool): Whether to return the normalised templates.
        return_uuids (bool): Whether to return the UUIDs.
    """
    # Define the path to the file you want to load
    save_dir = Path(data_dir)

    save_file = save_dir / "ibl_wvf_acg_pair_5neighbors_dmax60_per_depth.npz"

    # Load the data from the pickle file
    data = np.load(save_file, allow_pickle=True)

    depth_templates = data["waveforms"].reshape(-1, 5, 121)
    depth_acg3d = data["acg3d"].reshape(-1, 5, 10, 201)
    fold_idx = np.array(data["fold_idx"])
    depth = np.array(data["channel_depths"])
    pids = np.array(data["pids"])

    depth_acg3d = depth_acg3d[:, :, :, 100:]

    depth_templates = depth_templates[:, :, 12:102]

    if return_region in ["Allen", "Cosmos", "Beryl"]:
        depth_region = data.get(f"{return_region}_acnm")
        if return_depth:
            return (depth_templates, depth_acg3d, depth_region, fold_idx, depth, pids)
        else:
            return (
                depth_templates,
                depth_acg3d,
                depth_region,
                fold_idx,
            )
    else:
        raise ValueError(
            "Invalid return_region. Must be one of 'allen', 'cosmos', 'beryl' or None."
        )


def encode_ibl_training_data(
    model_path,
    vae_acg_path,
    vae_wvf_path,
    test_fold=[3, 4, 6],
    latent_dim=512,
    use_raw=False,
    model="contrastive",
    seed=42,
):
    if model == "contrastive":
        # Load the logistic regression logits
        encode_model = BimodalEmbeddingModel(
            layer_norm=False,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
        )
        checkpoint = torch.load(model_path)
        encode_model.load_state_dict(checkpoint["model_state_dict"])
    elif model == "simclr":
        simclr_wvf_patttern = SIMCLR_WVF_DIR + f"/checkpoint_acc_{seed}_epoch_*.pt"
        simclr_wvf_path = glob.glob(simclr_wvf_patttern)[0]

        simclr_acg_patttern = SIMCLR_ACG_DIR + f"/checkpoint_acc_{seed}_epoch_*.pt"
        simclr_ACG_path = glob.glob(simclr_acg_patttern)[0]

        wvf_model = SimclrEmbeddingModel(
            latent_dim=512, layer_norm=False, modality="wvf"
        )
        acg_model = SimclrEmbeddingModel(
            latent_dim=512, layer_norm=False, modality="acg"
        )

        wvf_model.load_state_dict(torch.load(simclr_wvf_path)["model_state_dict"])
        acg_model.load_state_dict(torch.load(simclr_ACG_path)["model_state_dict"])

        encode_model = BimodalEmbeddingModel(
            wvf_model.encoder, acg_model.encoder, latent_dim=512
        )

    elif model == "vae":
        acg_vae = load_acg_vae(
            vae_acg_path,
            WIN_SIZE // 2,
            BIN_SIZE,
            initialise=True,
            pool="avg",
            activation="gelu",
        )
        acg_head = VAEEncoder(acg_vae.encoder.to("cpu"), 10)  # maybe change this?

        wvf_vae = load_waveform_encoder(
            WVF_ENCODER_ARGS_SINGLE,
            vae_wvf_path,
            in_features=90,
            initialise=True,
        )
        wvf_head = VAEEncoder(wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"])

        encode_model = vae_encode_model(wvf_head, acg_head)

    encode_model.eval()

    single_wvf, single_acg, single_cosmos_region, single_fold_idx = (
        get_ibl_wvf_acg_pairs(return_region="cosmos")
    )

    training_idx = [
        index
        for index, element in enumerate(single_fold_idx)
        if element not in test_fold
    ]

    if use_raw:
        training_wvf_rep = single_wvf[training_idx].astype("float32")
        training_acg_rep = (
            single_acg[training_idx].reshape(-1, 10 * 101).astype("float32")
        )
    else:
        training_wvf = torch.tensor(single_wvf[training_idx].astype("float32"))
        training_acg = torch.tensor(single_acg[training_idx].astype("float32"))
        with torch.no_grad():
            if (model == "contrastive") | (model == "simclr"):
                training_wvf_rep, training_acg_rep = encode_model.representation(
                    training_wvf,
                    training_acg.reshape(-1, 1, 10, 101) * 10,
                )
            elif model == "vae":
                training_wvf_rep, training_acg_rep = encode_model.embed(
                    training_wvf,
                    training_acg.reshape(-1, 1, 10, 101) * 10,
                    return_pre_projection=True,
                )

            training_wvf_rep = training_wvf_rep.detach().cpu().numpy()
            training_acg_rep = training_acg_rep.detach().cpu().numpy()
    training_cosmos_region = single_cosmos_region[training_idx]

    return training_wvf_rep, training_acg_rep, training_cosmos_region


class IBL_acg_wvf_Dataset(Dataset):
    def __init__(self, WVF, ACG, y):
        self.X = np.concatenate([ACG.reshape(-1, 1010) * 10, WVF], axis=1)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
