import numpy as np
import pandas as pd
from celltype_ibl.params.config import (
    ALLEN_LABELLED_PATH,
    ALLEN_UNLABELLED_PATH,
    CELL_EXPLORER_PATH,
    SPLIT_INDEX_PATH,
    INDEX_MAPPING_PATH,
)


def get_allen_labeled_dataset(padding: bool = True):
    wvf, acgs, cell_types = get_cell_explorer_dataset(Allen_only=True, padding=padding)
    return wvf, acgs, cell_types


def get_allen_unlabeled_dataset(
    path: str = ALLEN_UNLABELLED_PATH, padding: bool = True
):
    allen_unlabelled_wvf_acg = np.load(path, allow_pickle=True)
    acg_3d = allen_unlabelled_wvf_acg["acg"].astype("float32")
    wvf = allen_unlabelled_wvf_acg["waveform"].astype("float32")
    N = wvf.shape[0]
    wvf_peak_idx = np.argmax(np.abs(wvf))
    wvf = -np.sign(wvf[np.arange(N), wvf_peak_idx])[:, None] * wvf
    acgs = acg_3d.reshape(-1, 10, 201)[:, :, 100:]
    if padding:
        wvf = np.pad(wvf, ((0, 0), (10, 0)), mode="edge")
        wvf = wvf[:, :90]
    return wvf, acgs


def get_cell_explorer_dataset(
    Allen_only: bool = False,
    padding: bool = False,
    filt: bool = False,
    merge_axo: bool = True,
    merge_juxta: bool = True,
    discard_VGAT: bool = True,
    # linear_ramp: bool = True,
):
    if Allen_only:
        path = ALLEN_LABELLED_PATH
    else:
        path = CELL_EXPLORER_PATH
    labelled_wvf_acg = np.load(path, allow_pickle=True)
    acg_3d = labelled_wvf_acg["acg3d"].astype("float32")
    if filt:
        wvf = labelled_wvf_acg["wvf_filt"].astype("float32")
    else:
        wvf = labelled_wvf_acg["wvf"].astype("float32")
    N = wvf.shape[0]
    wvf_peak_idx = np.argmax(np.abs(wvf))
    wvf = -np.sign(wvf[np.arange(N), wvf_peak_idx])[:, None] * wvf
    cell_types = labelled_wvf_acg["cell_types"].astype("str")
    if merge_axo:
        cell_types[cell_types == "Axo_Axonic"] = "PV"
    if merge_juxta:
        cell_types[cell_types == "Juxta"] = "Pyramidal"
    if discard_VGAT:
        wvf = wvf[cell_types != "VGAT"]
        acg_3d = acg_3d[cell_types != "VGAT"]
        cell_types = cell_types[cell_types != "VGAT"]
    acgs = acg_3d.reshape(-1, 10, 201)[:, :, 100:]
    if padding:
        wvf = np.pad(wvf, ((0, 0), (10, 39)), mode="linear_ramp")
        wvf = wvf[:, :90]

    return wvf, acgs, cell_types


def data_load_by_split(split_id: int = 1, training: bool = True, padding: bool = False):
    assert split_id in range(1, 11)
    wvf, acgs, cell_types = get_cell_explorer_dataset(
        discard_VGAT=False, padding=padding
    )

    mapping_df = pd.read_csv(INDEX_MAPPING_PATH)
    valid_to_full_map = pd.Series(
        mapping_df["Our_Index"].values, index=mapping_df["PhysMAP_Index"]
    ).to_dict()
    if training:
        idx_path = SPLIT_INDEX_PATH + f"/trainingIndex_seed{split_id}.csv"
    else:
        idx_path = SPLIT_INDEX_PATH + f"/testingsetIndex_seed{split_id}.csv"
    idx_df = pd.read_csv(idx_path, header=None)
    # transform the training_idx using the valid_to_full_map
    transformed_idx = idx_df[0].map(valid_to_full_map)
    split_wvf = wvf[transformed_idx]
    split_acgs = acgs[transformed_idx]
    split_cell_types = cell_types[transformed_idx]
    return split_wvf, split_acgs, split_cell_types
