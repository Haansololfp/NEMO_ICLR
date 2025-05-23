import pickle
import numpy as np
import pdb
from celltype_ibl.params.config import ULTRA_DATA_PATH, STIM_REMOVED_DATA_PATH


def get_ultra_wvf_acg_pairs(
    data_path: str = ULTRA_DATA_PATH,
    return_optotagged: bool = True,
    return_half: bool = True,
    return_radius: bool = False,
    load_new: bool = False,
) -> tuple[np.ndarray]:
    """
    Load the Ultra waveforms and ACG pairs from the pickle file.
    Args:
        data_path (str): The path to the pickle file.
    """
    # Load the data from the pickle file

    with open(data_path, "rb") as file:
        data = pickle.load(file)

    wavefoms = data["normalized_maxCH_waveforms"]
    acg3d = data["ACG_3D"]
    labels = data["labels"]
    optotagged = data["optotagged"]

    if load_new:
        genotype = data["genotype"]
        session_id = data["sessionID"]
    else:
        radius = data["radius"]
        session_id = data["session_id"]

    # pdb.set_trace()
    optotagged_idx = np.where(optotagged == 1)[0]
    unoptotagged_idx = np.where(optotagged == 0)[0]

    valid_acg_idx = np.where(~np.all(acg3d.reshape(-1, 2010) == 0, axis=1))[0]
    optotagged_idx = np.intersect1d(optotagged_idx, valid_acg_idx)
    unoptotagged_idx = np.intersect1d(unoptotagged_idx, valid_acg_idx)

    if return_half:
        acg3d = acg3d[:, :, 100:]

    if return_optotagged:
        if return_radius:
            return (
                wavefoms[optotagged_idx],
                acg3d[optotagged_idx],
                labels[optotagged_idx],
                session_id[optotagged_idx],
                radius[optotagged_idx],
            )
        else:
            return (
                wavefoms[optotagged_idx],
                acg3d[optotagged_idx],
                labels[optotagged_idx],
                session_id[optotagged_idx],
            )
    else:
        if load_new:
            return (
                wavefoms[unoptotagged_idx],
                acg3d[unoptotagged_idx],
                labels[unoptotagged_idx],
                session_id[unoptotagged_idx],
                genotype[unoptotagged_idx],
            )
        else:
            if return_radius:
                return (
                    wavefoms[unoptotagged_idx],
                    acg3d[unoptotagged_idx],
                    labels[unoptotagged_idx],
                    session_id[unoptotagged_idx],
                    radius[optotagged_idx],
                )
            else:
                return (
                    wavefoms[unoptotagged_idx],
                    acg3d[unoptotagged_idx],
                    labels[unoptotagged_idx],
                    session_id[unoptotagged_idx],
                )


def get_stim_removed_ultra_wvf_acg_pairs(
    data_path: str = STIM_REMOVED_DATA_PATH,
    return_optotagged: bool = True,
    return_half: bool = True,
) -> tuple[np.ndarray]:
    """
    Load the Ultra waveforms and ACG pairs from the pickle file.
    Args:
        data_path (str): The path to the pickle file.
    """
    # Load the data from the pickle file
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    genotype = data["genotype"]
    n_spikes = data["number_of_spikes"]

    selected_genotype = ["PV-Cre;Ai32", "SST-Cre;Ai32", "VIP-Cre;Ai32"]
    selected_idx = np.where(np.isin(genotype, selected_genotype) & (n_spikes > 100))[0]

    wavefoms = data["normalized_maxCH_waveforms"][selected_idx]
    acg3d = data["ACG_3D"][selected_idx]
    labels = data["labels"][selected_idx]
    optotagged = data["optotagged"][selected_idx]
    session_id = data["sessionID"][selected_idx]

    optotagged_idx = np.where(optotagged == 1)[0]
    unoptotagged_idx = np.where(optotagged == 0)[0]

    valid_acg_idx = np.where(~np.all(acg3d.reshape(-1, 2010) == 0, axis=1))[0]
    optotagged_idx = np.intersect1d(optotagged_idx, valid_acg_idx)
    unoptotagged_idx = np.intersect1d(unoptotagged_idx, valid_acg_idx)

    if return_half:
        acg3d = acg3d[:, :, 100:]

    if return_optotagged:
        return (
            wavefoms[optotagged_idx],
            acg3d[optotagged_idx],
            labels[optotagged_idx],
            session_id[optotagged_idx],
        )
    else:
        return (
            wavefoms[unoptotagged_idx],
            acg3d[unoptotagged_idx],
            labels[unoptotagged_idx],
            session_id[unoptotagged_idx],
        )
