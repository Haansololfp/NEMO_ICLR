import pickle
import numpy as np
from functools import reduce
from celltype_ibl.params.config import KENJI_ALLEN_PATH


def get_kenji_allen_wvf_acg_pairs(
    data_path: str = KENJI_ALLEN_PATH,
    return_optotagged: bool = True,
    return_half: bool = True,
    return_id: str = "unit",
    selected_project: str = "both",
) -> tuple:
    assert return_id in ["unit", "session", "structure"]
    # assert selected_project in ["both", "visual_coding", "visual_behavior"]
    assert selected_project in [
        "both",
        "visual_behavior",
        "brain_observatory",
        "functional_connectivity",
    ]

    with open(data_path, "rb") as file:
        data = pickle.load(file)
    wavefoms = data["normalized_maxCH_waveforms"]
    acg3d = data["ACG_3D"]
    labels = data["labels"]
    # project = data["project"]
    try:
        project = data["project_by_session_type"]
    except:
        project = data["project"]

    if selected_project == "both":
        project_idx = np.arange(len(labels))
    else:
        project_idx = np.where(project == selected_project)[0]

    if return_id == "unit":
        ID = data["ID"]
    elif return_id == "session":
        ID = data["session_id"]
    else:
        ID = data["ecephys_structure"]

    optotagged_idx = np.where(labels != "untagged")[0]
    unoptotagged_idx = np.where(labels == "untagged")[0]

    valid_acg_idx = np.where(~np.all(acg3d.reshape(-1, 2010) == 0, axis=1))[0]
    optotagged_idx = reduce(
        np.intersect1d, (optotagged_idx, valid_acg_idx, project_idx)
    )  # np.intersect1d(optotagged_idx, valid_acg_idx)
    unoptotagged_idx = reduce(
        np.intersect1d, (unoptotagged_idx, valid_acg_idx, project_idx)
    )  # np.intersect1d(unoptotagged_idx, valid_acg_idx)

    if return_half:
        acg3d = acg3d[:, :, 100:]

    if return_optotagged:
        return (
            wavefoms[optotagged_idx],
            acg3d[optotagged_idx],
            labels[optotagged_idx],
            ID[optotagged_idx],
        )
    else:
        return (
            wavefoms[unoptotagged_idx],
            acg3d[unoptotagged_idx],
            labels[unoptotagged_idx],
            ID[unoptotagged_idx],
        )
