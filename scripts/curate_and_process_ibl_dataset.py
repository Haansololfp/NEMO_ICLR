import numpy as np
from pathlib import Path
import pickle
import argparse
from datetime import datetime
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
import gc
import pdb
from celltype_ibl.params.config import DATASETS_DIRECTORY

parser = argparse.ArgumentParser(description="Gather IBL WVF ACG Pair Data")
parser.add_argument("-r", "--repeated_sites", action="store_true", default=False)

args = parser.parse_args()


def balanced_stratify(all_pids):
    num_folds = 10  # Example: 5-fold split
    _, ind, weights = np.unique(all_pids, return_counts=True, return_index=True)
    weights = weights[np.argsort(ind)]

    # Calculate weights (number of data points for each recording)
    weights = list(weights)

    # Sort recordings by weights (optional, based on your specific needs)
    sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)

    # Initialize folds
    folds = [[] for _ in range(num_folds)]
    fold_weights = [0] * num_folds

    # Assign recordings to folds
    for idx in sorted_indices:
        min_fold_index = np.argmin(fold_weights)
        folds[min_fold_index].append(idx)
        fold_weights[min_fold_index] += weights[idx]

    # Now, create a mapping for each neuron to its fold.
    neuron_to_fold = [0] * sum(
        weights
    )  # Initialize a list to hold the fold index for each neuron.

    cum_n_neuron = np.cumsum(weights)
    cum_n_neuron = np.insert(cum_n_neuron, 0, 0)

    # Populate the neuron_to_fold mapping.
    for fold_index, recording_indices in enumerate(folds):
        for recording_index in recording_indices:
            for current_neuron_index in range(
                cum_n_neuron[recording_index], cum_n_neuron[recording_index + 1]
            ):
                neuron_to_fold[current_neuron_index] = fold_index
    return neuron_to_fold


parser.add_argument("-k", type=int, default=1, help="the k-neighboring neurons")
parser.add_argument(
    "--d_max",
    type=int,
    default=50,
    help="the maximum depth difference from each location",
)


def find_neighbors(depths, depth, k, d_max):
    depth_diffs = np.abs(depths - depth)
    within_range = depth_diffs <= d_max
    eligible_indices = np.where(within_range)[0]
    sorted_indices = eligible_indices[np.argsort(depth_diffs[eligible_indices])][:k]
    return sorted_indices


def load_channel_region_depth_ssl(pid, one, ba):
    spike_loader = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    _, _, channels = spike_loader.load_spike_sorting()
    cleaned_channels = {key.split(".")[0]: value for key, value in channels.items()}
    channel_depths, idx = np.unique(cleaned_channels["axial_um"], return_index=True)
    allen_acnm = cleaned_channels["acronym"][idx]
    return channel_depths, allen_acnm


def collect_depth_neighbors(
    curate_data_path: str,
    k_neighbor: int = 5,
    d_neighbor: int = 60,
) -> None:
    with open(curate_data_path, "rb") as f:
        data = pickle.load(f)
        all_pids = data["curated_pids"]
        all_clusters_depth = data["curated_cluster_depth"]
        all_waveforms = data["curated_normalized_templates"]
        all_acg3d = data["curated_acg3d"]
        all_fold_idx = data["fold_idx"]

    # all_pids = all_pids[test_fold_idx]
    # all_clusters_depth = all_clusters_depth[test_fold_idx]
    # all_waveforms = all_waveforms[test_fold_idx]
    # all_acg3d = all_acg3d[test_fold_idx]
    # all_fold_idx = np.array(all_fold_idx)[test_fold_idx]

    one = ONE()
    ba = AllenAtlas()
    br = BrainRegions()
    new_fold_idx = []
    all_channel_depths = []
    all_Allen_acnm = []
    all_Beryl_acnm = []
    all_Cosmos_acnm = []
    all_k_pids = []
    all_k_waveforms = []
    all_k_acg3d = []

    for pid in list(np.unique(all_pids)):
        pid_idx = np.where(all_pids == pid)[0]
        fold_idx = np.array(all_fold_idx)[pid_idx][0]
        channel_depths, allen_acnm = load_channel_region_depth_ssl(pid, one, ba)
        beryl_acnm = br.acronym2acronym(allen_acnm, mapping="Beryl")
        cosmos_acnm = br.acronym2acronym(allen_acnm, mapping="Cosmos")
        pid_waveforms = all_waveforms[pid_idx]
        pid_acg3d = all_acg3d[pid_idx]
        wvf_depths = np.empty((len(channel_depths), k_neighbor, 121))
        acg_depths = np.empty((len(channel_depths), k_neighbor, 10, 201))
        wvf_depths[:] = np.nan
        acg_depths[:] = np.nan
        pid_depths = all_clusters_depth[pid_idx]
        i = 0
        for depth in channel_depths:
            depth_neighbors = find_neighbors(pid_depths, depth, k_neighbor, d_neighbor)
            k_nearest_idx = depth_neighbors[slice(0, k_neighbor)]
            wvf_depths[i, 0 : len(k_nearest_idx), :] = pid_waveforms[k_nearest_idx]
            acg_depths[i, 0 : len(k_nearest_idx), :, :] = pid_acg3d[k_nearest_idx]
            i += 1
        all_channel_depths.append(channel_depths)
        all_Allen_acnm.append(allen_acnm)
        all_Beryl_acnm.append(beryl_acnm)
        all_Cosmos_acnm.append(cosmos_acnm)
        all_k_pids.append(np.repeat(pid, len(channel_depths)))
        all_k_waveforms.append(wvf_depths)
        all_k_acg3d.append(acg_depths)
        new_fold_idx.append(np.repeat(fold_idx, len(channel_depths)))
        gc.collect()

    # save the directory
    save_dir = Path(DATASETS_DIRECTORY)
    save_file = (
        save_dir
        / f"ibl_wvf_acg_pair_{k_neighbor}neighbors_dmax{d_neighbor}_per_depth.npz"
    )
    np.savez(
        save_file,
        channel_depths=np.concatenate(all_channel_depths),
        Allen_acnm=np.concatenate(all_Allen_acnm),
        Beryl_acnm=np.concatenate(all_Beryl_acnm),
        Cosmos_acnm=np.concatenate(all_Cosmos_acnm),
        pids=np.concatenate(all_k_pids),
        waveforms=np.stack(all_k_waveforms),
        acg3d=np.stack(all_k_acg3d),
        fold_idx=np.concatenate(new_fold_idx),
    )


def main():
    # Define the path to the pickle file
    save_dir = Path(DATASETS_DIRECTORY)
    if args.repeated_sites:
        load_file = (
            save_dir
            / f"ibl_wvf_acg_pair_repeated_sites_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
    else:
        load_file = save_dir / f"ibl_wvf_acg_pair_2024-04-23.pkl"

    # Load the data from the pickle file
    with open(load_file, "rb") as f:
        data = pickle.load(f)

    # Extract region information from the data and mask out the neurons in 'void'
    cosmos_region = data["all_cluster_cosmos"]
    void_mask = [
        True for i in range(len(cosmos_region))
    ]  # (cosmos_region != "void") & (cosmos_region != "root")

    # Extract the waveforms from the data and mask out the neurons with too low or too high ptps
    templates = data["all_templates"].squeeze()
    ptps = np.ptp(templates, axis=1)
    maxCH_ptp = np.max(ptps, axis=1)
    maxCH = np.argmax(ptps, axis=1)
    maxCH_templates = templates[np.arange(len(maxCH)), :, maxCH]
    ptp_mask = (maxCH_ptp > 3) & (maxCH_ptp < 40)  # can be adjusted later

    # Extract the uuids, templates, acgs, allen labels, beryl labels and cosmos labels from the data
    all_pids = data["all_pids"]
    all_templates = data["all_templates"]
    all_acg3d = data["all_acg3d"]
    all_uuids = data["all_uuids"]
    all_cluster_allen = data["all_cluster_allen"]
    all_cluster_cosmos = data["all_cluster_cosmos"]
    all_cluster_beryl = data["all_cluster_beryl"]
    all_clusters_depth = data["all_clusters_depth"]

    print(len(np.unique(all_pids)))
    print(all_templates.shape)

    # Normalize the templates
    max_amplitude_idx = np.argmax(np.abs(maxCH_templates), axis=1)
    normalized_templates = maxCH_templates / np.abs(
        maxCH_templates[np.arange(len(max_amplitude_idx)), max_amplitude_idx][:, None]
    )

    # mask out the neurons with too low or too high acg or in 'void'
    curated_pids = all_pids[void_mask & ptp_mask]
    curated_templates = all_templates[void_mask & ptp_mask, :, :]
    curated_acg3d = all_acg3d[void_mask & ptp_mask, :]
    curated_uuids = all_uuids[void_mask & ptp_mask]
    curated_cluster_allen = all_cluster_allen[void_mask & ptp_mask]
    curated_cluster_cosmos = all_cluster_cosmos[void_mask & ptp_mask]
    curated_cluster_beryl = all_cluster_beryl[void_mask & ptp_mask]
    curated_normalized_templates = normalized_templates[void_mask & ptp_mask, :]
    curated_clusters_depth = all_clusters_depth[void_mask & ptp_mask]

    fold_idx = balanced_stratify(curated_pids)
    for pid in np.unique(curated_pids):
        folds = np.array(fold_idx)[curated_pids == pid]
        assert len(np.unique(np.array(folds))) == 1

    # Save the curated data to a new pickle file
    curated_data = {
        "curated_pids": curated_pids,
        "curated_templates": maxCH_templates[void_mask & ptp_mask, :],
        "curated_templates_multi_CH": curated_templates,
        "curated_acg3d": curated_acg3d,
        "curated_uuids": curated_uuids,
        "curated_cluster_allen": curated_cluster_allen,
        "curated_cluster_cosmos": curated_cluster_cosmos,
        "curated_cluster_beryl": curated_cluster_beryl,
        "curated_normalized_templates": curated_normalized_templates,
        "curated_cluster_depth": curated_clusters_depth,
        "fold_idx": fold_idx,
    }

    # save file name according to the repeated_sites flag
    if args.repeated_sites:
        curated_save_file = (
            save_dir
            / f"curated_ibl_wvf_acg_pair_with_depth_repeated_sites_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
    else:
        curated_save_file = (
            save_dir
            / f"curated_ibl_wvf_acg_pair_with_depth_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
    with open(curated_save_file, "wb") as f:
        pickle.dump(curated_data, f)


if __name__ == "__main__":
    main()
