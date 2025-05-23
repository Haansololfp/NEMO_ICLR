from pathlib import Path
import os
import glob
import re
import numpy as np
import pandas as pd
import pickle
import argparse

# ibl specific imports
# from one.api import ONE
from celltypes_ibl.params.config import DATASETS_DIRECTORY

from deploy.iblsdsc import OneSdsc as ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader
import pdb
from datetime import datetime


def find_files(directory: str, filename: str):
    # Use os.path.join to make the search path OS independent
    # Glob recursively searches for the filename pattern in all subdirectories
    search_pattern = os.path.join(directory, "**", filename)
    # The recursive=True parameter allows searching through all subdirectories
    found_files = glob.glob(search_pattern, recursive=True)
    return found_files


def extract_id(file_path: str):
    # Use a regular expression to extract the PID from the file path
    match = re.search(r"temps([a-f0-9\-]+)/", file_path)
    return match.group(1) if match else None


def load_data_pkl(file_path: str):
    return pd.read_pickle(file_path)


def load_data_npz(file_path: str):
    with np.load(file_path, allow_pickle=True) as data:
        return pd.DataFrame(data["uuids"])


def load_templates_npz(file_path: str):
    with np.load(file_path, allow_pickle=True) as data:
        return data["raw_templates"]


def find_indices(cluster_df: pd.DataFrame, uuid_list: list[str]):
    indices = []
    for uuid in uuid_list:
        # Check if the UUID is in the DataFrame
        if uuid in cluster_df["uuids"].values:
            idx = cluster_df.index[cluster_df["uuids"] == uuid].tolist()
            indices.extend(
                idx
            )  # Assuming UUIDs are unique, idx should contain only one element
        else:
            # If the UUID is not found, raise an error
            raise ValueError(f"UUID {uuid} not found in the DataFrame.")
    return indices


def load_cluster_region_depth_fr_ssl(
    pid: str, uuids: list[str], one, ba
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spike_loader = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = spike_loader.load_spike_sorting()

    cleaned_spikes = {key.split(".")[0]: value for key, value in spikes.items()}
    cleaned_clusters = {key.split(".")[0]: value for key, value in clusters.items()}
    cleaned_channels = {key.split(".")[0]: value for key, value in channels.items()}

    clusters = spike_loader.merge_clusters(
        cleaned_spikes, cleaned_clusters, cleaned_channels
    )
    cluster_df = pd.DataFrame(clusters)

    indeces = find_indices(cluster_df, uuids)
    allen_acnm = cluster_df.loc[indeces, ["acronym"]].values.squeeze(axis=1)
    cluster_depth = cluster_df.loc[indeces, ["depths"]].values.squeeze(axis=1)
    cluster_fr = cluster_df.loc[indeces, ["firing_rate"]].values.squeeze(axis=1)

    return allen_acnm, cluster_depth, cluster_fr


parser = argparse.ArgumentParser(description="Gather IBL WVF ACG Pair Data")
parser.add_argument("-r", "--repeated_sites", action="store_true", default=False)


def main():
    args = parser.parse_args()

    acg3d_main_dir = Path("/mnt/home/hyu10/ceph/bwm_700")
    # Specify the directory to search acg in and the filename to search for
    filename = "3d_acg_data.pkl"
    # Call the function and print the results
    acg3d_paths = find_files(acg3d_main_dir, filename)

    templates_main_dir = Path("/mnt/home/cwindolf/ceph/bwm_700")
    # Specify the directory to search templates in and the filename to search for
    filename = "templates.npz"
    # Call the function and print the results
    templates_paths = find_files(templates_main_dir, filename)

    # Create a mapping from PIDs to paths for the first list
    id_to_path1 = {extract_id(path): path for path in acg3d_paths}

    # Do the same for the second list
    id_to_path2 = {extract_id(path): path for path in templates_paths}

    if not args.repeated_sites:
        # Pair files by matching PIDs
        paired_files = []
        for id, path1 in id_to_path1.items():
            path2 = id_to_path2.get(id)
            if path2:
                paired_files.append((id, path1, path2))
    else:
        # load repeated sites pids
        reproducible_ephys_pids = np.load(
            DATASETS_DIRECTORY + "reproducible_ephys_2024_03.npy"
        )
        common_pids = []
        for pid in reproducible_ephys_pids:
            if pid in id_to_path2.keys():
                common_pids.append(pid)
        paired_files = []
        for id in common_pids:
            path1 = id_to_path1.get(id)
            path2 = id_to_path2.get(id)
            if path2:
                paired_files.append((id, path1, path2))

    # Load the data from the paired files
    one = ONE()
    ba = AllenAtlas()
    br = BrainRegions()

    all_pids = []
    all_uuids = []
    all_acg3d = []
    all_templates = []
    all_cluster_regions = []
    all_clusters_depth = []
    all_cluster_fr = []

    print(len(paired_files))

    for i in range(len(paired_files)):
        pid, pkl_file, npz_file = paired_files[i]

        acg_uuids = list(load_data_pkl(pkl_file).keys())
        acg_dict = load_data_pkl(pkl_file)
        temp_uuids = list(load_data_npz(npz_file).values.squeeze())

        # Create a dictionary mapping UUIDs in temp_uuids to their indices
        uuid_to_index = {uuid: idx for idx, uuid in enumerate(temp_uuids)}

        # Find the corresponding index in temp_uuids for each UUID in acg_uuids
        matched_indices = []
        acg3d = []
        correct_extraction = True
        k = 0
        if len(acg_uuids) > 0:
            while correct_extraction and k < len(acg_uuids):
                uuid = acg_uuids[k]
                if uuid in uuid_to_index:
                    matched_indices.append((acg_uuids.index(uuid), uuid_to_index[uuid]))
                    acg3d.append(acg_dict[uuid][0])
                else:
                    print(
                        f"What? {pid} does not have the same uuids in acg and templates!"
                    )
                    correct_extraction = False
                k += 1
            if correct_extraction:
                matched_indices = np.array(matched_indices)
                acg3d = np.stack(acg3d)
            else:
                continue
        else:
            print(
                f"Oh, {pid} seems to be a really bad recording that does not have any good unit!"
            )
            continue

        templates = load_templates_npz(npz_file)[matched_indices[:, 1]]
        assert acg3d.shape[0] == templates.shape[0]

        # Load the cluster region from the spike sorting loader
        cluster_region, cluster_depth, cluster_fr = load_cluster_region_depth_fr_ssl(
            pid, acg_uuids, one, ba
        )

        # Append the neuron info to the lists
        # save pid for every neuron
        if len(acg_uuids) > 0:
            all_pids.append([pid] * len(acg_uuids))
            all_uuids.append(acg_uuids)
            all_acg3d.append(acg3d)
            all_templates.append(templates)
            all_cluster_regions.append(cluster_region)
            all_clusters_depth.append(cluster_depth)
            all_cluster_fr.append(cluster_fr)
        else:
            print(f"{pid} has no good unit!")

    all_cluster_regions = np.concatenate(all_cluster_regions)
    beryl_id = br.acronym2acronym(all_cluster_regions, mapping="Beryl")
    cosmos_id = br.acronym2acronym(all_cluster_regions, mapping="Cosmos")

    print(len(all_pids))

    all_pids = np.concatenate(all_pids)
    all_uuids = np.concatenate(all_uuids)
    all_acg3d = np.concatenate(all_acg3d)
    all_templates = np.concatenate(all_templates)
    all_clusters_depth = np.concatenate(all_clusters_depth)
    all_cluster_fr = np.concatenate(all_cluster_fr)

    print(all_pids.shape)

    # Save the data to a file
    save_dir = Path(DATASETS_DIRECTORY)
    # save file name according to the repeated_sites flag
    if not args.repeated_sites:
        save_file = (
            save_dir / f"ibl_wvf_acg_pair_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
    else:
        save_file = (
            save_dir
            / f"ibl_wvf_acg_pair_repeated_sites_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
    # save_file = save_dir / "ibl_wvf_acg_pair.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "all_pids": all_pids,
                "all_uuids": all_uuids,
                "all_acg3d": all_acg3d,
                "all_templates": all_templates,
                "all_cluster_allen": all_cluster_regions,
                "all_cluster_beryl": beryl_id,
                "all_cluster_cosmos": cosmos_id,
                "all_clusters_depth": all_clusters_depth,
                "all_cluster_fr": all_cluster_fr,
            },
            f,
        )


if __name__ == "__main__":
    main()
