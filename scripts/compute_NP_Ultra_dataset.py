import numpy as np
import pandas as pd
import npyx
from npyx.c4 import fast_acg3d
from concurrent.futures import ThreadPoolExecutor, as_completed
from celltype_ibl.utils.preprocess import align_singleCH_spikes
from celltype_ibl.params.config import ULTRA_DATA_PATH

from pathlib import Path
import pickle
from tqdm import tqdm
import pdb


def make_3DACG(
    spike_trains: np.ndarray, window_size: int, bin_size: int, log_acg: bool = True
) -> np.ndarray:
    try:
        _, acg_3d = fast_acg3d(
            np.sort(spike_trains),
            window_size,
            bin_size,
        )
    except:
        # pdb.set_trace()  # IndexError:
        try:
            _, acg_3d = npyx.corr.crosscorr_vs_firing_rate(
                spike_trains,
                spike_trains,
                bin_size=bin_size,
                win_size=window_size,
            )
        except:  # IndexError:
            acg_3d = np.zeros((10, int(window_size / bin_size + 1)))

    if log_acg:
        acg_3d, _ = npyx.corr.convert_acg_log(acg_3d, bin_size, window_size)

    return acg_3d


# Parallelize the computation
def compute_acg_3d(spike_train, fs, win_size):
    return make_3DACG(np.round(spike_train * fs), win_size, bin_size=1)


def main():
    discard_idx = [251, 253, 256, 283, 511, 519, 780, 1000, 1057, 3402]

    dataset_dir = Path(ULTRA_DATA_PATH).parent

    spiketime_path = dataset_dir / "npultra_spiketimes.npy"
    spiketimes = np.load(spiketime_path, allow_pickle=True)

    waveform_path = dataset_dir / "npultra_waveforms.npy"
    waveforms = np.load(
        waveform_path, allow_pickle=True
    )  # shape (n_neurons, n_T, n_channels)

    meta_data_path = dataset_dir / "metrics.xlsx"
    df = pd.read_excel(meta_data_path, sheet_name="Sheet1")
    optotagged = df["optotagged"].values == 1
    labels = df["ct"].values
    ID = df["ID"].values

    # get the normalized max-channel waveform
    wvf_ptps = np.ptp(waveforms, axis=1)
    maxCH = np.argmax(wvf_ptps, axis=1)
    maxCH_waveforms = waveforms[np.arange(len(maxCH)), :, maxCH]

    # flipped the waveform
    positive_spikes = np.max(maxCH_waveforms, axis=1) > -1.2 * np.min(
        maxCH_waveforms, axis=1
    )
    maxCH_waveforms[positive_spikes] = -maxCH_waveforms[positive_spikes]

    aligned_spikes = align_singleCH_spikes(maxCH_waveforms, peak_T=21, peak_point="min")
    max_amplitude_idx = np.argmin(aligned_spikes, axis=1)
    normalized_waveforms = -(
        aligned_spikes
        / aligned_spikes[np.arange(len(max_amplitude_idx)), max_amplitude_idx][:, None]
    )

    keep_idx = np.arange(len(spiketimes))
    keep_idx = np.delete(keep_idx, discard_idx)

    fs = 30000
    win_size = 2000

    # get the 3d acg
    ACG_3D = []

    results_with_indices = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(compute_acg_3d, spike_train, fs, win_size): idx
            for idx, spike_train in enumerate(spiketimes)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(spiketimes),
            desc="Computing ACGs",
        ):
            original_idx = futures[future]
            result = future.result()
            results_with_indices.append((original_idx, result))

    # with ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(compute_acg_3d, spike_train, fs, win_size)
    #         for spike_train in spiketimes
    #     ]

    #     for idx, future in tqdm(
    #         enumerate(as_completed(futures)),
    #         total=len(spiketimes),
    #         desc="Computing ACGs",
    #     ):
    #         result = future.result()
    #         results_with_indices.append((idx, result))

    # Sort the results based on the original indices
    results_with_indices.sort(key=lambda x: x[0])

    # Extract the sorted results and append to ACG_3D
    ACG_3D.extend([result for _, result in results_with_indices])
    ACG_3D = np.array(ACG_3D)

    save_dir = dataset_dir
    save_path = save_dir / "npultra_wvf_acg_pair.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "waveforms": waveforms[keep_idx],
                "labels": labels[keep_idx],
                "optotagged": optotagged[keep_idx],
                "ID": ID[keep_idx],
                "normalized_maxCH_waveforms": normalized_waveforms[keep_idx],
                "ACG_3D": ACG_3D[keep_idx],
            },
            f,
        )


if __name__ == "__main__":
    main()
