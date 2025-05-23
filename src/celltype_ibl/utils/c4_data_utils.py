from npyx.c4.dataset_init import (
    N_CHANNELS,
    WAVEFORM_SAMPLES,
    extract_and_check,
    get_paths_from_dir,
    prepare_classification_dataset,
)
from torch.utils.data import Dataset
import npyx
import torch
import numpy as np
import os
import pandas as pd
from celltype_ibl.params.config import DATASETS_DIRECTORY


def get_c4_labeled_dataset(from_h5: bool = False, normalise: bool = True):
    if from_h5:
        datasets_abs = get_paths_from_dir(DATASETS_DIRECTORY)

        # Extract and check the datasets, saving a dataframe with the results
        _, dataset_class = extract_and_check(
            *datasets_abs,
            save=False,
            _labels_only=True,
            normalise_wvf=normalise,
            n_channels=N_CHANNELS,
            _extract_mli_clusters=False,
            _extract_layer=False,
        )

        # Apply quality checks and filter out granule cells
        checked_dataset = dataset_class.apply_quality_checks()
        LABELLING, CORRESPONDENCE, granule_mask = (
            checked_dataset.filter_out_granule_cells(return_mask=True)
        )
        dataset, _ = prepare_classification_dataset(
            checked_dataset,
            normalise_acgs=False,
            multi_chan_wave=False,
            process_multi_channel=False,
            _acgs_path=os.path.join(
                DATASETS_DIRECTORY, "acgs_vs_firing_rate", "acgs_3d_logscale.npy"
            ),
            _acg_mask=(~granule_mask),
            _acg_multi_factor=10,
            _n_channels=N_CHANNELS,
        )
        label = checked_dataset.labels_list

    else:
        features_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "labelled_raw_log_3d_acg_multi_wvf/features.csv",
            )
        )
        dataset = features_df.to_numpy()
        label_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "labelled_raw_log_3d_acg_multi_wvf/labels.csv",
            )
        )
        label = label_df.to_numpy().flatten()

        # filter out "GrC"
        dataset = dataset[label != "GrC"]
        label = label[label != "GrC"]

        LABELLING = {
            "PkC_cs": 4,
            "PkC_ss": 3,
            "MFB": 2,
            "MLI": 1,
            "GoC": 0,
            "unlabelled": -1,
        }
        CORRESPONDENCE = {
            4: "PkC_cs",
            3: "PkC_ss",
            2: "MFB",
            1: "MLI",
            0: "GoC",
            -1: "unlabelled",
        }
    acgs = dataset[:, :2010].reshape(-1, 10, 201)[:, :, 100:]
    if from_h5:
        acgs = acgs / 10
    label_idx = [LABELLING[i] for i in label]
    if normalise:
        waveforms = dataset[:, 2010:2100]
    else:
        templates = dataset[:, 2100:].reshape(-1, 22, 120)
        ptps = np.ptp(templates, axis=2)
        maxCH = np.argmax(ptps, axis=1)
        waveforms = templates[np.arange(len(maxCH)), maxCH, :]
    return waveforms, acgs, np.array(label_idx), label, LABELLING, CORRESPONDENCE


def get_c4_unlabelled_wvf_acg_pairs(from_h5: bool = False):
    if from_h5:
        dataset_paths = npyx.c4.get_paths_from_dir(
            DATASETS_DIRECTORY, include_hull_unlab=True
        )
        # Normalise waveforms so that the max in the dataset is 1 and the minimum is -1. Only care about shape.
        dataset_class = npyx.c4.extract_and_merge_datasets(
            *dataset_paths,
            quality_check=True,
            normalise_wvf=False,  # only applies to the multichannel waveforms
            _use_amplitudes=False,
            n_channels=N_CHANNELS,
            central_range=WAVEFORM_SAMPLES,
            labelled=False,
            flip_waveforms=True,
        )

        dataset_class.make_unlabelled_only()

        dataset_class.make_full_dataset(wf_only=True)
        wf = dataset_class.wf.reshape(-1, 10, 120)
        peak_chan = np.argmax(np.ptp(wf, axis=2), axis=1)

        peak_wf = []
        for i in range(len(wf)):
            peak_wf.append(npyx.datasets.preprocess_template(wf[i, peak_chan[i], :]))
        peak_wf = np.stack(peak_wf)

        acg_3d = np.load(
            os.path.join(
                DATASETS_DIRECTORY,
                "acgs_vs_firing_rate/unlabelled_acgs_3d_logscale.npy",
            )
        )
    else:
        features_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "unlabelled_raw_log_3d_acg_multi_wvf/features.csv",
            )
        )
        features = features_df.to_numpy()
        acg_3d = features[:, :2010]
        peak_wf = features[:, 2010:2100]

    acg_3d = acg_3d.reshape(-1, 10, 201)
    # Multiply by 10 to normalise to 100 Hz as the max.
    acg_3d = acg_3d[:, :, 100:]
    return peak_wf, acg_3d
