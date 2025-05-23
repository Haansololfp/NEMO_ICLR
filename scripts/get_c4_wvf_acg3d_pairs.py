import npyx
from npyx.c4.dataset_init import get_paths_from_dir, extract_and_check
from npyx.c4.acg_vs_firing_rate import main as acg_main
from npyx.c4.dataset_init import save_features
import pandas as pd
import npyx.feat as feat
import numpy as np
import os
from celltype_ibl.params.config import DATASETS_DIRECTORY

# import pdb

N_CHANNELS = 22

data_folder = DATASETS_DIRECTORY

# get datasets directories and eventually download them
datasets_abs = npyx.c4.get_paths_from_dir(data_folder)  # , include_hull_unlab=True)

# Extract and check the datasets, saving a dataframe with the results
dataset_df, dataset_class = extract_and_check(
    *datasets_abs,
    save=False,
    _labels_only=True,
    n_channels=N_CHANNELS,  # , labelled=False, n_channels=N_CHANNELS
)
dataset_class.make_labels_only()
# dataset_class.make_unlabelled_only()
# Do feature extraction and keep track of neurons lost in the process
quality_checked_dataset = dataset_class.apply_quality_checks()

# 2d acgs and peak waveform feature spaces
common_preprocessing = quality_checked_dataset.conformed_waveforms

# waveforms = quality_checked_dataset.wf

waveforms = []
for wf in quality_checked_dataset.wf:
    waveform = wf.reshape(N_CHANNELS, -1).ravel()
    waveforms.append(waveform)
waveforms = np.stack(waveforms, axis=0)

# pdb.set_trace()

labels = quality_checked_dataset.labels_list
lab_df = pd.DataFrame({"label": labels})
raw_wvf_single_common_preprocessing_df = pd.DataFrame(
    common_preprocessing,
    columns=[f"raw_wvf_{i}" for i in range(common_preprocessing.shape[1])],
)

raw_wvf_multi_common_preprocessing_df = pd.DataFrame(
    waveforms,
    columns=[f"raw_multiCH_{i}" for i in range(waveforms.shape[1])],
)


mouse_3d_acgs_path = DATASETS_DIRECTORY + "acgs_vs_firing_rate/acgs_3d_logscale.npy"
mouse_3d_acgs = np.load(mouse_3d_acgs_path)

mouse_3d_acgs_df = pd.DataFrame(
    mouse_3d_acgs,
    columns=[f"acg_3d_logscale_{i}" for i in range(mouse_3d_acgs.shape[1])],
)

labels = quality_checked_dataset.labels_list
lab_df = pd.DataFrame({"label": labels})

save_df = pd.concat(
    [
        lab_df,
        mouse_3d_acgs_df,
        raw_wvf_single_common_preprocessing_df,
        raw_wvf_multi_common_preprocessing_df,
    ],
    axis=1,
)

features_name = "labeled_raw_log_3d_acg_multi_wvf"
features_path = os.path.join(data_folder, features_name)
if not os.path.exists(features_path):
    os.mkdir(features_path)

features, labels = feat.prepare_classification(
    save_df, bad_idx=None, drop_cols=["label"]
)

prefix = ""

features.to_csv(os.path.join(features_path, f"{prefix}features.csv"), index=False)
labels.to_csv(os.path.join(features_path, f"{prefix}labels.csv"), index=False)
