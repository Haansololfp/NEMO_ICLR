# from one.api import ONE
from deploy.iblsdsc import OneSdsc as ONE

from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
import matplotlib.pyplot as plt
import npyx
from npyx.c4 import fast_acg3d
import argparse
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
import pandas as pd

import numpy as np
from one.api import ONE

BIN_SIZE: int = 1


def make_3DACG(
    spike_trains: np.ndarray, window_size: int, bin_size: int, log_acg: bool = True
) -> np.ndarray:
    try:
        _, acg_3d = fast_acg3d(
            spike_trains,
            bin_size,
            window_size,
        )
    except IndexError:
        try:
            _, acg_3d = npyx.corr.crosscorr_vs_firing_rate(
                spike_trains,
                spike_trains,
                bin_size=bin_size,
                win_size=window_size,
            )
        except IndexError:
            acg_3d = np.zeros((10, int(window_size / bin_size + 1)))

    if log_acg:
        acg_3d, _ = npyx.corr.convert_acg_log(acg_3d, bin_size, window_size)

    return acg_3d


def compute_3DACG_IBL(
    pid: str,
    allsyms_dir: Path,
    data_dir: Path,
    overwrite: bool = False,
    retry_err: bool = True,
    log_acg: bool = True,
    summarize_errors: bool = False,
) -> None:
    data_dir = Path(data_dir)  # Convert data_dir to a Path object
    symlink_dir = allsyms_dir / f"syms{pid}"
    temps_dir = data_dir / f"temps{pid}"
    if overwrite and temps_dir.exists():
        shutil.rmtree(symlink_dir)
        shutil.rmtree(temps_dir)

    done = temps_dir.exists() and (temps_dir / "3d_acg_data_2_3.pkl").exists()
    if done:
        print("already done")
        return

    had_err = temps_dir.exists() and (temps_dir / "error.pkl").exists()
    if had_err:
        with open(temps_dir / "error.pkl", "rb") as jar:
            e = pickle.load(jar)
            print(f"had old err {e=} {str(e)=} {repr(e)=}")
        if retry_err and not summarize_errors:
            (temps_dir / "error.pkl").unlink()
            print("retrying")
        else:
            return
    if summarize_errors:
        return

    temps_dir.mkdir(exist_ok=True)

    try:
        fs = 30000
        # load sorting results
        one = ONE()

        # one = ONE()
        ba = AllenAtlas()
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, _ = sl.load_spike_sorting()

        cleaned_spikes = {key.split(".")[0]: value for key, value in spikes.items()}
        cleaned_clusters = {key.split(".")[0]: value for key, value in clusters.items()}

        # only keep the good units
        good_units_mask = cleaned_clusters["metrics"]["label"] == 2 / 3

        uuids = cleaned_clusters["uuids"].values[good_units_mask]
        filtered_cluster_ids = np.where(good_units_mask)[0]

        # Initialize the dictionary to save the ACGs
        ACG_3D = {}

        # Loop through each unique cluster ID of interest
        for i in range(len(filtered_cluster_ids)):
            print(i)
            cid = filtered_cluster_ids[i]
            spike_train = cleaned_spikes["times"][cleaned_spikes["clusters"] == cid]
            if len(spike_train) == 0:
                acg_3d = None
            else:
                if log_acg:
                    win_size = 2000
                else:
                    win_size = 200
                acg_3d = make_3DACG(np.round(spike_train * fs), win_size, log_acg)
                ACG_3D[uuids[i]] = (acg_3d, len(spike_train))

        # Save the dictionary
        with open((temps_dir / "3d_acg_data_2_3.pkl"), "wb") as handle:
            pickle.dump(ACG_3D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"{e=} {str(e)=} {repr(e)=}")
        print(traceback.format_exc())
        with open(temps_dir / "error.pkl", "wb") as jar:
            pickle.dump(e, jar)
    else:
        if (temps_dir / "error.pkl").exists():
            print("previously had error but this time survived")
            (temps_dir / "error.pkl").unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--data_dir", default="~/ceph/bwm_700")
    ap.add_argument("--allsyms_dir", default="~/ceph/bwm_700_syms")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retry_err", action="store_true")
    ap.add_argument("--summarize_errors", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(exist_ok=True)

    allsyms_dir = Path(args.allsyms_dir).expanduser()
    allsyms_dir.mkdir(exist_ok=True)

    log_txt = data_dir / f"log{args.pid}.txt"
    with open(log_txt, "a") as logf:
        sys.stdout = logf
        sys.stderr = logf
        print("bwm_3DACG", time.strftime("%Y-%m-%d %H:%M"))
        print(f"{sys.executable=}")
        print(f"{args=}")

        compute_3DACG_IBL(
            args.pid,
            allsyms_dir,
            data_dir,
            overwrite=args.overwrite,
            retry_err=args.retry_err,
            summarize_errors=args.summarize_errors,
        )
