from scipy import signal
from scipy.signal import find_peaks
import numpy as np


def align_multichan_spikes(wfs, peak_T=42):
    """
    input: waveforms of shape NxT
    upsample the waveform in time first for finer scale alignment
    then downsample the waveform to its original shape
    """
    N, T, CH_N = np.shape(wfs)
    upsampled_wfs = signal.resample(wfs, 10 * T, axis=1)
    aligned_wfs = np.zeros([N, 2500, CH_N])

    peak_idx = 1050  # a value that is larger than all peak index values

    ptps = np.ptp(wfs, axis=1)
    max_CH = np.argmax(ptps, axis=1)
    maxCH_wfs = upsampled_wfs[np.arange(N), max_CH, :]
    peak_point = np.argmax(np.abs(maxCH_wfs), axis=1)

    shift = peak_idx - peak_point
    for i in range(len(shift)):
        aligned_wfs[i, shift[i] : (shift[i] + 10 * T), :] = upsampled_wfs[i, :, :]

    return signal.resample(
        aligned_wfs[
            :,
            round(peak_idx - peak_T * 10) : (
                round(peak_idx - peak_T * 10 + 10 * T) + 1
            ),
            :,
        ],
        T,
        axis=1,
    )


def align_singleCH_spikes(wfs, peak_T=42, peak_point="max_abs"):
    # input: waveforms of shape NxT
    # upsample the waveform in time first for finer scale alignment
    # then downsample the waveform to its original shape
    N, T = np.shape(wfs)
    upsampled_wfs = signal.resample(wfs, 10 * T, axis=1)
    aligned_wfs = np.zeros([N, 2500])

    peak_idx = 1050  # a value that is larger than all peak index values

    if peak_point == "max_abs":
        peak_point = np.argmax(np.absolute(upsampled_wfs), axis=1)
    elif peak_point == "min":
        peak_point = np.argmin(upsampled_wfs, axis=1)
    else:
        raise ValueError("peak_point must be 'max_abs' or 'min'")

    shift = peak_idx - peak_point

    for i in range(len(shift)):
        try:
            aligned_wfs[i, shift[i] : (shift[i] + 10 * T)] = upsampled_wfs[i, :]
        except:
            print(i)

    return signal.resample(
        aligned_wfs[
            :,
            round(peak_idx - peak_T * 10) : (
                round(peak_idx - peak_T * 10 + 10 * T) + 1
            ),
        ],
        T,
        axis=1,
    )


def align_singleCH_spikes_no_upsample(wfs, peak_T=42):
    # input: waveforms of shape NxT
    # upsample the waveform in time first for finer scale alignment
    # then downsample the waveform to its original shape
    N, T = np.shape(wfs)
    aligned_wfs = np.zeros([N, 250])

    peak_idx = 105  # a value that is larger than all peak index values

    peak_point = np.argmax(np.absolute(wfs), axis=1)
    # peak_point = np.argmin(upsampled_wfs, axis =1)

    shift = peak_idx - peak_point

    for i in range(len(shift)):
        try:
            aligned_wfs[i, shift[i] : (shift[i] + T)] = wfs[i, :]
        except:
            print(i)

    return (
        aligned_wfs[
            :,
            round(peak_idx - peak_T) : (round(peak_idx - peak_T + T) + 1),
        ],
    )
