import numpy as np
from torchvision import transforms

# from npyx.c4.waveform_augmentations import GaussianNoise


class AmpJitter(object):
    """Rescales waveform amplitude by some small factor"""

    def __init__(self, lo=0.9, hi=1.1):
        """
        Args:
            lo: float
                the low end of the amplitude distortion.
            hi: float
                the high end of the amplitude distortion.
        """
        self.lo = lo
        self.hi = hi

    def __call__(self, sample):
        wf = sample
        # randomly select amp jitter scaling value and apply to waveform in each channel
        amp_jit_value = np.random.uniform(self.lo, self.hi)
        wf = wf * amp_jit_value

        return wf


class ElectrodeDropout(object):
    """Zero out a channel to mimic the electrode breaking."""

    def __init__(self, prob=0.1):
        """
        Args:
            prob: float
                each channel will be dropped with this probability.
        """
        self.p_drop_chan = prob

    def __call__(self, wf):
        if len(wf.shape) == 1:
            return wf
        n_chan, n_times = wf.shape
        chan_mask = -1 * np.random.binomial(1, self.p_drop_chan, n_chan) + 1

        wf[chan_mask == 0] = np.zeros(n_times)
        return wf


class GaussianNoise(object):
    """Adds Gaussian noise to the waveform"""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, sample):
        noise = np.random.normal(0, self.std * np.std(sample), size=sample.shape)
        return sample + noise


def get_wf_transform(
    # wvf_augmentation: list,
    aug_p_dict={"gaussian_noise": 0.3, "amp_jitter": 0.4, "electrode_dropout": 0.1},
    std: float = 0.1,
):
    transformation_dic = {
        "gaussian_noise": GaussianNoise(std=std),
        "amp_jitter": AmpJitter(),
        "electrode_dropout": ElectrodeDropout(prob=aug_p_dict["electrode_dropout"]),
    }
    wvf_augmentation = ["gaussian_noise"]

    trans = []
    for transformation in wvf_augmentation:
        if transformation in transformation_dic.keys():
            if transformation != "electrode_dropout":
                trans.append(
                    transforms.RandomApply(
                        [transformation_dic[transformation]],
                        p=aug_p_dict[transformation],
                    )
                )
            else:
                trans.append(transformation_dic[transformation])
    """Return a set of data augmentation transformations on waveforms."""
    wvf_transforms = transforms.Compose(trans)
    return wvf_transforms
