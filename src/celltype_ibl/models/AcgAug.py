import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from scipy.ndimage import zoom


class gaussian_smoothing(object):
    """
    Smooths an autocorrelogram (acg) using a Gaussian filter.
    """

    def __init__(self, sigma: int = 2):
        self.sigma = sigma

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        smoothed_acg = gaussian_filter(acg, sigma=self.sigma)
        return smoothed_acg


class temporal_gaussian_smoothing(object):
    """
    Smooths an autocorrelogram (acg) using a Gaussian filter along the temporal axis (axis=1).
    """

    def __init__(self, sigma: int = 2):
        self.sigma = sigma

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        smoothed_acg = gaussian_filter(acg, sigma=(0, self.sigma))
        return smoothed_acg


class temporal_jittering(object):
    """
    Jitters the temporal axis of an autocorrelogram (acg) by a random amount.
    """

    def __init__(self, max_jitter: int = 3):
        self.max_jitter = max_jitter

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        num_rows, num_cols = acg.shape
        # Generate a random jitter for each row
        jitters = np.random.randint(
            -self.max_jitter, self.max_jitter + 1, size=num_rows
        )
        # Create an array of column indices for each row
        col_indices = (np.arange(num_cols) - jitters[:, np.newaxis]) % num_cols
        # Apply the jitter using advanced indexing
        jittered_acg = acg[np.arange(num_rows)[:, np.newaxis], col_indices]
        return jittered_acg


class amplitude_scaling(object):
    """
    Scales the amplitude of an autocorrelogram (acg) by a given scale factor.

    Parameters:
        acg (array-like): The autocorrelogram to be scaled.
        scale_factor (float): The factor by which to scale the amplitude.

    Returns:
        array-like: The scaled autocorrelogram.
    """

    def __init__(self, lo: float = 0.9, hi: float = 1.1) -> None:
        self.lo = lo
        self.hi = hi

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        scale_factor = np.random.uniform(self.lo, self.hi)
        scaled_acg = acg * scale_factor
        return scaled_acg


class additive_gaussian_noise(object):
    """
    Adds Gaussian noise to an autocorrelogram (acg).
    Kind of don't want to use it since it may mess up with the refractory period. But let's see.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        noise = np.random.normal(self.mean, self.std * np.max(acg), size=acg.shape)
        noisy_acg = acg + noise
        noisy_acg = np.clip(noisy_acg, 0, np.inf)
        return noisy_acg


class additive_pepper_noise(object):
    """
    Adds pepper noise to an autocorrelogram (acg).
    """

    def __init__(self, pepper_prob=0.05) -> None:
        self.pepper_prob = pepper_prob

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        noisy_acg = np.copy(acg)
        pepper_mask = np.random.random(acg.shape) < self.pepper_prob
        noisy_acg[pepper_mask] = 0.0
        return noisy_acg


class decile_adjust_add(object):
    """
    adjust decile by adding the low firing rate row
    """

    def __init__(self, mode="constant") -> None:
        self.mode = mode

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        padded_acg = np.pad(acg, pad_width=((1, 0), (0, 0)), mode=self.mode)
        resampled_acg = zoom(padded_acg, (10 / 11, 1), order=3)
        return resampled_acg


class decile_adjust_delete(object):
    """
    adjust decile by deleting the low firing rate row
    """

    def __init__(self) -> None:
        pass

    def __call__(self, acg: np.ndarray) -> np.ndarray:
        resampled_acg = zoom(acg[1:10, :], (10 / 9, 1), order=3)
        return resampled_acg


def get_acg_transform(
    aug_p_dict={
        "temporal_gaussian_smoothing": 0.5,
        "temporal_jittering": 0.5,
        "amplitude_scaling": 0.5,
        "decile_adjust_add": 0.5,
        "decile_adjust_delete": 0.5,
        "additive_gaussian_noise": 0.5,
        "additive_pepper_noise": 0.5,
    }
):
    """Return a set of data augmentation transformations on acgs."""
    acg_transforms = transforms.Compose(
        [
            transforms.RandomApply(
                [temporal_gaussian_smoothing()],
                p=aug_p_dict["temporal_gaussian_smoothing"],
            ),
            transforms.RandomApply(
                [temporal_jittering()], p=aug_p_dict["temporal_jittering"]
            ),
            transforms.RandomApply(
                [amplitude_scaling()], p=aug_p_dict["amplitude_scaling"]
            ),
            # transforms.RandomApply(
            #     [decile_adjust_add()], p=aug_p_dict["decile_adjust_add"]
            # ),
            # transforms.RandomApply(
            #     [decile_adjust_delete()], p=aug_p_dict["decile_adjust_delete"]
            # ),
            transforms.RandomApply(
                [additive_gaussian_noise()], p=aug_p_dict["additive_gaussian_noise"]
            ),
            transforms.RandomApply(
                [additive_pepper_noise()], p=aug_p_dict["additive_pepper_noise"]
            ),
        ]
    )
    return acg_transforms
