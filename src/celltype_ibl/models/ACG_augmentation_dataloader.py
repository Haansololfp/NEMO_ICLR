from celltype_ibl.models.WvfAug import get_wf_transform
from celltype_ibl.models.AcgAug import get_acg_transform
from celltype_ibl.params.config import DATASETS_DIRECTORY
from torch.utils.data import Dataset
import npyx
import torch
import numpy as np
import os
import pandas as pd



class EmbeddingDataset(Dataset):
    def __init__(
        self,
        wvf: np.ndarray,
        acg: np.ndarray,
        augmentation: bool = True,
        acg_multi_factor: float = 10,
        modality: str = "bimodal",
        acg_normalize: bool = False,
        std: float = 1.0,
    ) -> None:
        """
        wvf: np.array of shape (N, 90)
        acg: np.array of shape (N, )
        """
        self.wvf = wvf
        self.acg = acg
        assert wvf.shape[0] == acg.shape[0]
        self.N = wvf.shape[0]

        self.augmentation = augmentation
        if self.augmentation:
            self.wvf_transform = get_wf_transform(std = std)
            self.acg_transform = get_acg_transform()
        self.acg_multi_factor = acg_multi_factor

        assert modality in ["bimodal", "wvf", "acg"]
        self.modality = modality
        self.acg_normalize = acg_normalize

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idwvf: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modality == "bimodal":
            input1 = self.wvf
            input2 = self.acg
            if self.augmentation:
                transform1 = self.wvf_transform
                transform2 = self.acg_transform
        elif self.modality == "wvf":
            input1 = self.wvf
            input2 = self.wvf
            if self.augmentation:
                transform1 = self.wvf_transform
                transform2 = self.wvf_transform
        elif self.modality == "acg":
            input1 = self.acg
            input2 = self.acg
            if self.augmentation:
                transform1 = self.acg_transform
                transform2 = self.acg_transform

        if self.augmentation:
            output1_i = transform1(input1[idwvf]).copy()
            output2_i = transform2(input2[idwvf]).copy()
        else:
            output1_i = input1[idwvf].copy()
            output2_i = input2[idwvf].copy()

        if self.modality == "bimodal":
            output1_i = output1_i / np.max(np.abs(output1_i))
            if self.acg_normalize:
                output2_i = output2_i / np.max(np.abs(output2_i) + 1e-8)
            output2_i = output2_i * self.acg_multi_factor
        elif self.modality == "wvf":
            output1_i = output1_i / np.max(np.abs(output1_i))
            output2_i = output2_i / np.max(np.abs(output2_i))
        elif self.modality == "acg":
            if self.acg_normalize:
                output1_i = output1_i / np.max(np.abs(output1_i) + 1e-8)
                output2_i = output2_i / np.max(np.abs(output2_i) + 1e-8)
            output1_i = output1_i * self.acg_multi_factor
            output2_i = output2_i * self.acg_multi_factor

        return torch.from_numpy(output1_i.astype("float32")), torch.from_numpy(
            output2_i.astype("float32")
        )


class FOCALDataset(Dataset):
    def __init__(
        self,
        wvf: np.ndarray,
        acg: np.ndarray,
        augmentation: bool = True,
        acg_multi_factor: float = 10,
        modality: str = "bimodal",
        acg_normalize: bool = False,
    ) -> None:
        """
        wvf: np.array of shape (N, 90)
        acg: np.array of shape (N, )
        """
        self.wvf = wvf
        self.acg = acg
        assert wvf.shape[0] == acg.shape[0]
        self.N = wvf.shape[0]

        assert augmentation
        self.wvf_transform = get_wf_transform()
        self.acg_transform = get_acg_transform()
        self.acg_multi_factor = acg_multi_factor

        assert modality == "bimodal"
        assert not acg_normalize

    def __len__(self) -> int:
        return self.N

    def __getitem__(
        self, idwvf: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wvf_i = self.wvf
        acg_i = self.acg
        wvf_transform = self.wvf_transform
        acg_transform = self.acg_transform

        aug_wvf_i_1 = wvf_transform(wvf_i[idwvf]).copy()
        aug_acg_i_1 = acg_transform(acg_i[idwvf]).copy()
        aug_acg_i_1 = aug_acg_i_1 * self.acg_multi_factor

        aug_wvf_i_2 = wvf_transform(wvf_i[idwvf]).copy()
        aug_acg_i_2 = acg_transform(acg_i[idwvf]).copy()
        aug_acg_i_2 = aug_acg_i_2 * self.acg_multi_factor

        return (
            torch.from_numpy(aug_wvf_i_1.astype("float32")),
            torch.from_numpy(aug_acg_i_1.astype("float32")),
            torch.from_numpy(aug_wvf_i_2.astype("float32")),
            torch.from_numpy(aug_acg_i_2.astype("float32")),
        )
