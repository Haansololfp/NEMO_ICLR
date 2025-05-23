import os
from copy import copy

import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributions as dist
import torch.optim as optim
import torch.utils.data
from tqdm.auto import tqdm
from torchvision import transforms
from npyx.c4.dataset_init import (
    N_CHANNELS,
    WAVEFORM_SAMPLES,
    LABELLING,
    CORRESPONDENCE,
    calc_snr,
)
from npyx.c4.dl_utils import generate_kl_weight, define_forward_vae
from pathlib import Path
import matplotlib.pyplot as plt
from npyx.ml import set_seed
import npyx


class HorizontalCompression(object):
    """Compress or expand the signal horizontally by a given factor."""

    def __init__(self, p=0.3, max_compression_factor=0.6):
        self.max_compression_factor = max_compression_factor
        self.p = p

    def __call__(self, sample):
        if self.p <= np.random.rand():
            return sample
        wvf = np.squeeze(sample).reshape(1, -1)

        used_factor = np.random.choice(
            np.linspace(0.1, self.max_compression_factor, 5), size=1
        )

        factor = 1 + used_factor if np.random.choice([0, 1]) == 1 else 1 - used_factor

        new_wvf_shape = (1, int(np.ceil(wvf.shape[1] / factor)))
        new_wvf = np.zeros(new_wvf_shape)
        new_wvf[0] = np.interp(
            np.arange(0, wvf.shape[1], factor), np.arange(wvf.shape[1]), wvf[0]
        )
        if new_wvf.shape[1] != int(WAVEFORM_SAMPLES * 3 / 4):
            diff = new_wvf.shape[1] - int(WAVEFORM_SAMPLES * 3 / 4)
            if diff > 0:  # Crop
                crop_left = diff // 2
                crop_right = diff - crop_left
                new_wvf = new_wvf[:, crop_left:-crop_right]
            else:  # Pad
                pad_left = -diff // 2
                pad_right = -diff - pad_left
                new_wvf = np.pad(
                    new_wvf, ((0, 0), (pad_left, pad_right)), mode="reflect"
                )
        new_wvf = new_wvf.ravel().reshape(1, -1).copy().astype(np.float32)

        return new_wvf


class CerebellumWFDataset(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, WAVEFORM_SAMPLES)"""

    def __init__(self, data, labels, transforms=None):
        """
        Args:
            data (ndarray): Array of data points
            labels (string): Array of labels for the provided data
        """
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = (
            self.data[idx, :]
            .astype("float32")
            .reshape(1, int(WAVEFORM_SAMPLES * 3 / 4))
        )
        label = self.labels[idx].astype("int")

        if self.transforms:
            data_point = self.transforms(data_point)

        return data_point, label


class CustomMSELoss(nn.Module):
    def __init__(self, weight_factor=2.0):
        super(CustomMSELoss, self).__init__()
        self.weight_factor = weight_factor
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, input, target):
        # Compute the MSE loss for the entire image without reduction
        mse_loss = self.mse_loss(input, target)

        # Define the weight mask to assign higher weight to specific region
        weight_mask = torch.ones_like(target)
        weight_mask[:, :, 20:40] *= self.weight_factor

        return mse_loss * weight_mask


def ELBO_VAE(enc, dec, X, beta=1, n_samples=10):
    """

    INPUT:
    enc : Instance of `Encoder` class, which returns a distribution
          over Z when called on a batch of inputs X
    dec : Instance of `Decoder` class, which returns a distribution
          over X when called on a batch of inputs Z
    X   : A batch of datapoints, torch.FloatTensor of shape = (batch_size, 1, 10, 60).

    """

    batch_size = X.shape[0]
    ELBO = torch.zeros(batch_size).to(DEVICE)
    for _ in range(n_samples):
        q_z = enc.forward(X)  # q(Z | X)
        z = (
            q_z.rsample()
        )  # Samples from the encoder posterior q(Z | X) using the reparameterization trick

        reconstruction = dec.forward(z)  # distribution p(x | z)

        prior = dist.Normal(
            torch.zeros_like(q_z.loc).to(DEVICE), torch.ones_like(q_z.scale).to(DEVICE)
        )

        # compute MSE loss between input and reconstruction
        custom_mse = CustomMSELoss(weight_factor=1.0)
        MSE = custom_mse(reconstruction, X).sum(dim=(1, 2))

        KLD = dist.kl_divergence(q_z, prior).sum(dim=1)

        ELBO += MSE + beta * (batch_size / DATASET_LEN) * KLD

    return (ELBO / n_samples).mean()


DATASETS_DIRECTORY = "/mnt/sdceph/users/hyu10/cell-type_representation/"
FLIP = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 1234
USE_CUDA = torch.cuda.is_available()
BETA = 5
SAVE_DIRECTORY = "/mnt/sdceph/users/hyu10/c4_results_new/VAE_checkpoints"

dataset_paths = npyx.c4.get_paths_from_dir(DATASETS_DIRECTORY, include_hull_unlab=True)

# Normalise waveforms so that the max in the dataset is 1 and the minimum is -1. Only care about shape.
BASE_DATASET = npyx.c4.extract_and_merge_datasets(
    *dataset_paths,
    quality_check=False,
    normalise_wvf=False,  # only applies to the multichannel waveforms
    _use_amplitudes=False,
    n_channels=N_CHANNELS,
    central_range=WAVEFORM_SAMPLES,
    labelled=False,
    flip_waveforms=FLIP,
)

BASE_DATASET.make_unlabelled_only()
BASE_DATASET.make_full_dataset(wf_only=True)

relevant_list = []

for wave in tqdm(BASE_DATASET.wf):
    waveform_2d = wave.reshape(N_CHANNELS, -1)
    for wf in waveform_2d:
        if calc_snr(wf, return_db=True) <= 25:
            continue

        scaled_wvf = npyx.datasets.preprocess_template(
            wf, clip_size=[1e-3, 2e-3], peak_sign={True: "Negative", False: None}[FLIP]
        )
        if np.max(np.abs(scaled_wvf[40:])) > 1:
            continue
        relevant_list.append(scaled_wvf)
print(
    f"Extracted a total of {len(relevant_list)} waveforms out of {len(BASE_DATASET)*N_CHANNELS} ({len(relevant_list)/(len(BASE_DATASET)*N_CHANNELS)*100:.2f}%)"
)

DATASET_LEN = len(relevant_list)

train_params = {
    "beta": BETA,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 1e-4,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}

for SEED in range(1234, 1234 + 5):
    set_seed(SEED)
    torch.cuda.empty_cache()

    enc, dec = define_forward_vae(
        len(relevant_list[0]), params=train_params, device=DEVICE
    )

    BATCH_SIZE = train_params["batch_size"]
    # Add augmentation transforms if wanted
    labels = np.array([0] * DATASET_LEN)
    wf_dataset = np.stack(relevant_list)

    # Add augmentation transforms if wanted
    cerebellum_dataset = CerebellumWFDataset(
        wf_dataset,
        labels,
        transforms=transforms.Compose([HorizontalCompression()]),
    )
    train_loader = data.DataLoader(
        cerebellum_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer_name = train_params["optimizer"]
    lr = train_params["lr"]
    optim_args = {
        "params": itertools.chain(enc.parameters(), dec.parameters()),
        "lr": lr,
    }
    opt_vae = getattr(optim, optimizer_name)(**optim_args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_vae, 20, 1, last_epoch=-1
    )

    N_epochs = 60
    losses = []

    # these are used to anneal the KL divergence loss over training that helps with convergence of VAEs
    # (multiplies the KL half of the loss by kl_weights[epoch] which goes from 0 to BETA (5))
    kl_weights = generate_kl_weight(N_epochs, beta=train_params["beta"])

    for epoch in tqdm(range(N_epochs), desc="Epochs"):
        train_loss = 0.0
        for X, _ in train_loader:
            X = X.to(DEVICE)
            opt_vae.zero_grad()
            loss = ELBO_VAE(enc, dec, X, beta=kl_weights[epoch])
            loss.backward()
            opt_vae.step()
            train_loss += loss.item() * X.shape[0] / len(BASE_DATASET)
        scheduler.step()
        losses.append(train_loss)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"losses_seed_{SEED}.png")

    enc = enc.eval().cpu()
    dec = dec.eval().cpu()
    # Save the model
    f_s = {False: "_noflip", True: ""}[FLIP]

    torch.save(
        enc.state_dict(),
        SAVE_DIRECTORY + f"/wvf_singlechannel_encoder{f_s}_seed_{SEED}.pt",
    )
    torch.save(
        dec.state_dict(),
        SAVE_DIRECTORY + f"wvf_singlechannel_decoder{f_s}_seed_{SEED}.pt",
    )
