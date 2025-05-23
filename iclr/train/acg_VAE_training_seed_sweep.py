import os
import logging
import sys
import inspect
from copy import copy

import itertools
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributions as dist
import torch.optim as optim
from torch.optim import Adam, RMSprop
import torch.utils.data
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import npyx.c4.dl_utils as m
import npyx.c4.dl_transforms as custom_transforms
import torchvision.transforms as transforms
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import npyx


acg_3d = np.load(
    "/mnt/sdceph/users/hyu10/cell-type_representation/acgs_vs_firing_rate/unlabelled_acgs_3d_augmented_logscale.npy"
)
acg_3d = acg_3d.reshape(-1, 10, 201)
acg_3d = acg_3d * 10
DATASET_LEN = len(acg_3d)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
INIT_WEIGHTS = False
SAVE_DIRECTORY = "/mnt/sdceph/users/hyu10/c4_results_new/VAE_checkpoints"


class ACG3Dataset(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, CENTRAL_RANGE)"""

    def __init__(
        self,
        data: np.ndarray,
    ):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            labels (string): Array of labels for the provided data
            raw_spikes (ndarray): Array of raw spikes for the provided data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, transform=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx, :].astype("float32").reshape(1, -1)


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
        weight_mask[:, :, :, :20] *= self.weight_factor

        return mse_loss * weight_mask


class Encoder(nn.Module):
    def __init__(self, d_latent, initialise=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 10))
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.batchnorm1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, (5, 1))
        self.maxpool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 1 * 23, 200)
        self.fc2 = nn.Linear(200, d_latent * 2)
        self.d_latent = d_latent
        self.dropout = nn.Dropout(0.2)

        if initialise:
            self.conv1.weight.data.normal_(0, 0.001)
            self.conv1.bias.data.normal_(0, 0.001)

            self.conv2.weight.data.normal_(0, 0.001)
            self.conv2.bias.data.normal_(0, 0.001)

            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

            self.fc2.weight.data.normal_(0, 0.001)
            self.fc2.bias.data.normal_(0, 0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.gelu(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(F.gelu(self.fc1(x)))
        h = self.fc2(h)
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return dist.Normal(mu, torch.exp(log_var))


class Decoder(nn.Module):
    def __init__(self, d_latent, initialise=False):
        super().__init__()
        self.d_latent = d_latent
        self.fc1 = nn.Linear(d_latent, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, (10 * 101))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        if initialise:
            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

            self.fc2.weight.data.normal_(0, 0.001)
            self.fc2.bias.data.normal_(0, 0.001)

            self.fc3.weight.data.normal_(0, 0.001)
            self.fc3.bias.data.normal_(0, 0.001)

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(z)))
        h = self.dropout2(F.relu(self.fc2(h)))
        X_reconstructed = self.fc3(h)

        # X_reconstructed = torch.sigmoid(X_reconstructed)

        return X_reconstructed.reshape(-1, 1, 10, 101)


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

        custom_mse = CustomMSELoss(weight_factor=1.5)

        MSE = custom_mse(reconstruction, X).sum(dim=(1, 2, 3))

        KLD = dist.kl_divergence(q_z, prior).sum(dim=1)

        ELBO += MSE + beta * (batch_size / DATASET_LEN) * KLD

    return (ELBO / n_samples).mean()


def generate_kl_weight(epochs, beta=1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    weight = np.logspace(5, -20, epochs)
    weight = sigmoid(-np.log10(weight)) * beta

    return weight


D_LATENT = 10
BETA = 5
BATCH_SIZE = 32
lr = 5e-4

input_data = acg_3d[:, :, 100:]
dataset = ACG3Dataset(input_data)
train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
for SEED in range(1234, 1234 + 5):
    npyx.ml.set_seed(SEED)
    torch.cuda.empty_cache()
    initialise = False

    enc, dec = Encoder(D_LATENT, initialise), Decoder(D_LATENT, initialise)
    enc.to(DEVICE)
    dec.to(DEVICE)
    enc.train()
    dec.train()

    optim_args = {
        "params": itertools.chain(enc.parameters(), dec.parameters()),
        "lr": lr,
    }
    opt_vae = optim.Adam(**optim_args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_vae, 20, 1, last_epoch=-1
    )

    N_epochs = 60
    losses = []
    kl_weights = generate_kl_weight(N_epochs, beta=BETA)

    for epoch in tqdm(range(N_epochs), desc="Epochs"):
        train_loss = 0.0
        for X in train_loader:
            X = X.to(DEVICE)
            X = X.reshape(-1, 1, 10, 101)
            opt_vae.zero_grad()
            loss = ELBO_VAE(enc, dec, X, beta=kl_weights[epoch])
            loss.backward()
            opt_vae.step()
            train_loss += loss.item() * X.shape[0] / len(dataset)
        scheduler.step()
        losses.append(train_loss)

    enc.cpu().eval()
    dec.cpu().eval()

    torch.save(
        enc.state_dict(), SAVE_DIRECTORY + f"3DACG_logscale_encoder_gelu_seed_{SEED}.pt"
    )

    torch.save(
        dec.state_dict(), SAVE_DIRECTORY + f"3DACG_logscale_decoder_gelu_seed_{SEED}.pt"
    )
