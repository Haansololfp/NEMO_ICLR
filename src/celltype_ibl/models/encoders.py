import torch
import torch.nn as nn
import torch.nn.functional as F

WVF_ENCODER_ARGS_SINGLE = {
    "beta": 5,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 5e-5,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}


class SimpleWaveformEncoder(nn.Module):
    def __init__(self, latent_size: int = 16):
        super().__init__()
        self.linear1 = nn.LazyLinear(64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, latent_size)

    def forward(self, x: torch.FloatTensor | torch.cuda.FloatTensor):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return self.linear3(x)


class SimpleAcgEncoder(nn.Module):
    def __init__(self, latent_size: int = 16):
        super().__init__()
        self.linear1 = nn.LazyLinear(128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, latent_size)

    def forward(self, x: torch.FloatTensor | torch.cuda.FloatTensor):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return self.linear4(x)


############################
# adapted from https://github.com/m-beau/NeuroPyxels/blob/master/npyx/c4/dl_utils.py
# replaced the final layer with a linear projector


class LinearProjector(nn.Module):
    def __init__(
        self,
        dim_rep: int,
        dim_embed: int,
        initialise: bool = False,
        apply_layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_rep, dim_embed)
        self.dim_rep = dim_rep
        self.dim_embed = dim_embed
        self.apply_layer_norm = apply_layer_norm
        self.apply_batch_norm = batch_norm
        if initialise:
            self.fc1.weight.data.normal_(0, 0.001)
        if apply_layer_norm:
            self.layer_norm = nn.LayerNorm([dim_embed])
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim_embed)

    def forward(self, x: torch.FloatTensor | torch.cuda.FloatTensor):
        h = self.fc1(x)
        if self.apply_layer_norm:
            h = self.layer_norm(h)
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        return h


class NonLinearProjector(nn.Module):
    def __init__(
        self,
        dim_rep: int,
        dim_embed: int,
        initialise: bool = False,
        apply_layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_rep, dim_embed)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dim_embed, dim_embed)
        self.dim_rep = dim_rep
        self.dim_embed = dim_embed
        self.apply_layer_norm = apply_layer_norm
        self.apply_batch_norm = batch_norm
        if initialise:
            self.fc1.weight.data.normal_(0, 0.001)
        if apply_layer_norm:
            self.layer_norm = nn.LayerNorm([dim_embed])
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim_embed)

    def forward(self, x: torch.FloatTensor | torch.cuda.FloatTensor):
        h = self.fc1(x)
        if self.apply_layer_norm:
            h = self.layer_norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        return h


class ConvolutionalEncoder(nn.Module):
    def __init__(
        self,
        dim_rep: int,
        initialise: bool = False,
        pool: str = "avg",
        activation: str = "gelu",
        acg_dropout: float | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 10))
        self.maxpool1 = (
            nn.MaxPool2d(kernel_size=(2, 2))
            if pool == "max"
            else nn.AvgPool2d(kernel_size=(2, 2))
        )
        self.batchnorm1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, (5, 1))
        self.maxpool2 = (
            nn.MaxPool2d(kernel_size=(1, 2))
            if pool == "max"
            else nn.AvgPool2d(kernel_size=(1, 2))
        )
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 1 * 23, dim_rep)
        if acg_dropout is None:
            acg_dropout = 0.2
        self.dropout = nn.Dropout(acg_dropout)
        self.dim_rep = dim_rep

        if initialise:
            self.conv1.weight.data.normal_(0, 0.001)
            self.conv1.bias.data.normal_(0, 0.001)

            self.conv2.weight.data.normal_(0, 0.001)
            self.conv2.bias.data.normal_(0, 0.001)

            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

        if activation == "relu":
            self.activation_F = F.relu
        elif activation == "gelu":
            self.activation_F = F.gelu
        elif activation == "tanh":
            self.activation_F = F.tanh
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x: torch.FloatTensor | torch.cuda.FloatTensor):
        x = self.conv1(x)
        x = self.activation_F(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.activation_F(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(self.activation_F(self.fc1(x)))
        return h


def define_forward_encoder(
    in_features: int = 90,
    init_weights: bool = True,
    params: dict | None = None,
    wvf_dropout: None | float = None,
):
    if params is None:
        best_params = WVF_ENCODER_ARGS_SINGLE
    else:
        best_params = params

    if wvf_dropout is not None:
        best_params["dropout_l0"] = wvf_dropout
        best_params["dropout_l1"] = wvf_dropout

    n_layers = best_params["n_layers"]
    encoder_layers = []

    for i in range(n_layers):
        out_features = best_params[f"n_units_l{i}"]
        p = best_params[f"dropout_l{i}"]

        # Create and properly init encoder layer
        cur_enc_layer = nn.Linear(in_features, out_features)
        if init_weights:
            cur_enc_layer.weight.data.normal_(0, 0.001)
            cur_enc_layer.bias.data.normal_(0, 0.001)

        encoder_layers.append(cur_enc_layer)
        encoder_layers.append(nn.GELU())
        encoder_layers.append(nn.Dropout(p))
        in_features = out_features
    encoder = nn.Sequential(*encoder_layers)
    return encoder


class SingleWVFEncoder(nn.Module):
    def __init__(
        self,
        params=None,
        initialise=False,
        adjust_to_ce=False,
        adjust_to_ultra=False,
        wvf_dropout=None,
    ):
        super().__init__()
        if params is None:
            params = WVF_ENCODER_ARGS_SINGLE
        assert adjust_to_ce & adjust_to_ultra == False

        if not (adjust_to_ce | adjust_to_ultra):
            in_features = 90
        elif adjust_to_ce:
            in_features = 41
        else:
            in_features = 82

        self.dim_rep = params[f"n_units_l{params['n_layers']-1}"]
        self.encoder = define_forward_encoder(
            in_features=in_features,
            init_weights=initialise,
            params=params,
            wvf_dropout=wvf_dropout,
        )

    def forward(self, x):
        h = self.encoder(x)
        return h
