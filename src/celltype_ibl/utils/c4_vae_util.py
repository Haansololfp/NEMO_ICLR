import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from npyx.c4.dl_utils import BaseVAE


class VAEEncoder(nn.Module):
    # Encoder network for a VAE, support outputting the pre-projection output
    def __init__(self, encoder, d_latent):
        super().__init__()
        self.encoder = encoder.float()
        self.d_latent = d_latent

    def forward(self, x, return_pre_projection=False, isacg=False):
        # Flatten the image
        if not isacg:
            for i, layer in enumerate(list(self.encoder.children())[:-1]):
                x = layer(x)
                if (
                    i == len(list(self.encoder.children())[:-2])
                    and return_pre_projection
                ):
                    # Return the output of the layer before the last linear projection
                    pre_projection_output = x
            h = self.encoder[-1](x)
        else:
            pre_projection_output = self.encoder(x, return_pre_projection=True)
            h = self.encoder(x, return_mu=True)

        # Split the output into mu and log_var
        mu = h[:, : self.d_latent]
        # log_var = h[:, self.d_latent :]

        if return_pre_projection:
            return pre_projection_output
        else:
            return mu

    def embed(self, x, return_pre_projection=False, isacg=False):
        return self.forward(x, return_pre_projection, isacg)


class vae_encode_model(nn.Module):
    def __init__(self, wvf_encoder, acg_encoder):
        super().__init__()
        self.wvf_encoder = wvf_encoder
        self.acg_encoder = acg_encoder

    def embed(self, wvf, acg, return_pre_projection=False):
        wvf_rep = self.wvf_encoder(wvf, return_pre_projection=return_pre_projection)
        acg_rep = self.acg_encoder(
            acg, return_pre_projection=return_pre_projection, isacg=True
        )
        return wvf_rep, acg_rep


class ConvolutionalEncoder(nn.Module):
    def __init__(self, d_latent, initialise=False, pool="max", activation="relu"):
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

        if activation == "relu":
            self.activation_F = F.relu
        elif activation == "gelu":
            self.activation_F = F.gelu
        elif activation == "tanh":
            self.activation_F = F.tanh
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x, return_mu=False, return_pre_projection=False) -> dist.Normal:
        x = self.conv1(x)
        x = self.activation_F(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.activation_F(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(self.activation_F(self.fc1(x)))
        if return_pre_projection:
            return h
        h = self.fc2(h)
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return (
            mu
            if return_mu
            else dist.Normal(mu, torch.exp(log_var), validate_args=False)
        )


class ForwardDecoder(nn.Module):
    def __init__(
        self,
        d_latent,
        central_range,
        n_channels,
        hidden_units=None,
        initialise=True,
        activation="relu",
    ):
        super().__init__()
        self.central_range = central_range
        self.n_channels = n_channels
        self.d_latent = d_latent
        if hidden_units is None:
            hidden_units = [100, 200]
        self.fc1 = nn.Linear(d_latent, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], int(n_channels * central_range))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        if initialise:
            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

            self.fc2.weight.data.normal_(0, 0.001)
            self.fc2.bias.data.normal_(0, 0.001)

            self.fc3.weight.data.normal_(0, 0.001)
            self.fc3.bias.data.normal_(0, 0.001)

        if activation == "relu":
            self.activation_F = F.relu
        elif activation == "gelu":
            self.activation_F = F.gelu
        elif activation == "tanh":
            self.activation_F = F.tanh
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        h = self.dropout1(self.activation_F(self.fc1(z)))
        h = self.dropout2(self.activation_F(self.fc2(h)))
        X_reconstructed = self.fc3(h)

        return X_reconstructed.reshape(-1, 1, self.n_channels, self.central_range)


class ACG3DVAE(BaseVAE):
    def __init__(
        self, d_latent, acg_bins, acg_width, device=None, pool="max", activation="relu"
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = ConvolutionalEncoder(
            d_latent, pool=pool, activation=activation
        ).to(self.device)
        self.decoder = ForwardDecoder(
            d_latent,
            acg_width,
            acg_bins,
            hidden_units=[250, 500],
            activation=activation,
        ).to(self.device)
        self.acg_bins = acg_bins
        self.acg_width = acg_width

    def encode_numpy(self, x):
        with torch.no_grad():
            self.encoder.eval()
            x_tensor = (
                torch.tensor(x)
                .to(self.device)
                .float()
                .reshape(-1, 1, self.acg_bins, self.acg_width)
            )
            return self.encode(x_tensor).detach().cpu().numpy()


class ACGForwardVAE(nn.Module):
    def __init__(self, encoder, decoder, in_features, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features

    def encode_numpy(self, x):
        with torch.no_grad():
            self.encoder.eval()
            x_tensor = (
                torch.tensor(x).to(self.device).float().reshape(-1, 1, self.in_features)
            )
            return self.encode(x_tensor).detach().cpu().numpy()


def load_acg_vae(
    encoder_path,
    win_size,
    bin_size,
    d_latent=10,
    initialise=True,
    device=None,
    pool="max",
    activation="relu",
):
    # Initialise and apply the encoder to the waveforms

    vae = ACG3DVAE(
        d_latent,
        acg_bins=10,
        acg_width=int(win_size // bin_size + 1),
        device=device,
        pool=pool,
        activation=activation,
    )

    if initialise:
        decoder_path = encoder_path.replace("encoder", "decoder")

        vae.load_weights(encoder_path, decoder_path)

    return vae
