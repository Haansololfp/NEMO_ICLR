import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from celltype_ibl.models.encoders import (
    SingleWVFEncoder,
    ConvolutionalEncoder,
    LinearProjector,
)


class SimclrContractiveLoss(torch.nn.Module):
    def __init__(
        self, temperature: torch.float32, similarity_adjust: bool = True
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.similarity_adjust = similarity_adjust

    def EmbeddingSimilarity(self, X, Y):
        """
        X, Y are two embedding matrices with size B(batch size)xD(embedding dimension)
        """
        return torch.matmul(X, Y.transpose(0, 1))

    def target(self, X: torch.tensor, Y: torch.tensor):
        """
        similarity-adjusted target
        t: temperature hyperparameter
        """
        sim1 = self.EmbeddingSimilarity(X, X)
        sim2 = self.EmbeddingSimilarity(Y, Y)
        return F.softmax((sim1 + sim2) / (2 * self.temperature), dim=-1)

    def forward(self, X: torch.tensor, Y: torch.tensor):
        if self.similarity_adjust:
            xy_target = self.target(X, Y)
        else:
            xy_target = torch.arange(len(X)).to(X.device)
        xysim = self.EmbeddingSimilarity(X, Y) / self.temperature
        yxsim = self.EmbeddingSimilarity(Y, X) / self.temperature
        xxsim = self.EmbeddingSimilarity(X, X) / self.temperature
        yysim = self.EmbeddingSimilarity(Y, Y) / self.temperature
        xx_mask = torch.eye(xxsim.shape[0], device=xxsim.device) * 1e6
        xxsim = xxsim - xx_mask
        yy_mask = torch.eye(yysim.shape[0], device=xxsim.device) * 1e6
        yysim = yysim - yy_mask

        x_loss = F.cross_entropy(
            torch.cat((xysim, xxsim), dim=1),
            xy_target,
            reduction="mean",
        )
        y_loss = F.cross_entropy(
            torch.cat((yxsim, yysim), dim=1),
            xy_target,
            reduction="mean",
        )

        return (x_loss + y_loss) / 2


class BiModalContractiveLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: torch.float32,
        similarity_adjust: bool = True,
        loss_flood: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.similarity_adjust = similarity_adjust
        self.loss_flood = loss_flood

    def EmbeddingSimilarity(self, X, Y):
        """
        X, Y are two embedding matrices with size B(batch size)xD(embedding dimension)
        """
        return torch.matmul(X, Y.transpose(0, 1))

    def target(self, X: torch.tensor, Y: torch.tensor):
        """
        similarity-adjusted target
        t: temperature hyperparameter
        """
        sim1 = self.EmbeddingSimilarity(X, X)
        sim2 = self.EmbeddingSimilarity(Y, Y)
        return F.softmax((sim1 + sim2) / (2 * self.temperature), dim=-1)

    def forward(self, X: torch.tensor, Y: torch.tensor):
        xsim = self.EmbeddingSimilarity(X, Y) / self.temperature
        if self.similarity_adjust:
            xy_target = self.target(X, Y)
        else:
            xy_target = torch.arange(len(X)).to(xsim.device)
        x_loss = F.cross_entropy(xsim, xy_target, reduction="mean")
        y_loss = F.cross_entropy(xsim.transpose(0, 1), xy_target, reduction="mean")

        if not self.loss_flood:
            return (x_loss + y_loss) / 2
        else:
            return torch.abs((x_loss + y_loss) / 2 - self.loss_flood) + self.loss_flood


class BiModalContractiveLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: torch.float32,
        similarity_adjust: bool = True,
        loss_flood: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.similarity_adjust = similarity_adjust
        self.loss_flood = loss_flood

    def EmbeddingSimilarity(self, X, Y):
        """
        X, Y are two embedding matrices with size B(batch size)xD(embedding dimension)
        """
        return torch.matmul(X, Y.transpose(0, 1))

    def target(self, X: torch.tensor, Y: torch.tensor):
        """
        similarity-adjusted target
        t: temperature hyperparameter
        """
        sim1 = self.EmbeddingSimilarity(X, X)
        sim2 = self.EmbeddingSimilarity(Y, Y)
        return F.softmax((sim1 + sim2) / (2 * self.temperature), dim=-1)

    def forward(self, X: torch.tensor, Y: torch.tensor):
        xsim = self.EmbeddingSimilarity(X, Y) / self.temperature
        if self.similarity_adjust:
            xy_target = self.target(X, Y)
        else:
            xy_target = torch.arange(len(X)).to(xsim.device)
        x_loss = F.cross_entropy(xsim, xy_target, reduction="mean")
        y_loss = F.cross_entropy(xsim.transpose(0, 1), xy_target, reduction="mean")

        if not self.loss_flood:
            return (x_loss + y_loss) / 2
        else:
            return torch.abs((x_loss + y_loss) / 2 - self.loss_flood) + self.loss_flood


class RINCE_loss(torch.nn.Module):
    def __init__(
        self, temperature: torch.float32, lam: float = 0.01, q: float = 0.1
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.lam = lam
        self.q = q

    def EmbeddingSimilarity(self, X, Y):
        """
        X, Y are two embedding matrices with size B(batch size)xD(embedding dimension)
        """
        return torch.matmul(X, Y.transpose(0, 1))

    def forward(self, X: torch.tensor, Y: torch.tensor):
        xsim = torch.exp(self.EmbeddingSimilarity(X, Y) / self.temperature)
        neg_mask = torch.ones(xsim.shape, device=xsim.device)
        neg_mask.fill_diagonal_(0)
        neg = (
            torch.sum(xsim * neg_mask, dim=1) + torch.sum(xsim * neg_mask, dim=0)
        ) / 2
        pos = torch.diag(xsim)

        neg = ((self.lam * (pos + neg)) ** self.q) / self.q
        pos = -(pos**self.q) / self.q

        return pos.mean() + neg.mean()


class BimodalEmbeddingModel(nn.Module):
    """
    joint representation learning with wfs and acg
    """

    def __init__(
        self,
        wvf_encoder: nn.Module | None = None,
        acg_encoder: nn.Module | None = None,
        temperature=0.5,
        layer_norm: bool = True,
        latent_dim: int = 10,
        similarity_adjust: bool = False,
        l2_norm: bool = True,
        activation: str = "gelu",
        batch_norm: bool = False,
        adjust_to_ce: bool = False,
        adjust_to_ultra: bool = False,
        acg_dropout: Optional[float] = None,
        wvf_dropout: Optional[float] = None,
        loss_flood: Optional[int] = None,
        use_RINCE_loss: bool = False,
        lam: float = 0.01,
        q: float = 0.1,
    ):
        super().__init__()
        if wvf_encoder is None:
            self.wvf_encoder = SingleWVFEncoder(
                adjust_to_ce=adjust_to_ce,
                adjust_to_ultra=adjust_to_ultra,
                wvf_dropout=wvf_dropout,
            )
        else:
            self.wvf_encoder = wvf_encoder

        if acg_encoder is None:
            self.acg_encoder = ConvolutionalEncoder(
                dim_rep=200,
                initialise=False,
                activation=activation,
                acg_dropout=acg_dropout,
            )
        else:
            self.acg_encoder = acg_encoder
        self.layer_norm = layer_norm

        wvf_rep_dim = self.wvf_encoder.dim_rep
        acg_rep_dim = self.acg_encoder.dim_rep

        self.wvf_projector = LinearProjector(
            wvf_rep_dim,
            latent_dim,
            apply_layer_norm=layer_norm,
            initialise=False,
            batch_norm=batch_norm,
        )
        self.acg_projector = LinearProjector(
            acg_rep_dim,
            latent_dim,
            apply_layer_norm=layer_norm,
            initialise=False,
            batch_norm=batch_norm,
        )

        # self.wvf_embedder = nn.Sequential(self.wvf_encoder, self.wvf_projector)
        # self.acg_embedder = nn.Sequential(self.acg_encoder, self.acg_projector)

        if temperature is None:
            self.train_temperature = True
            self.temperature = torch.nn.Parameter(torch.ones(1) * 0.1)
            self.temperature.requires_grad = True
        else:
            self.train_temperature = False
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature])
            )  # recheck this
            self.temperature.requires_grad = False

        if use_RINCE_loss:
            self.loss = RINCE_loss(self.temperature, lam=lam, q=q)
        else:
            self.loss = BiModalContractiveLoss(
                self.temperature,
                similarity_adjust=similarity_adjust,
                loss_flood=loss_flood,
            )
        self.l2_norm = l2_norm
        # print(self.l2_norm)

    def forward(self, wfs, acg):
        """
        wfs: torch.tensor of shape (N, 90)
        acg: torch.tensor of shape (N, 1, 10, 101)
        """
        wfs_embedding, acg_embedding = self.embed(wfs, acg)
        return self.loss(wfs_embedding, acg_embedding)

    def representation(self, wfs, acg):
        # call model.eval() before calling this function
        return self.wvf_encoder(wfs), self.acg_encoder(acg)

    def embed(self, wfs, acg):
        # call model.eval() before calling this function
        wfs_rep, acg_rep = self.representation(wfs, acg)
        wfs_embedding = self.wvf_projector(wfs_rep)
        acg_embedding = self.acg_projector(acg_rep)
        # print(self.l2_norm)
        if self.l2_norm:
            wfs_embedding = wfs_embedding / wfs_embedding.norm(dim=1, keepdim=True)
            acg_embedding = acg_embedding / acg_embedding.norm(dim=1, keepdim=True)

        return wfs_embedding, acg_embedding


class SimclrEmbeddingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        temperature=0.5,
        layer_norm: bool = True,
        latent_dim: int = 10,
        similarity_adjust: bool = False,
        l2_norm: bool = True,
        activation: str = "gelu",
        modality: str = "wvf",
        adjust_to_ce: bool = False,
    ):
        super().__init__()

        assert modality in ["wvf", "acg"], "modality must be either wvf or acg"

        if encoder is None:
            if modality == "wvf":
                self.encoder = SingleWVFEncoder(adjust_to_ce=adjust_to_ce)
            else:
                self.encoder = ConvolutionalEncoder(
                    dim_rep=200, initialise=False, activation=activation
                )
        else:
            self.encoder = encoder
        self.layer_norm = layer_norm

        rep_dim = self.encoder.dim_rep

        self.linear_projector = LinearProjector(
            rep_dim, latent_dim, apply_layer_norm=layer_norm, initialise=False
        )

        if temperature is None:
            self.train_temperature = True
            self.temperature = torch.nn.Parameter(torch.ones(1) * 0.1)
            self.temperature.requires_grad = True
        else:
            self.train_temperature = False
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature])
            )  # recheck this
            self.temperature.requires_grad = False

        self.loss = SimclrContractiveLoss(
            self.temperature, similarity_adjust=similarity_adjust
        )
        self.l2_norm = l2_norm
        # print(self.l2_norm)

    def forward(self, view1, view2):
        """
        wfs: torch.tensor of shape (N, 90)
        acg: torch.tensor of shape (N, 1, 10, 101)
        """
        embedding1, embedding2 = self.embed(view1, view2)
        return self.loss(embedding1, embedding2)

    def representation(self, view1, view2):
        # call model.eval() before calling this function
        return self.encoder(view1), self.encoder(view2)

    def embed(self, view1, view2):
        # call model.eval() before calling this function
        rep1, rep2 = self.representation(view1, view2)
        embedding1 = self.linear_projector(rep1)
        embedding2 = self.linear_projector(rep2)
        # print(self.l2_norm)
        if self.l2_norm:
            embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
            embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)
        return embedding1, embedding2
