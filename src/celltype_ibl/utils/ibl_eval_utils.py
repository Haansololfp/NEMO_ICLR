import torch
import torch.nn as nn
import torch.nn.functional as F


class ibl_representation_MLP_classifier(nn.Module):
    def __init__(
        self,
        encode_model: nn.Module,
        embedding_model: str = "contrastive",
        n_classes: int = 10,
        freeze_encoder_weights: bool = True,
        use_final_embed: bool = False,
        modality: str = "both",
    ) -> None:
        super().__init__()
        self.encode_model = encode_model
        self.embedding_model = embedding_model
        if freeze_encoder_weights:
            for param in self.encode_model.parameters():
                param.requires_grad = False
        print(modality)
        if modality == "both":
            n_input = 500
        elif modality == "wvf":
            n_input = 300
        elif modality == "acg":
            n_input = 200
        else:
            raise ValueError("Unknown modality")
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.LazyLinear(n_classes)
        self.use_final_embed = use_final_embed

    def forward(
        self, x: torch.FloatTensor | torch.cuda.FloatTensor
    ) -> torch.FloatTensor | torch.cuda.FloatTensor:
        acg = x[:, :1010]
        wvf = x[:, 1010:]
        n_input = self.fc1.in_features

        if (
            (self.embedding_model == "contrastive")
            | (self.embedding_model == "supervise")
            | (self.embedding_model == "simclr")
        ):
            if self.use_final_embed:
                assert n_input == 500
                wvf_rep, acg_rep = self.encode_model.embed(
                    wvf, acg.reshape(-1, 1, 10, 101)
                )
            else:
                if n_input == 500:
                    wvf_rep, acg_rep = self.encode_model.representation(
                        wvf, acg.reshape(-1, 1, 10, 101)
                    )
                    x = torch.cat([wvf_rep, acg_rep], dim=1)
                elif n_input == 300:
                    x = self.encode_model(wvf)
                elif n_input == 200:
                    x = self.encode_model(acg.reshape(-1, 1, 10, 101))

        elif self.embedding_model == "vae":  # | (self.embedding_model == "supervise"):
            if n_input == 500:
                wvf_rep, acg_rep = self.encode_model.embed(
                    wvf,
                    acg.reshape(-1, 1, 10, 101),
                    return_pre_projection=not self.use_final_embed,
                )
                x = torch.cat([wvf_rep, acg_rep], dim=1)
            elif n_input == 300:
                x = self.encode_model(
                    wvf, return_pre_projection=not self.use_final_embed
                )
            elif n_input == 200:
                x = self.encode_model(
                    acg.reshape(-1, 1, 10, 101),
                    return_pre_projection=not self.use_final_embed,
                    isacg=True,
                )
        else:
            raise ValueError("Unknown embedding model")

        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.2)(x)
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        pred = self.forward(x)
        return F.softmax(pred, dim=1).detach().numpy()
