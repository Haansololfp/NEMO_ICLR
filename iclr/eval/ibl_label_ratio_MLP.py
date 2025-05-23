import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import numpy as np
from celltype_ibl.utils.c4_vae_util import VAEEncoder, vae_encode_model, load_acg_vae
from celltype_ibl.models.BiModalEmbedding import (
    BimodalEmbeddingModel,
    SimclrEmbeddingModel,
)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from celltype_ibl.utils.ibl_data_util import (
    get_ibl_wvf_acg_pairs,
)
from npyx.c4.dataset_init import BIN_SIZE, WIN_SIZE
from npyx.c4.dl_utils import (
    load_waveform_encoder,
)
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import csv
import argparse
import glob
import pdb
from celltype_ibl.params.config import (
    WVF_ENCODER_ARGS_SINGLE,
    CLIP_MODEL_DIR,
    VAE_DIR,
    SIMCLR_ACG_DIR,
    SIMCLR_WVF_DIR,
)
from celltype_ibl.utils.ibl_data_util import IBL_acg_wvf_Dataset
from celltype_ibl.utils.ibl_eval_utils import ibl_representation_MLP_classifier


def label_ratio_sweep(
    train_wvf: np.ndarray,
    train_acg: np.ndarray,
    train_labels: np.ndarray,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    save_path: str,
    device: torch.device,
    modality: str = "both",
    embedding_model: str = "contrastive",
    n_epochs: int = 1000,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    label_ratios: list[float] = [0.5],
    fine_tuning: bool = False,
    verbose: bool = True,
    seed: int = 42,
) -> dict:
    """
    Train the model with different label ratios
    """
    test_accs = []
    test_f1s = []
    for label_ratio in label_ratios:
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.random.manual_seed(seed)
        # Initialize the model

        model = ibl_representation_MLP_classifier(
            encode_model=initialize_encoder(
                embedding_model, seed=seed, modality=modality
            ),
            embedding_model=embedding_model,
            n_classes=10,
            freeze_encoder_weights=not fine_tuning,
            modality=modality,
        )
        model.to(device)

        # Split the training data according to the label_ratio
        if label_ratio < 1.0:
            (
                train_idx,
                _,
            ) = train_test_split(
                np.arange(len(train_labels)),
                stratify=train_labels,
                train_size=label_ratio,
                random_state=42,
            )
        else:
            train_idx = np.arange(len(train_labels))

        train_loader = DataLoader(
            IBL_acg_wvf_Dataset(
                train_wvf[train_idx], train_acg[train_idx], train_labels[train_idx]
            ),
            batch_size=128,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if (embedding_model == "supervise") | (embedding_model == "suervise_flatten"):
            scheduler = CosineAnnealingWarmRestarts(optimizer, 20, 1, last_epoch=-1)
        else:
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)  # 0.1)

        weight = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels[train_idx],
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weight, dtype=torch.float32).to(device)
        )

        highest_acc = 0
        all_loss = []
        all_val_accs = []
        for epoch in range(n_epochs):
            model.train()
            loss_temp = 0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                loss_temp += loss.cpu().detach().numpy() * len(y)
            all_loss.append(loss_temp / len(train_idx))

            if verbose:
                print(
                    f"Epoch {epoch+1}/{n_epochs}, Label ratio: {label_ratio}, Train loss: {loss.cpu().detach().numpy()}"
                )

            # pdb.set_trace()

            model.eval()
            with torch.no_grad():
                pred_all = []
                target_all = []
                for x, y in validation_loader:
                    x = x.to(device)
                    outputs = model(x)
                    _, pred = torch.max(outputs.data, 1)
                    pred_all.append(pred.cpu().detach().numpy())
                    target_all.append(y)

                val_acc = balanced_accuracy_score(
                    np.concatenate(target_all), np.concatenate(pred_all)
                )
                all_val_accs.append(val_acc)

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{n_epochs}, Label ratio: {label_ratio}, Validation accuracy: {val_acc}"
                    )

            if val_acc > highest_acc:
                highest_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    format_float_for_filename(label_ratio, save_path),
                )

            scheduler.step()

        best_checkpoint = torch.load(format_float_for_filename(label_ratio, save_path))
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pred_all = []
            target_all = []
            for x, y in test_loader:
                x = x.to(device)
                outputs = model(x)
                _, pred = torch.max(outputs.data, 1)
                pred_all.append(pred.cpu().detach().numpy())
                target_all.append(y)

            test_acc = balanced_accuracy_score(
                np.concatenate(target_all), np.concatenate(pred_all)
            )
            test_f1 = f1_score(
                np.concatenate(target_all), np.concatenate(pred_all), average="macro"
            )
            test_accs.append(test_acc)
            test_f1s.append(test_f1)

        fig, axs = plt.subplots(2, 1, figsize=[10, 10])
        axs[0].plot(all_loss)
        axs[0].set_title("Training loss")
        axs[1].plot(all_val_accs)
        axs[1].set_title("Validation accuracy")
        formatted_number = f"{label_ratio:.2f}"
        # Replace commas and period with underscores
        formatted_number = formatted_number.replace(".", "_")
        plt.savefig(f"{save_path}/label_ratio_{formatted_number}.png")

    return {
        "label_ratio": label_ratios,
        "test_accs": test_accs,
        "test_f1s": test_f1s,
    }


def initialize_encoder(
    embedding_model: str = "contrastive",
    latent_dim: int = 512,
    seed: int = 0,
    modality: str = "both",
):
    simclr_seed = [42, 26, 29, 65, 70]
    if (embedding_model == "contrastive") | (embedding_model == "supervise"):
        # Load the logistic regression logits
        encode_model = BimodalEmbeddingModel(
            layer_norm=False,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
        )
        if embedding_model == "contrastive":
            model_path = CLIP_MODEL_DIR + f"/seed_{seed + 42}_checkpoint.pt"
            checkpoint = torch.load(model_path)
            encode_model.load_state_dict(checkpoint["model_state_dict"])
        encode_model.eval()
    elif embedding_model == "vae":
        vae_acg_path = f"{VAE_DIR}/3DACG_logscale_seed_{seed + 1234}_encoder_gelu.pt"
        acg_vae = load_acg_vae(
            vae_acg_path,
            WIN_SIZE // 2,
            BIN_SIZE,
            initialise=True,
            pool="avg",
            activation="gelu",
        )
        acg_head = VAEEncoder(acg_vae.encoder, 10)  # maybe change this?

        vae_wvf_path = f"{VAE_DIR}/wvf_singlechannel_seed_{seed + 1234}_encoder.pt"
        wvf_vae = load_waveform_encoder(
            WVF_ENCODER_ARGS_SINGLE,
            vae_wvf_path,
            in_features=90,
            initialise=True,
        )
        wvf_head = VAEEncoder(wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"])

        encode_model = vae_encode_model(wvf_head, acg_head)

    elif embedding_model == "simclr":
        simclr_wvf_patttern = (
            SIMCLR_WVF_DIR + f"/checkpoint_acc_{simclr_seed[seed]}_epoch_*.pt"
        )
        simclr_wvf_path = glob.glob(simclr_wvf_patttern)[0]

        simclr_acg_patttern = (
            SIMCLR_ACG_DIR + f"/checkpoint_acc_{simclr_seed[seed]}_epoch_*.pt"
        )
        simclr_ACG_path = glob.glob(simclr_acg_patttern)[0]

        wvf_model = SimclrEmbeddingModel(
            latent_dim=512, layer_norm=False, modality="wvf"
        )
        acg_model = SimclrEmbeddingModel(
            latent_dim=512, layer_norm=False, modality="acg"
        )

        wvf_model.load_state_dict(torch.load(simclr_wvf_path)["model_state_dict"])
        acg_model.load_state_dict(torch.load(simclr_ACG_path)["model_state_dict"])

        encode_model = BimodalEmbeddingModel(
            wvf_model.encoder, acg_model.encoder, latent_dim=512, layer_norm=False
        )

    else:
        raise ValueError("Unknown embedding model")
    if modality == "both":
        return encode_model
    elif modality == "wvf":
        return encode_model.wvf_encoder
    elif modality == "acg":
        return encode_model.acg_encoder


def format_float_for_filename(value: float, save_path: str) -> str:
    # Convert the float to a string with comma separators
    formatted_number = f"{value:.2f}"  # Example format: '1,234,567.89'

    # Replace commas and period with underscores
    formatted_number = formatted_number.replace(".", "_")

    return save_path + f"/best_model_{formatted_number}.pt"


parser = argparse.ArgumentParser(description="ibl_label_ratio_sweep")
parser.add_argument("--embedding_model", type=str, default="contrastive")
parser.add_argument("--fine_tuning", action="store_false", default=True)
parser.add_argument(
    "--label_ratios", type=float, nargs="+", default=[0.01, 0.1, 0.3, 0.5, 0.8, 1.0]
)
parser.add_argument("--modality", type=str, default="both")
parser.add_argument("--i_start", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    fine_tuning = args.fine_tuning
    embedding_model = args.embedding_model
    modality = args.modality
    i_start = args.i_start
    training_fold = [0, 1, 2, 5, 7, 8, 9]
    validation_fold = [4]
    test_set = [3, 6]

    simclr_seed = [42, 26, 29, 65, 70]

    # Load the dataset
    ibl_wvf, ibl_acg, cosmos_region, fold_idx = get_ibl_wvf_acg_pairs(
        return_region="cosmos"
    )
    ibl_wvf = ibl_wvf.astype("float32")
    ibl_acg = ibl_acg.astype("float32")
    training_idx = [
        index for index, element in enumerate(fold_idx) if element in training_fold
    ]
    validation_idx = [
        index for index, element in enumerate(fold_idx) if element in validation_fold
    ]
    test_idx = [index for index, element in enumerate(fold_idx) if element in test_set]

    unique_labels, train_region_idx = np.unique(
        cosmos_region[training_idx], return_inverse=True
    )
    labelling = {l: i for i, l in enumerate(unique_labels)}
    val_region_index = [labelling[label] for label in cosmos_region[validation_idx]]
    test_region_index = [labelling[label] for label in cosmos_region[test_idx]]

    test_dataset = IBL_acg_wvf_Dataset(
        ibl_wvf[test_idx], ibl_acg[test_idx], test_region_index
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    validation_dataset = IBL_acg_wvf_Dataset(
        ibl_wvf[validation_idx], ibl_acg[validation_idx], val_region_index
    )
    validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

    # save_dir = (
    #     "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_label_ratio_sweep/vae_fix"
    # )
    save_dir = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_label_ratio_sweep/temporary"

    for i in range(i_start, 5):
        if embedding_model == "vae":
            seed = 1234 + i
        elif embedding_model == "simclr":
            seed = simclr_seed[i]
        else:
            seed = 42 + i
        save_path = f"{save_dir}/" + f"{embedding_model}3_seed_{seed}"
        if not fine_tuning:
            save_path += "freeze"
        if modality != "both":
            save_path += modality
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        lr = 0.00001
        # Train the model
        if embedding_model == "supervise":
            lr = 0.0001
            n_epochs = 5000
        elif embedding_model == "supervise_flatten":
            n_epochs = 2000
            lr = 0.001
        else:
            n_epochs = 3000

        sweep_results = label_ratio_sweep(
            ibl_wvf[training_idx],
            ibl_acg[training_idx],
            train_region_idx,
            validation_loader,
            test_loader,
            save_path=save_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            modality=modality,
            embedding_model=embedding_model,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=0,  # 0.2,
            label_ratios=args.label_ratios,
            fine_tuning=fine_tuning,
            verbose=True,
            seed=i,
        )

        # Specify the filename
        if not fine_tuning:
            filename = (
                save_dir
                + f"/label_ratio_sweep_{embedding_model}3_seed_{seed}_freeze_{modality}.csv"
            )
        else:
            filename = (
                save_dir
                + f"/label_ratio_sweep_{embedding_model}3_seed_{seed}_{modality}.csv"
            )

        # Write the dictionary to a CSV file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sweep_results.keys())
            writer.writeheader()
            # Writing data rows
            writer.writerows(
                [
                    dict(zip(sweep_results.keys(), row))
                    for row in zip(*sweep_results.values())
                ]
            )

        print(f"Data saved to {filename}")
