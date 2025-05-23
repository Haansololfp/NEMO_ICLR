import numpy as np
import torch
import os
import json
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.models.ACG_augmentation_dataloader import (
    EmbeddingDataset,
)
from celltype_ibl.utils.c4_data_utils import (
    get_c4_unlabelled_wvf_acg_pairs,
    get_c4_labeled_dataset,
)
from celltype_ibl.utils.ultra_data_util import get_ultra_wvf_acg_pairs
from celltype_ibl.utils.kenji_allen_data_utils import get_kenji_allen_wvf_acg_pairs
from celltype_ibl.models.metrics import topk, rankme
import celltype_ibl.models.linear_classifier as linear_probe
from celltype_ibl.utils.ibl_data_util import get_ibl_wvf_acg_pairs
from celltype_ibl.utils.cell_explorer_data_util import (
    get_cell_explorer_dataset,
    data_load_by_split,
    get_allen_unlabeled_dataset,
    get_allen_labeled_dataset,
)

from celltype_ibl.utils.visualize_util import umap_from_embedding
from celltype_ibl.params.set_params import parse_base_args
from celltype_ibl.params.config import (
    WANDB_DIR,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from celltype_ibl.models.linear_probe import linear_probe_train_val
from celltype_ibl.utils.wandb import wandblog
import wandb
from datetime import datetime
import git

FLIP = True

# BLEEP: --similarity_adjust: Ture
#        --l2_norm: False
#        --layer_norm: True
# CLIP:  --similarity_adjust: False
#        --l2_norm: True
#        --layer_norm: False


def format_saved_model_name(args) -> str:
    # Format the model name to reflect the arguments
    model_name_parts = [
        f"temp{'T' if args.temperature is None else 'F'}",
        # f"lnorm{'T' if args.layer_norm else 'F'}",
        f"dim{args.dim_embed}",
        # f"simadj{'T' if args.similarity_adjust else 'F'}",
        f"aug{'T' if args.augmentation else 'F'}",
        # f"l2norm{'T' if args.l2_norm else 'F'}",
        f"batch{args.batch_size}",
        f"act{args.activation}",
        f"data_{args.dataset}",
        f"seed{args.seed}",
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    ]
    if args.dataset == "c4":
        model_name_parts.append(f"h5{'T' if args.from_h5 else 'F'}")
    elif args.dataset == "ibl":
        model_name_parts.append(f"root{'T' if args.with_root else 'F'}")
        model_name_parts.append(f"heldout{'T' if args.held_out else 'F'}")
    elif args.dataset == "cell_explorer":
        model_name_parts.append(f"_split_{args.split_id}")
    elif args.dataset == "kenji_allen":
        model_name_parts.append(f"_{args.project}")
    if args.initialise:
        model_name_parts.append("init")
    if args.acg_normalization:
        model_name_parts.append("acg_norm")
    if args.batch_norm:
        model_name_parts.append("batch_norm")
    if args.adjust_to_ce:
        model_name_parts.append("ATC")
    if args.adjust_to_ultra:
        model_name_parts.append("ATU")

    # Join the parts with underscores
    model_name = "_".join(model_name_parts)
    return model_name


def load_train_dataset(args, dataset: str = "c4"):
    """
    Load the training dataset.
    """
    fold_idx = None
    train_labels = None
    # load data from one of the dataset
    if dataset == "c4":
        wvf_raw, acg_raw = get_c4_unlabelled_wvf_acg_pairs(from_h5=args.from_h5)
    elif dataset == "ibl":
        wvf_raw, acg_raw, train_labels, fold_idx = get_ibl_wvf_acg_pairs(
            with_root=args.with_root,
            return_region="cosmos",
            adjust_to_ce=args.adjust_to_ce,
        )

    elif dataset == "ibl_repeated":
        wvf_raw, acg_raw = get_ibl_wvf_acg_pairs(repeated_sites=True)
    elif dataset == "allen":
        wvf_raw, acg_raw = get_allen_unlabeled_dataset()
    elif dataset == "allen_labelled":
        wvf_raw, acg_raw, _ = get_allen_labeled_dataset()
    elif dataset == "cell_explorer":
        wvf_raw, acg_raw, train_labels = data_load_by_split(split_id=args.split_id)
    elif dataset == "Ultra":
        wvf_raw, acg_raw, _, _ = get_ultra_wvf_acg_pairs(return_optotagged=False)
    elif dataset == "kenji_allen":
        wvf_raw, acg_raw, _, _ = get_kenji_allen_wvf_acg_pairs(
            return_optotagged=False, selected_project=args.project
        )
    else:
        raise ValueError("Dataset not recognised")
    return wvf_raw, acg_raw, train_labels, fold_idx


def split_validation_set(
    args,
    wvf_raw: np.ndarray,
    acg_raw: np.ndarray,
    args_dict: dict,
    fold_idx: np.ndarray | None,
    labels: np.ndarray | None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    """Split the dataset into training and validation sets."""
    label_val = None
    if (args.dataset != "ibl") or (not args.held_out):
        if labels is not None:
            wvf_train, wvf_val, acg_train, acg_val, label_train, label_val = (
                train_test_split(
                    wvf_raw,
                    acg_raw,
                    labels,
                    test_size=0.1,
                    random_state=42,
                    stratify=labels,
                )
            )
        else:
            wvf_train, wvf_val, acg_train, acg_val = train_test_split(
                wvf_raw, acg_raw, test_size=0.2, random_state=42
            )
            label_val = None
            label_train = None
    else:
        assert fold_idx is not None
        n_fold = len(np.unique(fold_idx))
        ind = np.random.choice(range(n_fold), size=(n_fold,), replace=False)
        n_train = round(n_fold * 0.7)
        n_val = round(n_fold * 0.1)
        train_idx = ind[:n_train]
        val_idx = ind[n_train : n_train + n_val]
        test_idx = ind[n_train + n_val :]
        val_idx = np.array([4])
        test_idx = np.array([3, 6])
        train_idx = np.array([0, 1, 2, 5, 7, 8, 9])
        wvf_train = wvf_raw[np.isin(fold_idx, train_idx)]
        acg_train = acg_raw[np.isin(fold_idx, train_idx)]
        wvf_val = wvf_raw[np.isin(fold_idx, val_idx)]
        acg_val = acg_raw[np.isin(fold_idx, val_idx)]
        if labels is not None:
            label_val = labels[np.isin(fold_idx, val_idx)]
            label_train = labels[np.isin(fold_idx, train_idx)]
        args_dict["test_idx"] = test_idx.tolist()
        args_dict["val_idx"] = val_idx.tolist()
    return wvf_train, wvf_val, acg_train, acg_val, label_train, label_val


def normalize_acg(acg_data):
    """Normalize ACG data."""
    return acg_data / np.max(acg_data.reshape(-1, 1010), axis=1)[:, None, None]


def concatenate_dataset(acg_data, wvf_data):
    """Concatenate ACG and waveform data into a test dataset."""
    return np.concatenate((acg_data.reshape(-1, 1010) * 10, wvf_data), axis=1).astype(
        "float32"
    )


def get_labels_info(labels):
    """Generate label indices and correspondence dictionaries."""
    unique_labels, label_idx = np.unique(labels, return_inverse=True)
    correspondence = {i: l for i, l in enumerate(unique_labels)}
    labelling = {l: i for i, l in enumerate(unique_labels)}
    return label_idx, correspondence, labelling


def get_linear_probe_testset(args, acg_val=None, wvf_val=None, label_val=None):
    """Get the test dataset for linear probe."""
    print(f"Loading {args.test_data} dataset for linear probe.")
    session_idx = []
    if args.test_data == "c4_labelled":
        wvf_data, acg_data, label_idx, _, labelling, correspondence = (
            get_c4_labeled_dataset(from_h5=args.from_h5)
        )
        loo = True
        # by_session = False
    elif args.test_data == "ibl":
        assert acg_val is not None and wvf_val is not None and label_val is not None
        acg_data = acg_val
        wvf_data = wvf_val
        label_idx, correspondence, labelling = get_labels_info(label_val)
        loo = False
        # by_session = False
    elif args.test_data == "allen":
        wvf_data, acg_data, labels = get_allen_labeled_dataset()
        label_idx, correspondence, labelling = get_labels_info(labels)
        loo = True
        # by_session = False
    elif args.test_data == "cell_explorer":
        wvf_data, acg_data, cell_types = get_cell_explorer_dataset(filt=True)
        label_idx, correspondence, labelling = get_labels_info(cell_types)
        loo = False
        # by_session = False
    elif args.test_data == "cell_explorer_cv":
        acg_data = acg_val
        wvf_data = wvf_val
        label_idx, correspondence, labelling = get_labels_info(label_val)
        loo = False
        # by_session = False
        # wvf_data, acg_data, cell_types = data_load_by_split(split_id=args.split_id)
        # label_idx, correspondence, labelling = get_labels_info(cell_types)
        # loo = False
    elif args.test_data == "Ultra":
        wvf_data, acg_data, labels, session_idx = get_ultra_wvf_acg_pairs()
        label_idx, correspondence, labelling = get_labels_info(labels)
        loo = False
        # by_session = False

    elif args.test_data == "kenji_allen":
        wvf_data, acg_data, labels, session_idx = get_kenji_allen_wvf_acg_pairs(
            return_id="session",
            selected_project=args.project,
        )
        label_idx, correspondence, labelling = get_labels_info(labels)
        loo = True
        # by_session = True

    else:
        raise ValueError("linear probe test data not recognised")

    if args.acg_normalization:
        acg_data = normalize_acg(acg_data)
    test_dataset = concatenate_dataset(acg_data, wvf_data)

    return (
        test_dataset,
        label_idx,
        labelling,
        correspondence,
        session_idx,
        loo,
        # by_session,
    )


def ibl_train_val_rep(
    model, wvf_tensor, acg_tensor, wvf_val_tensor, acg_val_tensor, representation=True
):
    train_wvf_acg_input = {}
    train_wvf_acg_input["wvf"] = wvf_tensor
    train_wvf_acg_input["acg"] = acg_tensor.reshape(-1, 1, 10, 101) * 10
    if representation:
        wvf_rep_train, acg_rep_train = model.representation(
            train_wvf_acg_input["wvf"], train_wvf_acg_input["acg"]
        )
    else:
        wvf_rep_train, acg_rep_train = model.embed(
            train_wvf_acg_input["wvf"], train_wvf_acg_input["acg"]
        )

    val_wvf_acg_input = {}
    val_wvf_acg_input["wvf"] = wvf_val_tensor
    val_wvf_acg_input["acg"] = acg_val_tensor.reshape(-1, 1, 10, 101) * 10
    if representation:
        wvf_rep_val, acg_rep_val = model.representation(
            val_wvf_acg_input["wvf"], val_wvf_acg_input["acg"]
        )
    else:
        wvf_rep_val, acg_rep_val = model.embed(
            val_wvf_acg_input["wvf"], val_wvf_acg_input["acg"]
        )
    training_data = (
        torch.concat((acg_rep_train, wvf_rep_train), dim=1).cpu().detach().numpy()
    )
    testing_data = (
        torch.concat((acg_rep_val, wvf_rep_val), dim=1).cpu().detach().numpy()
    )
    return training_data.squeeze(), testing_data.squeeze()


def plot_training_log(
    args,
    loss_all,
    contrastive_acc1,
    contrastive_acc2,
    contrastive_val_acc1,
    contrastive_val_acc2,
    temperature_all,
):
    # Update the plot
    if args.split_validate:
        if args.train_temperature:
            fig, ax = plt.subplots(4, 1, figsize=(10, 20))
        else:
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    else:
        if args.train_temperature:
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Contrastive Accuracy")
    ax[0].plot(np.array(loss_all), label="Training Loss")
    ax[0].legend()
    for k in args.top_k:
        ax[1].plot(
            np.array(contrastive_acc1[f"t{k}"]), label=f"Training Accuracy1, top{k}"
        )
        ax[1].plot(
            np.array(contrastive_acc2[f"t{k}"]), label=f"Training Accuracy2, top{k}"
        )
    ax[1].legend()
    plt_idx = 2

    if args.split_validate:

        ax[plt_idx].set_xlabel("Epoch")
        ax[plt_idx].set_ylabel("Accuracy")
        for k in args.top_k:
            ax[plt_idx].plot(
                np.array(contrastive_val_acc1[f"t{k}"]),
                label=f"Validation Accuracy1, top{k}",
            )
            ax[plt_idx].plot(
                np.array(contrastive_val_acc2[f"t{k}"]),
                label=f"Validation Accuracy2, top{k}",
            )
        ax[plt_idx].legend()
        plt_idx = 3

    if args.train_temperature:
        ax[plt_idx].plot(np.array(temperature_all), label="Temperature")
        ax[plt_idx].legend()

    return fig, ax


def train():
    args = parse_base_args()

    if args.train_temperature:
        args.temperature = None
    args_dict = vars(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # !!!wandb check whether the previously trained model has the same configuration.

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load training dataset

    wvf_raw, acg_raw, train_labels, fold_idx = load_train_dataset(args, args.dataset)

    # if args.dataset == "cell_explorer":
    #     args.split_validate = False

    if args.split_validate:
        wvf_train, wvf_val, acg_train, acg_val, label_train, label_val = (
            split_validation_set(
                args, wvf_raw, acg_raw, args_dict, fold_idx, train_labels
            )
        )

        wvf_val_tensor = torch.from_numpy(wvf_val.astype("float32")).clone().to(device)
        if args.acg_normalization:
            acg_val = acg_val / np.max(acg_val.reshape(-1, 1010), axis=1)[:, None, None]

        acg_val_tensor = torch.from_numpy(acg_val.astype("float32")).clone().to(device)

        del wvf_raw, acg_raw
    else:
        acg_val, wvf_val, label_val = [], [], []
        wvf_train, acg_train, label_train = wvf_raw, acg_raw, train_labels

    if args.save:
        model_name = format_saved_model_name(args)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        args_dict["git_hash"] = sha
        if args.use_wandb:
            wb = wandblog(
                name=model_name,
                config=args_dict,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                # project="bimodal contrastive neuron embedding",
                topk=topk,
                wandb_dir=WANDB_DIR,
            )

        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir + "/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
        print(f"Arguments saved to {output_dir}/args.json")

    if args.linear_probe:
        (
            test_dataset,
            label_idx,
            labelling,
            correspondence,
            session_idx,
            loo,
            # by_session,
        ) = get_linear_probe_testset(args, acg_val, wvf_val, label_val)

    by_session = args.by_session
    if by_session:
        loo = False

    # prepare the training dataset
    dataset = EmbeddingDataset(
        wvf_train,
        acg_train,
        augmentation=args.augmentation,
        acg_normalize=args.acg_normalization,
        std=args.std,
    )

    embedding_dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True
    )

    model = BimodalEmbeddingModel(
        temperature=args.temperature,
        layer_norm=args.layer_norm,
        latent_dim=args.dim_embed,
        similarity_adjust=args.similarity_adjust,
        l2_norm=args.l2_norm,
        activation=args.activation,
        batch_norm=args.batch_norm,
        adjust_to_ce=args.adjust_to_ce,
        adjust_to_ultra=args.adjust_to_ultra,
        acg_dropout=args.acg_dropout,
        wvf_dropout=args.wvf_dropout,
        loss_flood=args.loss_flood,
        use_RINCE_loss=args.use_RINCE_loss,
        lam=args.lam,
        q=args.rince_q,
    )

    print("initialize to train...")
    start_epoch = 0
    highest_acc = 0
    highest_F1 = 0
    loss_all = []
    contrastive_acc1 = {}
    contrastive_acc2 = {}
    # initialise the top k accuracy for validation set
    best_val_top_k = {}
    contrastive_val_acc1 = {}
    contrastive_val_acc2 = {}
    for k in args.top_k:
        contrastive_acc1[f"t{k}"] = []
        contrastive_acc2[f"t{k}"] = []
        contrastive_val_acc1[f"t{k}"] = []
        contrastive_val_acc2[f"t{k}"] = []
        best_val_top_k[f"t{k}"] = 0
    temperature_all = []
    N = wvf_train.shape[0]

    model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 20, 1, last_epoch=-1
    )

    # continue training from an existing model
    if (args.load_pt_path is not None) and not args.initialise:
        # if use use the same datset, check whether we are using the same training and validation set as the previous model
        # if not, raise an error
        # leave for future implementation
        optimizer.load_state_dict(torch.load(args.load_pt_path)["optimizer_state_dict"])
        print("Optimizer loaded from", args.load_pt_path)
        loss_all = list(torch.load(args.load_pt_path)["loss"])
        start_epoch = len(loss_all)

        for k in args.top_k:
            contrastive_acc1["t" + str(k)] = list(
                torch.load(args.load_pt_path)[f"training_acc1t{k}"]
            )
            contrastive_acc2["t" + str(k)] = list(
                torch.load(args.load_pt_path)[f"training_acc2t{k}"]
            )
        temperature_all = list(torch.load(args.load_pt_path)["temperature_all"])
        if args.split_validate:
            for k in args.top_k:
                contrastive_val_acc1["t" + str(k)] = list(
                    torch.load(args.load_pt_path)[f"validation_acc1t{k}"]
                )
                contrastive_val_acc2["t" + str(k)] = list(
                    torch.load(args.load_pt_path)[f"validation_acc2t{k}"]
                )

    model.eval()
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.split_validate:
            val_loss = model(
                wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10
            )
        model.train()
        tmp_losses = []
        for idx, (wvf_batch, acg_batch) in enumerate(embedding_dataloader):
            optimizer.zero_grad()
            loss = model(
                wvf_batch.to(device),
                acg_batch.reshape(-1, 1, 10, 101).to(device),
            )
            loss.backward()
            optimizer.step()
            tmp_losses.append(loss.cpu().detach().numpy() * wvf_batch.size(dim=0))
        loss_all.append(sum(tmp_losses) / N)

        # print(
        #     f"Epoch {epoch}, Loss: {loss_all[-1]}",
        # )  # Print the loss
        if args.use_wandb:
            currlog = {
                "Loss": loss_all[-1],
            }
            if args.split_validate:
                currlog["Val_loss"] = val_loss.cpu().detach().numpy()
        scheduler.step()

        temperature_all.append(model.temperature.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            wvf_embed, acg_embed = model.embed(
                torch.from_numpy(wvf_train.astype("float32")).to(device),
                torch.from_numpy(acg_train.astype("float32"))
                .to(device)
                .reshape(-1, 1, 10, 101)
                * 10,
            )

        similarity_matrix = torch.matmul(wvf_embed, acg_embed.transpose(0, 1))
        labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
        rankme_loss = rankme(torch.cat((wvf_embed, acg_embed), dim=1))
        if args.use_wandb:
            currlog["rankme_loss"] = rankme_loss.cpu().detach().numpy()

        del wvf_embed, acg_embed

        torch.cuda.empty_cache()
        # compute top k accuracy for the training set
        for k in args.top_k:
            topk1 = topk(similarity_matrix, labels, k=k)
            topk2 = topk(torch.transpose(similarity_matrix, 0, 1), labels, k=k)
            contrastive_acc1[f"t{k}"].append(topk1.cpu().detach().numpy())
            contrastive_acc2[f"t{k}"].append(topk2.cpu().detach().numpy())
            if args.use_wandb:
                currlog[f"contrastive_acc1_t{k}"] = topk1.cpu().detach().numpy()
                currlog[f"contrastive_acc2_t{k}"] = topk2.cpu().detach().numpy()

        if args.split_validate:
            with torch.no_grad():
                wvf_embed_val, acg_embed_val = model.embed(
                    wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10
                )
            similarity_matrix_val = torch.matmul(
                wvf_embed_val, acg_embed_val.transpose(0, 1)
            )
            labels_val = torch.arange(len(similarity_matrix_val)).to(
                similarity_matrix_val.device
            )
            rankme_loss_val = rankme(torch.cat((wvf_embed_val, acg_embed_val), dim=1))
            if args.use_wandb:
                currlog["rankme_loss_val"] = rankme_loss_val.cpu().detach().numpy()

            # calculate top k accuracy for validation set
            for k in args.top_k:
                topk1_val = (
                    topk(similarity_matrix_val, labels_val, k=k).cpu().detach().numpy()
                )
                topk2_val = (
                    topk(torch.transpose(similarity_matrix_val, 0, 1), labels_val, k=k)
                    .cpu()
                    .detach()
                    .numpy()
                )
                contrastive_val_acc1[f"t{k}"].append(topk1_val)
                contrastive_val_acc2[f"t{k}"].append(topk2_val)
                if args.use_wandb:
                    currlog[f"contrastive_val_acc1_t{k}"] = topk1_val
                    currlog[f"contrastive_val_acc2_t{k}"] = topk2_val
                avg = (topk1_val + topk2_val) / 2

                if avg > best_val_top_k[f"t{k}"]:
                    best_val_top_k[f"t{k}"] = avg

                    checkpoint_path = os.path.join(
                        output_dir, "best_top{}_checkpoint.pt".format(k)
                    )

                    data_save = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": np.array(loss_all),
                        "training_acc1": contrastive_acc1,
                        "training_acc2": contrastive_acc2,
                        "validation_acc1": contrastive_val_acc1,
                        "validation_acc2": contrastive_val_acc2,
                        "temperature_all": np.array(temperature_all),
                        "best_epoch": epoch,
                    }
                    torch.save(
                        data_save,
                        checkpoint_path,
                    )
                    if args.use_wandb:
                        wb.log_weights(checkpoint_path, k)

        if not args.use_wandb:
            fig, _ = plot_training_log(
                args,
                loss_all,
                contrastive_acc1,
                contrastive_acc2,
                contrastive_val_acc1,
                contrastive_val_acc2,
                temperature_all,
            )
            fig.savefig(os.path.join(output_dir, "training_log.png"))
            plt.close(fig)

        if (
            epoch % args.log_every_n_steps == 0
            or epoch == start_epoch + args.num_epochs - 1
        ):
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            data_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.array(loss_all),
                "training_acc1": contrastive_acc1,
                "training_acc2": contrastive_acc2,
                "temperature_all": np.array(temperature_all),
            }
            if args.split_validate:
                data_save["validation_acc1"] = contrastive_val_acc1
                data_save["validation_acc2"] = contrastive_val_acc2
            torch.save(
                data_save,
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")
            if args.linear_probe:
                if (args.test_data == "ibl") | (
                    args.test_data == "cell_explorer_cv"
                ):  #################
                    wvf_tensor = torch.from_numpy(wvf_train.astype("float32")).to(
                        device
                    )
                    acg_tensor = torch.from_numpy(acg_train.astype("float32")).to(
                        device
                    )
                    training_data, testing_data = ibl_train_val_rep(
                        model,
                        wvf_tensor,
                        acg_tensor,
                        wvf_val_tensor,
                        acg_val_tensor,
                    )
                    torch.cuda.empty_cache()
                    cv_path, img_path = linear_probe_train_val(
                        os.path.join(output_dir, "linear_probe_epoch_{}".format(epoch)),
                        training_data,
                        testing_data,
                        label_train,
                        label_val,
                        labelling,
                    )
                else:
                    cv_path, img_path, _ = linear_probe.main(
                        test_dataset,
                        label_idx,
                        labelling,
                        correspondence,
                        os.path.join(output_dir, "linear_probe_epoch_{}".format(epoch)),
                        session_idx=session_idx,
                        embedding_model_path=checkpoint_path,
                        loo=loo,
                        by_session=by_session,
                        n_runs=args.n_runs,
                        embedding_model=args.model,
                        layer_norm=args.layer_norm,
                        latent_dim=args.dim_embed,
                        l2_norm=args.l2_norm,
                        activation=args.activation,
                        batch_norm=args.batch_norm,
                        seed=args.seed,
                        undersample=args.undersample,
                        use_smote=args.use_smote,
                    )
                im = plt.imread(img_path)
                if args.use_wandb:
                    # if not args.sweep:
                    wb.log_results(cv_path, epoch)
                    currlog["confusion_matrix"] = [wandb.Image(im)]

                with open(cv_path, "rb") as f:
                    data = pickle.load(f)
                    # f1_score = np.array(data["f1_scores"])
                    true_classes = data["true_targets"]
                    pred_classes = data["predicted_classes"]
                acc = balanced_accuracy_score(true_classes, pred_classes)
                f1 = f1_score(true_classes, pred_classes, average="macro")
                # if len(f1_score) > 1:
                #     f1_score = np.mean(f1_score)
                if args.use_wandb:
                    # currlog["f1_score"] = f1_score.squeeze()
                    currlog["f1_score"] = f1
                    currlog["balanced_accuracy"] = acc
                    if acc > highest_acc:
                        highest_acc = acc
                        currlog["highest_acc"] = acc
                    if f1 > highest_F1:
                        # highest_F1 = f1_score
                        # currlog["highest_F1"] = f1_score
                        highest_F1 = f1
                        currlog["highest_F1"] = f1

            if args.embedding_viz:
                if args.test_data == "ibl":
                    wvf_rep_val, acg_rep_val = model.representation(
                        wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10
                    )
                    wvf_rep = wvf_rep_val
                    acg_rep = acg_rep_val
                    labels = np.array([correspondence[i] for i in label_idx])
                    dot_size = 1
                elif (
                    (args.dataset == "c4")
                    | (args.test_data == "cell_explorer")
                    | (args.test_data == "cell_explorer_cv")
                    | (args.test_data == "Ultra")
                    | (args.test_data == "kenji_allen")
                ):
                    acg_viz = test_dataset[:, :1010].reshape(-1, 1, 10, 101)
                    wvf_viz = test_dataset[:, 1010:]
                    wvf_acg_input = {}
                    wvf_acg_input["wvf"] = torch.tensor(wvf_viz).to(device)
                    wvf_acg_input["acg"] = torch.tensor(
                        acg_viz.reshape(-1, 1, 10, 101)
                    ).to(device)
                    wvf_rep, acg_rep = model.representation(
                        wvf_acg_input["wvf"], wvf_acg_input["acg"]
                    )
                    labels = np.array([correspondence[i] for i in label_idx])
                    dot_size = 3
                else:
                    raise ValueError("Embedding visualization not supported")
                fig, _ = umap_from_embedding(
                    wvf_rep.to("cpu").detach().numpy(),
                    acg_rep.to("cpu").detach().numpy(),
                    labels,
                    n_class=len(np.unique(labels)),
                    standardize=True,
                    dot_size=dot_size,
                )
                umap_img_path = os.path.join(
                    output_dir, f"embedding_visualization_epoch_{epoch}.png"
                )
                fig.savefig(umap_img_path)
                im = plt.imread(umap_img_path)
                if args.use_wandb:
                    currlog["embedding umap"] = [wandb.Image(im)]
                plt.close(fig)
        if args.use_wandb:
            wb.log(currlog)
    if args.use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":

    train()
