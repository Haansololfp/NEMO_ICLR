import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import os
from celltype_ibl.utils.c4_vae_util import VAEEncoder, vae_encode_model, load_acg_vae

# Assuming the following classes and functions are defined as in your training script
from celltype_ibl.utils.ibl_data_util import get_ibl_wvf_acg_pairs
from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.utils.c4_vae_util import vae_encode_model, load_acg_vae
from celltype_ibl.utils.ibl_data_util import IBL_acg_wvf_Dataset
from celltype_ibl.utils.ibl_eval_utils import ibl_representation_MLP_classifier
from celltype_ibl.params.config import WVF_ENCODER_ARGS_SINGLE
from npyx.c4.dataset_init import BIN_SIZE, WIN_SIZE
from npyx.c4.dl_utils import load_waveform_encoder


# Load the test data
# Assumed that the test data loading function and indices are properly set up


save_dir = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_label_ratio_sweep/vae_fix"
)

test_set = [3, 6]
ibl_wvf, ibl_acg, cosmos_region, fold_idx = get_ibl_wvf_acg_pairs(
    return_region="cosmos"
)

unique_labels, train_region_idx = np.unique(cosmos_region, return_inverse=True)
labelling = {l: i for i, l in enumerate(unique_labels)}

test_idx = [index for index, element in enumerate(fold_idx) if element in test_set]
ibl_wvf = ibl_wvf.astype("float32")
ibl_acg = ibl_acg.astype("float32")

test_region_index = [labelling[label] for label in cosmos_region[test_idx]]
test_dataset = IBL_acg_wvf_Dataset(
    ibl_wvf[test_idx], ibl_acg[test_idx], test_region_index
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
label_ratios = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0]
simclr_seed = [42, 26, 29, 65, 70]
for embedding_model in ["vae"]:  # ["simclr"]:  # "supervise",
    for i in range(5):
        for label_ratio in label_ratios:
            formatted_number = f"{label_ratio:.2f}"  # Example format: '1,234,567.89'
            # Replace commas and period with underscores
            formatted_number = formatted_number.replace(".", "_")
            if (
                (embedding_model == "supervise")
                | (embedding_model == "contrastive")
                | (embedding_model == "simclr")
            ):
                if embedding_model == "simclr":
                    seed = simclr_seed[i]
                    ln = True
                else:
                    seed = i + 42
                    ln = False
                encode_model = BimodalEmbeddingModel(
                    layer_norm=ln,
                    latent_dim=512,
                    l2_norm=True,
                    activation="gelu",
                )
            elif embedding_model == "vae":

                seed = i + 1234
                acg_vae = load_acg_vae(
                    None,
                    WIN_SIZE // 2,
                    BIN_SIZE,
                    initialise=False,
                    pool="avg",
                    activation="gelu",
                )
                acg_head = VAEEncoder(acg_vae.encoder, 10)  # maybe change this?

                wvf_vae = load_waveform_encoder(
                    WVF_ENCODER_ARGS_SINGLE,
                    None,
                    in_features=90,
                    initialise=False,
                )
                wvf_head = VAEEncoder(
                    wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"]
                )
                encode_model = vae_encode_model(wvf_head, acg_head)

            model = ibl_representation_MLP_classifier(
                encode_model=encode_model, embedding_model=embedding_model
            )

            num = 2
            if (
                (embedding_model == "contrastive")
                | (embedding_model == "simclr")
                | (embedding_model == "vae")
            ):
                num = 3

            model_path = (
                f"{save_dir}/"
                + f"{embedding_model}{num}_seed_{seed}/best_model_{formatted_number}.pt"
            )

            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path)["model_state_dict"])
            model.to(device)
            model.eval()

            # Make predictions
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    output = model(data)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.numpy())

            # Generate classification report
            report = classification_report(
                all_labels, all_preds, digits=4
            )  # Add more class names as needed

            output_dir = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_label_ratio_classification_report"
            # Save the classification report
            # report_filename = (
            #     f"{output_dir}/"
            #     + f"{embedding_model}_freeze_seed_{seed}_classification_report_{formatted_number}.txt"
            # )
            report_filename = (
                f"{output_dir}/"
                + f"{embedding_model}_seed_{seed}_classification_report_{formatted_number}.txt"
            )

            with open(report_filename, "w") as f:
                f.write(report)
            print(f"Classification report saved to {report_filename}")

            cm = confusion_matrix(all_labels, all_preds)
            output_dir = (
                "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_single_model_cm"
            )
            # Save the confusion matrix
            # cm_filename = (
            #     f"{output_dir}/"
            #     + f"{embedding_model}{num}_freeze_seed_{seed}_cm_{formatted_number}.npy"
            # )
            cm_filename = (
                f"{output_dir}/"
                + f"{embedding_model}{num}_seed_{seed}_cm_{formatted_number}.npy"
            )
            np.save(cm_filename, cm)

            print(f"Classification report saved to {cm_filename}")
