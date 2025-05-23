import torch
import numpy as np
from celltype_ibl.utils.ibl_data_util import encode_ibl_training_data
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump
import os
import csv
from celltype_ibl.params.config import (
    CLIP_MODEL_DIR,
    VAE_DIR,
)


def linear_label_ratio_sweep(
    save_path: str,
    embedding_model: str = "contrastive",
    label_ratios: list[float] = [0.5],
    seed: int = 42,
    c: float = 1.0,
) -> dict:
    """
    Train the model with different label ratios
    """
    simclr_seed = [42, 26, 29, 65, 70]
    use_raw = False
    if embedding_model == "supervise":
        embedding_model = "contrastive"
        use_raw = True
    if embedding_model == "contrastive":
        model_path = CLIP_MODEL_DIR + f"/seed_{seed + 42}_checkpoint.pt"

    vae_acg_path = f"{VAE_DIR}/3DACG_logscale_seed_{seed + 1234}_encoder_gelu.pt"
    vae_wvf_path = f"{VAE_DIR}/wvf_singlechannel_seed_{seed + 1234}_encoder.pt"

    X1, X2, Y = encode_ibl_training_data(
        model_path=model_path,
        vae_acg_path=vae_acg_path,
        vae_wvf_path=vae_wvf_path,
        test_fold=[3, 4, 6],
        model=embedding_model,
        use_raw=use_raw,
        seed=simclr_seed[seed],
    )
    X1_test, X2_test, Y_test = encode_ibl_training_data(
        model_path=model_path,
        vae_acg_path=vae_acg_path,
        vae_wvf_path=vae_wvf_path,
        test_fold=[0, 1, 2, 4, 5, 7, 8, 9],
        model=embedding_model,
        use_raw=use_raw,
        seed=simclr_seed[seed],
    )
    X = np.concatenate([X1, X2], axis=1)
    X_test = np.concatenate([X1_test, X2_test], axis=1)
    train_labels = Y
    test_labels = Y_test

    test_accs = []
    test_f1s = []
    for label_ratio in label_ratios:
        print(f"Training with label ratio {label_ratio}")
        np.random.seed(42 + i)
        torch.cuda.manual_seed(42 + i)
        torch.random.manual_seed(42 + i)
        # Initialize the model
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, tol=1e-5, class_weight="balanced", C=c),
        )
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

        pred_labels = clf.fit(X[train_idx], train_labels[train_idx]).predict(X_test)

        dump(
            clf,
            f"{save_path}/{embedding_model}_linear_label_ratio_{label_ratio}.joblib",
        )

        test_acc = balanced_accuracy_score(test_labels, pred_labels)
        test_f1 = f1_score(test_labels, pred_labels, average="macro")
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    return {
        "label_ratio": label_ratios,
        "test_accs": test_accs,
        "test_f1s": test_f1s,
    }


if __name__ == "__main__":
    save_dir = (
        "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_label_ratio_sweep/vae_fix"
    )
    simclr_seed = [42]  # [42, 26, 29, 65, 70]
    for embedding_model in [
        "vae"
    ]:  # ["simclr"]:  # ["supervise", "contrastive", "vae"]:
        for i in range(5):
            seed = 42
            if embedding_model == "simclr":
                seed = simclr_seed[i]
            elif embedding_model == "vae":
                seed = 42 + i

            save_path = f"{save_dir}/" + f"{embedding_model}_seed_{seed}_linear"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if embedding_model == "contrastive":
                c = 0.02
            elif embedding_model == "vae":
                c = 0.4  # 0.001
            elif embedding_model == "supervise":
                c = 0.001
            elif embedding_model == "simclr":
                c = 2.5
            elif embedding_model == "FOCAL":
                c = 0.8

            sweep_results = linear_label_ratio_sweep(
                save_path=save_path,
                embedding_model=embedding_model,
                label_ratios=[0.01, 0.1, 0.3, 0.5, 0.8, 1.0],
                seed=i,
                c=c,
            )

            # Specify the filename
            filename = (
                save_dir
                + f"/label_ratio_sweep_{embedding_model}_seed_{seed}_linear.csv"
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
