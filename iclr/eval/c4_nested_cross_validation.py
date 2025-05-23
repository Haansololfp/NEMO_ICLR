import numpy as np
from celltype_ibl.utils.c4_data_utils import get_c4_labeled_dataset
from celltype_ibl.models.bimodal_embedding_main import concatenate_dataset
from celltype_ibl.models.linear_classifier import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
import celltype_ibl.models.mlp_classifier_multi_folds as mlp_probe_multi_folds
import celltype_ibl.models.linear_classifier_multi_folds as linear_probe_multi_folds
from celltype_ibl.utils.nested_cv_util import StratifiedKFoldHandler
import argparse

model_directory = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/contrastive_experiment"
)

model_paths = [
    "tempF_dim512_augT_batch1024_actgelu_data_c4_seed42_2025-02-22-14-44-31_h5F_init",
    "tempF_dim512_augT_batch1024_actgelu_data_c4_seed43_2025-02-24-19-32-17_h5F_init",
    "tempF_dim512_augT_batch1024_actgelu_data_c4_seed44_2025-02-25-04-06-37_h5F_init",
    "tempF_dim512_augT_batch1024_actgelu_data_c4_seed45_2025-02-25-19-22-12_h5F_init",
    "tempF_dim512_augT_batch1024_actgelu_data_c4_seed46_2025-02-25-19-22-12_h5F_init",
]
# Set the path to the model
save_dir = "/mnt/home/hyu10/ceph/c4_results_new/nemo_results"


def main(args):
    waveforms, acgs, label_idx, label, labelling, correspondence = (
        get_c4_labeled_dataset()
    )
    input_data = concatenate_dataset(acgs, waveforms)
    for seed in [42, 43, 44, 45, 46]:
        model_path = model_directory + f"/{model_paths[seed-42]}"

        if args.recompute_f1:
            cv_f1 = []
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
            for i in range(0, 3001, 100):
                embedding_model_path = model_path + f"/checkpoint_epoch_{i}.pt"
                f1 = np.zeros((50,))
                j = 0
                for train_idx, val_idx in rskf.split(input_data, label_idx):
                    results_dict = cross_validate(
                        input_data[train_idx],
                        label_idx[train_idx],
                        embedding_model="contrastive",
                        embedding_model_path=embedding_model_path,
                        pool_type="avg",
                        loo=False,
                        n_runs=10,
                        save_folder=None,
                        device="cpu",
                        use_final_embed=False,
                        layer_norm=False,
                        latent_dim=512,
                        l2_norm=True,
                        batch_norm=False,
                        activation="gelu",
                        seed=seed,
                    )
                    f1[j] = np.mean(results_dict["f1_scores"])
                    j += 1
                cv_f1.append(f1[:, None])
            cv_f1 = np.concatenate(cv_f1, axis=1)
            np.save(save_dir + f"/cv_f1_seed_{seed}.npy", cv_f1)
        else:
            cv_f1 = np.load(save_dir + f"/cv_f1_seed_{seed}.npy")

        max_idx = np.argmax(cv_f1, axis=1)  # [5 fold x 10 repeats] x n epochs
        k_fold_handler = StratifiedKFoldHandler(
            n_splits=5, n_repeats=10, random_state=seed
        )
        epochs = np.arange(0, 3001, 100)
        embedding_model_paths = [
            model_path + f"/checkpoint_epoch_{epochs[max_idx[i]]}.pt" for i in range(50)
        ]
        num_calls = 10
        for j in range(num_calls):
            k_fold = k_fold_handler.get_folds_for_repeat(
                input_data, label_idx, repeat_id=j
            )
            sub_folder = save_dir + f"/seed_{seed}/run_{j}"

            mlp_probe_multi_folds.main(
                input_data,
                label_idx,
                labelling,
                correspondence,
                output_folder=sub_folder,
                embedding_model_paths=embedding_model_paths[j * 5 : (j + 1) * 5],
                loo=False,
                n_runs=1,
                latent_dim=512,
                seed=seed,
                freeze_encoder_weights=True,
                k_fold=k_fold,
                adjust_to_ultra=False,
            )

            mlp_probe_multi_folds.main(
                input_data,
                label_idx,
                labelling,
                correspondence,
                output_folder=sub_folder,
                embedding_model_paths=embedding_model_paths[j * 5 : (j + 1) * 5],
                loo=False,
                n_runs=1,
                latent_dim=512,
                seed=seed,
                freeze_encoder_weights=False,
                k_fold=k_fold,
                adjust_to_ultra=False,
            )

            linear_probe_multi_folds.main(
                input_data,
                label_idx,
                labelling,
                correspondence,
                output_folder=sub_folder,
                embedding_model_paths=embedding_model_paths[j * 5 : (j + 1) * 5],
                loo=False,
                n_runs=10,
                latent_dim=512,
                seed=seed,
                k_fold=k_fold,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for nested cross validation")
    parser.add_argument(
        "--recompute_f1", action="store_true", help="recompute f1 scores"
    )
    args = parser.parse_args()
    main(args)
