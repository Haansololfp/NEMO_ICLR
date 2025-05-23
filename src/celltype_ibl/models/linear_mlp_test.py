import celltype_ibl.models.linear_classifier as linear_probe
import celltype_ibl.models.mlp_classifier as mlp_probe
# from celltype_ibl.models.ACG_augmentation_dataloader import get_c4_labeled_dataset
from celltype_ibl.utils.ibl_data_util import get_ibl_wvf_acg_pairs
from celltype_ibl.utils.cell_explorer_data_util import get_allen_labeled_dataset
from celltype_ibl.utils.kenji_allen_data_utils import get_kenji_allen_wvf_acg_pairs
from celltype_ibl.utils.ultra_data_util import get_ultra_wvf_acg_pairs

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="embedding classification performance")
parser.add_argument("--model", type=str, default="mlp", help="model to use")
parser.add_argument("--dataset", type=str, default="c4", help="dataset to use")
parser.add_argument(
    "--embedding_model_path", type=str, default=None, help="embedding model path"
)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--acg_vae_path", type=str, default=None, help="acg vae path")
parser.add_argument("--wvf_vae_path", type=str, default=None, help="wvf vae path")
parser.add_argument(
    "--use_final_embed", action="store_true", default=False, help="use final embed"
)
parser.add_argument("--latent_dim", type=int, default=10, help="latent dimension")
parser.add_argument(
    "--embedding_model", type=str, default="contrastive", help="embedding model to use"
)
parser.add_argument(
    "--activation", type=str, default="gelu", help="activation function"
)
parser.add_argument(
    "--layer_norm", action="store_true", default=False, help="use layer norm"
)
parser.add_argument("--l2_norm", action="store_true", default=False, help="use l2 norm")
parser.add_argument(
    "--freeze_encoder_weights",
    action="store_false",
    default=True,
    help="freeze encoder weights",
)
parser.add_argument("--loo", action="store_true", default=False, help="use loo")
parser.add_argument("--adjust_to_ultra", action="store_false", default=True, help="adjust to ultra")
parser.add_argument("--n_runs", type=int, default=10, help="number of runs")


def main():
    args = parser.parse_args()

    # if args.dataset == "c4":
    #     wvf, acg, label_idx, _, LABELLING, CORRESPONDENCE = get_c4_labeled_dataset(
    #         from_h5=False
    #     )
    if args.dataset == "ibl":
        wvf, acg, cosmos_region, _ = get_ibl_wvf_acg_pairs(return_region="cosmos")
        unique_labels, label_idx = np.unique(cosmos_region, return_inverse=True)
        acg = acg * 10
        CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
        LABELLING = {l: i for i, l in enumerate(unique_labels)}
    elif args.dataset == "ibl_repeated":
        wvf, acg, cosmos_region = get_ibl_wvf_acg_pairs(
            return_region="cosmos", repeated_sites=True
        )
        acg = acg * 10
        unique_labels, label_idx = np.unique(cosmos_region, return_inverse=True)
        CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
        LABELLING = {l: i for i, l in enumerate(unique_labels)}
    elif args.dataset == "allen":
        wvf, acg, labels = get_allen_labeled_dataset()
        acg = acg * 10
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
        LABELLING = {l: i for i, l in enumerate(unique_labels)}
    elif args.dataset == "kenji_allen":
        wvf, acg, labels, session_id = get_kenji_allen_wvf_acg_pairs(return_id = "session")
        acg = acg * 10
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
        LABELLING = {l: i for i, l in enumerate(unique_labels)}
    elif args.dataset == "Ultra":
        wvf, acg, labels = get_ultra_wvf_acg_pairs()
        acg = acg * 10
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
        LABELLING = {l: i for i, l in enumerate(unique_labels)}

    else:
        raise ValueError("Invalid dataset")

    test_dataset = np.concatenate((acg.reshape(-1, 1010), wvf), axis=1).astype(
        "float32"
    )

    assert args.model in ["mlp", "linear"]
    assert args.embedding_model in ["contrastive", "VAE"]

    if args.model == "linear":
        if not args.freeze_encoder_weights:
            raise ValueError("Linear probe does not support unfreezing encoder weights")
        linear_probe.main(
            test_dataset,
            label_idx,
            LABELLING,
            CORRESPONDENCE,
            output_folder=args.save_dir,
            acg_vae_path=args.acg_vae_path,
            wvf_vae_path=args.wvf_vae_path,
            embedding_model_path=args.embedding_model_path,
            loo=args.loo,
            n_runs=args.n_runs,
            embedding_model=args.embedding_model,
            use_final_embed=args.use_final_embed,
            latent_dim=args.latent_dim,
            activation=args.activation,
            adjust_to_ultra=args.adjust_to_ultra
        )
    else:
        mlp_probe.main(
            test_dataset,
            label_idx,
            LABELLING,
            CORRESPONDENCE,
            output_folder=args.save_dir,
            acg_vae_path=args.acg_vae_path,
            wvf_vae_path=args.wvf_vae_path,
            embedding_model_path=args.embedding_model_path,
            loo=args.loo,
            # n_runs=args.n_runs,
            embedding_model=args.embedding_model,
            use_final_embed=args.use_final_embed,
            layer_norm=False,
            latent_dim=args.latent_dim,
            l2_norm=True,
            activation=args.activation,
            freeze_encoder_weights=args.freeze_encoder_weights,
            adjust_to_ultra=args.adjust_to_ultra
        )


if __name__ == "__main__":
    main()
