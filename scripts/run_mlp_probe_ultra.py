import numpy as np
import argparse
from celltype_ibl.models import mlp_classifier as mlp_probe
from celltype_ibl.utils.ultra_data_util import get_ultra_wvf_acg_pairs
from celltype_ibl.params.config import ULTRA_DATA_PATH
import os
import re


def extract_seed_and_epoch(checkpoint_path):
    # Use regular expressions to extract seed and epoch numbers
    seed_match = re.search(r"_seed(\d+)_", checkpoint_path)
    epoch_match = re.search(r"checkpoint_epoch_(\d+)\.pt", checkpoint_path)

    if seed_match and epoch_match:
        seed = seed_match.group(1)
        epoch = epoch_match.group(1)
        return seed, epoch
    else:
        raise ValueError("Cannot extract seed and epoch from the checkpoint path.")


def run_experiment(checkpoint_path):
    # Load data
    wvf, acg, labels, _ = get_ultra_wvf_acg_pairs()

    # Preprocess data
    acg = acg * 10
    unique_labels, label_idx = np.unique(labels, return_inverse=True)
    CORRESPONDENCE = {i: l for i, l in enumerate(unique_labels)}
    LABELLING = {l: i for i, l in enumerate(unique_labels)}

    # Prepare dataset
    test_dataset = np.concatenate((acg.reshape(-1, 1010), wvf), axis=1).astype(
        "float32"
    )

    # Extract seed and epoch from checkpoint path
    seed, epoch = extract_seed_and_epoch(checkpoint_path)

    # Create a unique save directory
    base_save_dir = ULTRA_DATA_PATH + "/Ultra_results"
    save_dir = os.path.join(base_save_dir, f"seed_{seed}_epoch_{epoch}_freeze")
    os.makedirs(save_dir, exist_ok=True)

    # Run the model
    mlp_probe.main(
        test_dataset,
        label_idx,
        LABELLING,
        CORRESPONDENCE,
        output_folder=save_dir,
        embedding_model_path=checkpoint_path,
        loo=False,
        embedding_model="contrastive",
        use_final_embed=False,
        latent_dim=512,
        modality="both",
        adjust_to_ultra=True,
        freeze_encoder_weights=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLP probe with a specific checkpoint path."
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the checkpoint file."
    )
    args = parser.parse_args()

    run_experiment(args.checkpoint_path)
