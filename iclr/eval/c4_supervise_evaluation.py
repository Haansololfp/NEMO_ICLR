from celltype_ibl.utils.c4_data_utils import get_c4_labeled_dataset
import celltype_ibl.models.mlp_classifier as mlp_probe
import celltype_ibl.models.linear_classifier as linear_probe
import numpy as np

VAE_DIRECTORY = "/mnt/sdceph/users/hyu10/c4_results_new/VAE_checkpoints"
OUT_DIR = "/mnt/home/hyu10/ceph/c4_results_new/supervise_results"

waveforms, acgs, label_idx, label, labelling, correspondence = get_c4_labeled_dataset()
test_dataset = np.concatenate((acgs.reshape(-1, 1010) * 10, waveforms), axis=1).astype(
    "float32"
)

for seed in range(1234, 1234 + 5):
    acg_vae_path = VAE_DIRECTORY + f"/3DACG_logscale_encoder_gelu_seed_{seed}.pt"
    wvf_vae_path = VAE_DIRECTORY + f"/wvf_singlechannel_encoder_seed_{seed}.pt"

    mlp_probe.main(
        test_dataset,
        label_idx,
        labelling,
        correspondence,
        output_folder=OUT_DIR + f"/seed_{seed}",
        acg_vae_path=acg_vae_path,
        wvf_vae_path=wvf_vae_path,
        loo=False,
        embedding_model="VAE",
        use_final_embed=False,
        activation="gelu",
        freeze_encoder_weights=False,
        seed=seed,
        adjust_to_ultra=False,
        initialise=False,
    )
