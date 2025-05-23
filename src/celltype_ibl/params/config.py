C4_COLORS = {
    "PkC_ss": [28, 120, 181],
    "PkC_cs": [0, 0, 0],
    "MLI": [224, 85, 159],
    "MFB": [214, 37, 41],
    "GrC": [143, 103, 169],
    "GoC": [56, 174, 62],
    "laser": [96, 201, 223],
    "drug": [239, 126, 34],
    "background": [244, 242, 241],
    "MLI_A": [224, 85, 150],
    "MLI_B": [220, 80, 160],
}

REGION_COLORS = {
    "CB": [240, 240, 128],
    "CNU": [152, 214, 249],
    "CTXsp": [138, 218, 135],
    "HB": [255, 155, 136],
    "HPF": [126, 208, 75],
    "HY": [230, 68, 56],
    "Isocortex": [112, 255, 113],
    "MB": [255, 100, 255],
    "OLF": [154, 210, 189],
    "TH": [255, 112, 128],
}

V1_COLORS = {
    "PV": [214, 37, 41],
    "SST": [143, 103, 169],
    "VIP": [56, 174, 62],
}


WVF_ENCODER_ARGS_SINGLE = {
    "beta": 5,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 1e-4,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}

# direcotries for ibl models
CLIP_MODEL_DIR = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_CLIP_picked_wo_amp_jitter"
)
VAE_DIR = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_vaes"
SIMCLR_ACG_DIR = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/SimCLR/ACG_chosen_by_acc"
)
SIMCLR_WVF_DIR = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/SimCLR/WVF_chosen_by_acc"
)

# wandb directotry
WANDB_DIR = "/mnt/sdceph/users/hyu10/cell-type_representation/wandb_save"
WANDB_PROJECT = "ibl"
WANDB_ENTITY = "contrastive_neuron"

# c4, ibl data directories
DATASETS_DIRECTORY = "/mnt/sdceph/users/hyu10/cell-type_representation/"


ALLEN_LABELLED_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/cell_explorer/Allen_cell_explorer_preprocessed.npz"
ALLEN_UNLABELLED_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/allen_data/wvf82norm_acg_nontuned (1).npz"

CELL_EXPLORER_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/cell_explorer/cell_explorer_wvf_acg_normalized_all.npz"

SPLIT_INDEX_PATH = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/cell_explorer/split_index"
)
INDEX_MAPPING_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/cell_explorer/valid_index_mapping_to_collected.csv"
KENJI_ALLEN_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/allen_visual_coding_visual_behavior/Allen_wvf_acg_pair_v1_no_opto_stim.pkl"


# ultra data directories
ULTRA_DATA_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/UHD_optotagged/Ultra_data/npultra_wvf_acg_pair.pkl"
STIM_REMOVED_DATA_PATH = (
    "/mnt/sdceph/users/hyu10/uhd_celltype/npultra_wvf_acg_pair_remove_stim_combined.pkl"
)
