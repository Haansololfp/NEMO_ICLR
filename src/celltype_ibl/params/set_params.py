import os
import argparse


def none_or_float(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")


def parse_base_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Bimodal Embedding Model")
    parser.add_argument(
        "-m", "--model", type=str, default="contrastive", help="model to use"
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.5)
    # whether to train temperature or not
    parser.add_argument("--train_temperature", action="store_true", default=False)
    parser.add_argument("-l", "--layer_norm", action="store_true", default=False)
    parser.add_argument(
        "-d", "--dim_embed", type=int, default=512
    )  # embedding dimension
    parser.add_argument(
        "-s", "--similarity_adjust", action="store_true", default=False
    )  # whether to adjust the target by the similarity
    parser.add_argument(
        "-a", "--augmentation", action="store_false", default=True
    )  # whether to use augmentation
    parser.add_argument("--std", type=float, default=0.1)  # noise std for augmentation
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=1500
    )  # number of epochs to train
    parser.add_argument("-b", "--batch_size", type=int, default=1024)
    parser.add_argument(
        "-k", "--top_k", type=int, action="append", default=[]
    )  # use like --top_k 1 --top_k 5 --top_k 10 # check top k accuracy for the training and validation set during training
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--save", action="store_false", default=True
    )  # whether to save the model
    parser.add_argument(
        "--use_wandb", action="store_false", default=True
    )  # whether to use wandb
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
    )  # where to save the model
    parser.add_argument(
        "--linear_probe", action="store_false"
    )  # whether to validate during traing with a linear probe
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--n_runs", type=int, default=1
    )  # number of runs for cross validation
    parser.add_argument(
        "--log_every_n_steps", default=100, type=int, help="Log every n steps"
    )  # log every n steps
    parser.add_argument(
        "--activation", type=str, default="gelu"
    )  # activation function to use
    parser.add_argument(
        "--split_validate", action="store_false"
    )  # Default is True if not specified, whether to split the data into train and validation
    parser.add_argument(
        "--load_pt_path", type=str, default=None
    )  # load a previously trained model for continued training

    parser.add_argument(
        "--initialise", action="store_false", default=True
    )  # initialise weights with gaussian noise
    parser.add_argument(
        "--dataset", type=str, default="ibl"
    )  # which dataset to use for training
    parser.add_argument(
        "--project", type=str, default="both"
    )  # the project to usefor the kenji allen dataset
    parser.add_argument(
        "--with_root", action="store_true"
    )  # whether to include root in the ibl dataset
    parser.add_argument(
        "--from_h5", action="store_true"
    )  # whether to load c4 data from h5 file
    parser.add_argument(
        "--held_out", action="store_false", default=True
    )  # for ibl data, we train the model on part of the data and test on a held-out subset
    parser.add_argument(
        "--test_data", type=str, default="ibl"
    )  # which dataset to test on
    parser.add_argument(
        "--embedding_viz", action="store_false", default=True
    )  # whether to visualize the embeddings
    parser.add_argument(
        "--undersample", action="store_true", default=False
    )  # whether to undersample the majority class during validation
    parser.add_argument(
        "--use_smote", action="store_true", default=False
    )  # whether to use SMOTE for oversampling during validation

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--acg_normalization", action="store_true", default=False
    )  # whether to normalize acg
    parser.add_argument(
        "--batch_norm", action="store_true", default=False
    )  # whether to use batch norm
    parser.add_argument(
        "--adjust_to_ce", action="store_true"
    )  # whether to adjust the model to be compatible with cell-explorer, which has a different waveform length
    parser.add_argument("--adjust_to_ultra", action="store_true")
    # whether to adjust the model to be compatible with ultra, which has a different waveform length

    parser.add_argument(
        "--acg_dropout", type=none_or_float, default=None
    )  # dropout rate for acg, default is 0.2
    parser.add_argument(
        "--wvf_dropout", type=none_or_float, default=None
    )  # dropout rate for acg, default is 0.1
    parser.add_argument(
        "--by_session", action="store_true", default=False
    )  # whether to cross validate by session
    parser.add_argument("--loss_flood", type=none_or_float, default=None)
    parser.add_argument(
        "--use_RINCE_loss", action="store_true", default=False
    )  # whether to use RINCE loss https://arxiv.org/pdf/2201.11736
    parser.add_argument("--lam", type=float, default=0.01)  # lambda for RINCE loss
    parser.add_argument("-q", "--rince_q", type=float, default=0.1)  # q for RINCE loss

    parser.add_argument(
        "--split_id", type=int, default=1
    )  # split index for cell_explorer
    parser.add_argument(
        "-n", "--l2_norm", action="store_false"
    )  # Default is True if not specified

    args = parser.parse_args()
    return args
