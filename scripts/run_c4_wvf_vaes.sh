#!/bin/bash
#SBATCH --job-name=c4_wvf_vaes
#SBATCH --output=output_c4_wvf_vaes.txt
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=10G

module load cuda/12.3.2
source /mnt/home/hyu10/.bashrc
conda activate celltype_ibl

python celltype_ibl/scripts/wvf_VAE_training_seed_sweep.py