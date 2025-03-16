# bimodal neuron embedding
This is the codebase for bimodal neuron embedding

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
```console
conda create --name nemo python=3.10
```

and activate it:
```console
conda activate nemo
```

install pytorch
```python
# installation for pytorch on linux with cuda=11.8. Go to pytorch.org for other options.
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

install [npyx](https://github.com/m-beau/NeuroPyxels/tree/master)
```console
pip install laplace-torch
pip install npyx
```

install [wandb](https://pypi.org/project/wandb/)
```console
pip install wandb
```
:warning: Please change your default local wandb saving path by running the following command (Linux, macOS):
```console
echo 'export WANDB_DIR="/path/to/your/new/dir"' >> ~/.bashrc
# Reload the configuration
source ~/.bashrc
```
:warning: Alternatively, change the directory for ```wandb.init```


Download NEMO from github and then install its dependencies and the package:
```console
git clone https://github.com/Haansololfp/celltype_ibl.git
cd celltype_ibl
pip install -e .
```

## Example: Training NEMO on the [C4 dataset](https://www.c4-database.com/apps/about) Step-by-Step

1. Initialize the dataset by following the instructions in [this notebook](https://github.com/m-beau/NeuroPyxels/blob/master/npyx/c4/replicate_c4_results.ipynb).
2. Set the `DATASETS_DIRECTORY` variable in `src/celltype_ibl/params/config.py` to the directory where you stored the dataset.
3. Navigate to the `scripts` directory:  
   ```bash
   cd scripts
   ```
4. run the training script:
    ```bash
    source train_c4_nemo.sh
    ```

