# temporal-shape-dataset
A simpler action template dataset, to evaluate temporal modeling abilities and cross-domain robustness.

## Setting up

Set up a conda environment in the following way.

`conda create -n myenv python=3.8 scipy=1.5.2`

`conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge opencv`

`pip install torchsummary`

`conda install -c conda-forge scikit-learn`

`conda install av -c conda-forge`

`conda install -c conda-forge ipdb`

`conda install -c conda-forge prettytable`

`conda install pytorch-lightning -c conda-forge`

`conda install -c anaconda pandas`

`conda install -c conda-forge tqdm`

`pip install perlin-noise`


You also will want a wandb-account to keep track of your experiments.

`pip install wandb`

### Generate dataset
`cd src/dataset/; python generate_classification_dataset.py --num-sequences 10 --object-mode dot --symbol-size 2 --textured-background 0`

### Test run to train on existing data
`cd src/; python main.py --config configs/convlstm.json --job_identifier test --fast_dev_run=True --log_every_n_steps=5 --gpus=1`

or, if running on a Slurm cluster, use the provided `.sbatch`-file under `run_scripts`.
