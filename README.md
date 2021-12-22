# temporal-shape-dataset
This repository contains code for the Temporal Shape dataset, presented in "Recur, Attend or Convolve? Frame Dependency Modeling Matters for Cross-Domain Robustness in Action Recognition" by Broomé et al., arXiv 2021, with the purpose to evaluate principal temporal modeling abilities and cross-domain robustness in a light-weight manner.


### Gifs corresponding to Fig. 1 in the article

|       2Dot                |    5Dot  |    MNIST    |    MNIST-bg   | 
| --------------------- |:---------:|:---------:|:---------:|
| ![](assets/2dot_1240.gif)  | ![](assets/5dot_1700.gif) | ![](assets/mnist_140.gif)  | ![](assets/mnist_1500.gif) | 



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
