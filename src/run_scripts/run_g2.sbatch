#!/bin/bash
#SBATCH --gpus=2
#SBATCH -t 72:00:00
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user sbroome@kth.se
#SBATCH --output /proj/berzelius_pilot/users/x_sofbr/temporal-shape-dataset/logs/%J_slurm.out
#SBATCH --error  /proj/berzelius_pilot/users/x_sofbr/temporal-shape-dataset/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius_pilot/x_sofbr/xdata
cd /proj/berzelius_pilot/users/x_sofbr/temporal-shape-dataset/src/
python main.py --config ${CONFIG_FILE} --job_identifier ${SLURM_JOB_ID} --gpus=2 --accelerator=ddp --precision 16 --log_every_n_steps 5
