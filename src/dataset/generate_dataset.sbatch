#!/bin/bash
#SBATCH -t 25:00:00
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user sbroome@kth.se
#SBATCH --output /proj/berzelius-2021-50/users/x_sofbr/temporal-shape-dataset/logs/%J_slurm.out
#SBATCH --error  /proj/berzelius-2021-50/users/x_sofbr/temporal-shape-dataset/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2021-50/users/x_sofbr/xdata
cd /proj/berzelius-2021-50/users/x_sofbr/temporal-shape-dataset/src/dataset/
python generate_classification_dataset.py --num-sequences ${NUM_SEQUENCES} --object-mode ${OBJECT_MODE} --symbol-size ${SYMBOL_SIZE} --train-test-mnist ${TRAIN_TEST_MNIST} --textured-background ${TEXTURED_BACKGROUND} --perlin-intensity ${PERLIN_INTENSITY}
