#!/bin/bash
#SBATCH --job-name=train_ablate
#SBATCH --array=0-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00

# Set ABLATE_IDX based on SLURM_ARRAY_TASK_ID
export ABLATE_IDX=$SLURM_ARRAY_TASK_ID

# Run the training
python train.py