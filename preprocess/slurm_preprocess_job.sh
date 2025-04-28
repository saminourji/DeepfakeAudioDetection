#!/bin/bash
#SBATCH --job-name=mfcc_processing
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --partition=standard
#SBATCH --array=0-71236    
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

module load python/3.9
source /users/snourji/final-project-1470-ML/DeepfakeAudioDetection/env/bin/activate

FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" flac_files.txt)
python preprocess_single_file.py "$FILE"