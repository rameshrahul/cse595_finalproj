#!/bin/bash


#SBATCH --job-name=eval_inference
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END

# Script to run SFT training on the full dataset.
# Designed for larger-scale training on clusters.

echo "evaluating songbert..."

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full SFT training

module load python
module load cuda

python evaluate_inference.py

echo "evaluation script finished."
