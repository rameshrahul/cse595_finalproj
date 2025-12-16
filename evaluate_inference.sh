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

python evaluate_inference.py \
    --songbert-p1-dir trained_models/songbert_p1_no_scorer/final_model \
    --songbert-p2-dir trained_models/songbert_p3_comparison/final_model \
    --songbert-final-dir trained_models/songbert_full/final_model \
    --song-tsv data/chunk_data_4.tsv \
    --playlist-json data/mpd.slice.4000-4999.json

echo "evaluation script finished."
