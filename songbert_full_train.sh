#!/bin/bash


#SBATCH --job-name=songbert_full
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

echo "training songbert full..."
date

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full SFT training
module load python
module load cuda


python train_songbert_phase3.py \
    --model_name answerdotai/ModernBERT-base \
    --train_file data/playlists.tsv \
    --lyrics_file data/final_data.tsv \
    --batch_size 4 \
    --epochs 8 \
    --max_length 256 \
    --name_length 5 \
    --num_context_songs 5 \
    --num_samples 3 \
    --learning_rate 3e-5 \
    --output_dir trained_models/songbert_p3_comparison 
    # --p1_dir trained_models/songbert_p1_no_scorer/final_model

echo "training script finished."
