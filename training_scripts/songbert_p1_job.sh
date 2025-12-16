#!/bin/bash


#SBATCH --job-name=continue_songbert_p1_compile
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END


echo "training songbert..."

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

module load python
module load cuda


python train_songbert_phase1.py \
    --model_name answerdotai/ModernBERT-base \
    --train_file data/playlists.tsv \
    --lyrics_file data/final_data.tsv \
    --batch_size 16 \
    --epochs 8 \
    --max_length 256 \
    --num_context_songs 8 \
    --learning_rate 3e-5 \
    --output_dir trained_models/songbert_p1_no_scorer 

    
echo "training script finished."
