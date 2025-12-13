#!/usr/bin/env python
# coding: utf-8

import os
import csv
import ast
import copy
import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EvalPrediction
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src.Phase1Dataset import PlaylistDataset, PlaylistDataCollator
from src.SongBertPhase1 import SongBertModelPhase1

import wandb


# ============================================================
# Command-Line Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase-1 SongBERT")

    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--train_file", type=str, default="data/playlists.tsv")
    parser.add_argument("--lyrics_file", type=str, default="data/final_data.tsv")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_context_songs", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    parser.add_argument("--output_dir", type=str, default="./song_bert_output")
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()
    return args


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred, **kwargs):

    # HuggingFace returns predictions as a tuple, so extract logits
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert numpy -> torch
    logits = torch.tensor(logits)
    probs = torch.sigmoid(logits)

    # Convert back to numpy for sklearn
    probs_np = probs.numpy()
    labels_np = eval_pred.label_ids

    preds_np = (probs_np > 0.5).astype(int)

    # Flatten because task is per-context binary classification
    preds_flat = preds_np.flatten()
    labels_flat = labels_np.flatten()

    accuracy = accuracy_score(labels_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average='binary'
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }



# ============================================================
# Build adjacency list from playlist TSV
# ============================================================

def build_adjacency_list(path):
    adj = defaultdict(set)

    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            raw = row[1]
            if raw != "[]":
                ids = [int(x) for x in raw.replace("'", "").strip("[]").split(",")]
            else:
                ids = []

            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        adj[ids[i]].add(ids[j])

    return adj


# ============================================================
# Tokenized Lyrics Loader
# ============================================================

def get_tokenized_lyrics(path, tokenizer, max_length):
    total = sum(1 for _ in open(path, encoding="latin1", errors="ignore"))
    songid_to_tokenized = {}

    with open(path, encoding="latin1", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)

        for row in tqdm(reader, total=total):
            try:
                song_id = int(row[0])
                lyrics = row[3]

                tokenized = tokenizer(
                    lyrics,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )

                songid_to_tokenized[song_id] = tokenized

            except:
                pass

    return songid_to_tokenized


# ============================================================
# Main Training Logic
# ============================================================

def main():
    args = parse_args()

    # Detect device (CPU / CUDA / MPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}")

    # ----------------------------
    # Load tokenizer + base model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name).to(device)

    # ----------------------------
    # Build adjacency list
    # ----------------------------
    print("\nBuilding adjacency list...")
    adj = build_adjacency_list(args.train_file)

    # ----------------------------
    # Load or tokenize lyrics
    # ----------------------------
    tok_cache_path = "tokenized_lyrics.pt"

    try:
        songid_to_tokenized = torch.load(tok_cache_path, weights_only=False)
        print("Loaded cached tokenized lyrics.")
    except:
        print("Tokenizing lyrics...")
        songid_to_tokenized = get_tokenized_lyrics(args.lyrics_file, tokenizer, args.max_length)
        torch.save(songid_to_tokenized, tok_cache_path)
        print("Saved cache to tokenized_lyrics.pt.")

    # Fallback for missing lyrics
    unk_tokens = tokenizer(
        "[UNK]", truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )
    songid_to_tokenized = defaultdict(lambda: copy.deepcopy(unk_tokens), songid_to_tokenized)

    # ----------------------------
    # Train/Val Split
    # ----------------------------
    all_songs = list(adj.keys())
    split_idx = int(0.8 * len(all_songs))

    train_songs = all_songs[:split_idx]
    val_songs = all_songs[split_idx:]

    train_adj = {s: adj[s] for s in train_songs}
    val_adj = {s: adj[s] for s in val_songs}

    print("\nCreating datasets...")
    train_dataset = PlaylistDataset(train_adj, songid_to_tokenized, args.max_length, args.num_context_songs)
    val_dataset = PlaylistDataset(val_adj, songid_to_tokenized, args.max_length, args.num_context_songs)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    # ----------------------------
    # Initialize SongBERT Phase 1
    # ----------------------------
    print("\nInitializing model...")
    model = SongBertModelPhase1(base_model, loss_fn=torch.nn.BCEWithLogitsLoss()).to(device)

    # ----------------------------
    # Training Arguments
    # ----------------------------

    wandb.login(key="b52c89333beb85a2b1137e25b011353bac754299")
    
    wandb_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_context_songs":  args.num_context_songs
    }
    wandb.init(project="songbert", name="phase1-training", config=wandb_config, resume=args.resume)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=16,

        learning_rate=args.learning_rate,
        weight_decay=0.01,

        bf16=torch.cuda.is_available(),     # A100 fast path
        bf16_full_eval=torch.cuda.is_available(),
        fp16=False,                         # fp16 optional
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=1,           
        remove_unused_columns=False,
        report_to=["wandb"], 
        disable_tqdm=False,
        torch_compile=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PlaylistDataCollator(),
        compute_metrics=compute_metrics
    )

    # ----------------------------
    # Training
    # ----------------------------
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ----------------------------
    # Save Final Model
    # ----------------------------
    final_path = os.path.join(args.output_dir, "final_model")
    print(f"\nSaving final model → {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
