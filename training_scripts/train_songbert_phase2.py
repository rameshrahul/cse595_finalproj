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

from src.Phase2Dataset import PlaylistDataset2, PlaylistData2Collator
from src.SongBertPhase2 import SongBertModelPhase2
from src.SongBertPhase1 import SongBertModelPhase1
from safetensors.torch import load_file
import wandb



def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase-3 SongBERT")

    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--train_file", type=str, default="data/playlists.tsv")
    parser.add_argument("--lyrics_file", type=str, default="data/final_data.tsv")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--name_length", type=int, default=10)
    parser.add_argument("--num_context_songs", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    parser.add_argument("--output_dir", type=str, default="./song_bert_output")
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--p1_dir", type=str, default = "")

    args = parser.parse_args()
    return args



def compute_metrics(eval_pred):

    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.tensor(logits)
    probs = torch.sigmoid(logits)

    probs_np = probs.numpy()
    labels_np = eval_pred.label_ids

    preds_np = (probs_np > 0.5).astype(int)

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




def read_playlist_tsv(path):
    # playlist data is [(playlist_name, [song_ids])]
    playlists = []

    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            name = row[0]
            raw = row[1]
            if raw != "[]":
                ids = [int(x) for x in raw.replace("'", "").strip("[]").split(",")]
                playlists.append((name, ids))

    return playlists



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



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.p1_dir == "":
        base_model = AutoModel.from_pretrained(args.model_name).to(device)
    else:
        #initialize model
        base_model = SongBertModelPhase1(
            bert_model=AutoModel.from_pretrained("answerdotai/ModernBERT-base"),
            loss_fn=torch.nn.BCEWithLogitsLoss()
        )

        state_path = os.path.join(args.p1_dir, "model.safetensors")
        state_dict = load_file(state_path)
        base_model.load_state_dict(state_dict)

        base_model = base_model.bert


    print("Reading in playlist data...")
    playlists = read_playlist_tsv(args.train_file)

    tok_cache_path = "tokenized_lyrics.pt"

    try:
        songid_to_tokenized = torch.load(tok_cache_path, weights_only=False)
        print("Loaded cached tokenized lyrics.")
    except:
        print("Tokenizing lyrics...")
        songid_to_tokenized = get_tokenized_lyrics(args.lyrics_file, tokenizer, args.max_length)
        torch.save(songid_to_tokenized, tok_cache_path)
        print("Saved cache to tokenized_lyrics.pt.")

    total_songs = max(songid_to_tokenized.keys())

    # Fallback for missing lyrics
    unk_tokens = tokenizer(
        "[UNK]", truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )
    songid_to_tokenized = defaultdict(lambda: copy.deepcopy(unk_tokens), songid_to_tokenized)

    split_idx = int(0.8 * len(playlists))

    train_list = playlists[:split_idx]
    val_list = playlists[split_idx:]

    print("\nCreating datasets...")
    train_dataset = PlaylistDataset2(train_list, total_songs, tokenizer, songid_to_tokenized, args.num_context_songs, args.num_samples, args.name_length)
    val_dataset = PlaylistDataset2(val_list, total_songs, tokenizer, songid_to_tokenized, args.num_context_songs, args.num_samples, args.name_length)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")


    print("\nInitializing model...")
    model = SongBertModelPhase2(base_model, loss_fn=torch.nn.BCEWithLogitsLoss()).to(device)

    wandb.login(key="") # Vrinda's key
    
    wandb_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_context_songs":  args.num_context_songs
    }
    wandb.init(project="songbert", name="phase2-training", config=wandb_config, resume=args.resume)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8, # try 8 in the next round, was 16 originally

        learning_rate=args.learning_rate,
        weight_decay=0.01,

        bf16=torch.cuda.is_available(),     
        bf16_full_eval=torch.cuda.is_available(),
        fp16=False,                       
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=1,           
        remove_unused_columns=False,
        report_to=["wandb"], 
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PlaylistData2Collator(),
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    final_path = os.path.join(args.output_dir, "final_model")
    print(f"\nSaving final model → {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
