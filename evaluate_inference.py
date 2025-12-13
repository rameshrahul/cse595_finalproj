#!/usr/bin/env python
# coding: utf-8

"""
Modular evaluation framework for playlist generation models.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from transformers import DataCollatorWithPadding

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from src.RandomBaselineInference import RandomBaseline
from src.BertInference import BertInferencer
from src.SongBertPhase3 import SongBertModelPhase3
from src.SongBertPhase1 import SongBertModelPhase1


# =====================================================================
# Data Loading
# =====================================================================

def load_song_data(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip",
    )

    # Ensure required columns exist
    df = df[["song", "lyrics"]]

    # Drop rows with missing lyrics or song names
    df = df.dropna(subset=["song", "lyrics"])

    # Force string type (VERY IMPORTANT)
    df["song"] = df["song"].astype(str)
    df["lyrics"] = df["lyrics"].astype(str)

    return df.reset_index(drop=True)


def load_playlists(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["playlists"]

def tokenize_lyrics(lyrics_list, tokenizer, max_length, batch_size=32):
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    all_batches = []

    for i in tqdm(range(0, len(lyrics_list), batch_size), desc="Tokenizing lyrics (once)"):
        batch = lyrics_list[i:i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        batch_inputs = collator(
            [dict(zip(encoded, t)) for t in zip(*encoded.values())]
        )
        all_batches.append(batch_inputs)

    return all_batches

# =====================================================================
# Metrics
# =====================================================================

def precision_at_k(generated: List[str], actual: List[str]) -> float:
    generated = {s.lower().strip() for s in generated}
    actual = {s.lower().strip() for s in actual}
    hits = sum(1 for g in generated if g in actual)
    return hits / max(len(generated), 1)

def score_playlist_jaccard(generated, actual):
    g = {s.lower().strip() for s in generated}
    a = {s.lower().strip() for s in actual}
    if not g and not a:
        return 1.0
    return len(g & a) / len(g | a)

METRIC_REGISTRY: Dict[str, Callable[[List[str], List[str]], float]] = {
    "precision@k": precision_at_k,
    "jaccard": score_playlist_jaccard
}

# =====================================================================
# Model loading
# =====================================================================


def load_random_model(song_list, **kwargs):
    return RandomBaseline(song_list)


def load_bert_baseline(song_list, tokenized_lyrics, tokenizer, max_length, **kwargs):
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    model.eval()
    return BertInferencer(song_list, tokenized_lyrics, model, tokenizer, max_length)


def load_songbert(song_list, tokenized_lyrics, tokenizer, max_length, model_dir, model_cls, **kwargs):
    if model_dir is None:
        raise ValueError("model_dir must be provided for SongBERT model", model_cls)

    #initialize model
    model = model_cls(
        bert_model=AutoModel.from_pretrained("answerdotai/ModernBERT-base"),
        loss_fn=torch.nn.BCEWithLogitsLoss()
    )

    state_path = os.path.join(model_dir, "model.safetensors")
    state_dict = load_file(state_path)
    model.load_state_dict(state_dict)
    model.eval()

    return BertInferencer(song_list, tokenized_lyrics, model.bert, tokenizer, max_length)


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(
    playlists,
    models: Dict[str, object],
    metrics: Dict[str, Callable],
    sample_limit: int | None = None,
):
    results = {model_name: {m: [] for m in metrics} for model_name in models}

    iterator = playlists if sample_limit is None else playlists[:sample_limit]

    for playlist in tqdm(iterator, desc="Evaluating playlists"):
        tracks = [t["track_name"] for t in playlist["tracks"]]
        k = len(tracks)
        playlist_name = playlist["name"]

        for model_name, model in models.items():
            if model_name == "random":
                generated = model.generate_playlist(k)
            else:
                generated = model.generate_playlist(k, playlist_name)

            for metric_name, metric_fn in metrics.items():
                score = metric_fn(generated, tracks)
                results[model_name][metric_name].append(score)

    return results


# =====================================================================
# Visualization
# =====================================================================

def plot_results(results, output_path: str):
    model_names = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())

    num_models = len(model_names)
    num_metrics = len(metric_names)

    # Compute mean scores
    means = {
        metric: [
            sum(results[model][metric]) / len(results[model][metric])
            for model in model_names
        ]
        for metric in metric_names
    }

    x = range(num_models)
    width = 0.8 / num_metrics  # keep bars nicely spaced

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metric_names):
        offsets = [xi + i * width for xi in x]
        plt.bar(
            offsets,
            means[metric],
            width=width,
            label=metric
        )

    plt.xticks(
        [xi + width * (num_metrics - 1) / 2 for xi in x],
        model_names,
        rotation=15
    )

    plt.ylabel("Score")
    plt.title("Playlist Generation Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot -> {output_path}")



# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate playlist generation models")

    parser.add_argument("--song-tsv", default="data/chunk_data.tsv")
    parser.add_argument("--playlist-json", default="data/mpd.slice.2000-2999.json")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--songbert-p1-dir", default="trained_models/songbert_p1_10_epochs/final_model_phase1")
    parser.add_argument("--songbert-p3-dir", default="trained_models/songbert_p3/final_model")
    parser.add_argument("--plot-out", default="baseline_performance.png")

    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    print("Loading song data...")
    df = load_song_data(args.song_tsv)
    song_list = df["song"].tolist()
    lyrics_list = df["lyrics"].tolist()

    print("Loading playlists...")
    playlists = load_playlists(args.playlist_json)


    print("Tokenizing lyrics...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    tokenized_lyrics = tokenize_lyrics(
        lyrics_list,
        tokenizer,
        args.max_length,
        batch_size=32
    )

    print("Loading models...")
    models = {}
    models["random"] = load_random_model(song_list)
    models["baseline bert"] = load_bert_baseline(song_list, tokenized_lyrics, tokenizer, args.max_length)
    models["songbert_p1"] = load_songbert(song_list, tokenized_lyrics, tokenizer, args.max_length, args.songbert_p1_dir, SongBertModelPhase1)
    models["songbert_p3"] = load_songbert(song_list, tokenized_lyrics, tokenizer, args.max_length, args.songbert_p3_dir, SongBertModelPhase3)


    metrics = {m: METRIC_REGISTRY[m] for m in METRIC_REGISTRY}

    print("Evaluating...")
    results = evaluate(
        playlists,
        models=models,
        metrics=metrics,
        sample_limit=args.sample_limit,
    )

    print("\n=== AVERAGE SCORES ===")
    for model_name, metric_dict in results.items():
        for metric_name, values in metric_dict.items():
            avg = sum(values) / len(values)
            print(f"{model_name:15s} {metric_name:12s}: {avg:.4f}")

    plot_results(results, args.plot_out)


if __name__ == "__main__":
    main()
