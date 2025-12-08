#!/usr/bin/env python
# coding: utf-8

"""
Evaluate three playlist-generation baselines:
1. Random baseline
2. Pretrained ModernBERT baseline
3. Phase-1 SongBERT (fine-tuned)
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# Local dependencies
from src.RandomBaselineInference import RandomBaseline
from src.BertInference import BertBaseline
from src.SongBertPhase1 import SongBertModelPhase1


# -------------------------------------------------------
# Data Loading
# -------------------------------------------------------

def load_song_data(tsv_dir="data/song_data/"):
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]
    dfs = [pd.read_csv(os.path.join(tsv_dir, f), sep="\t") for f in tsv_files]
    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates()


def load_playlists(path="data/playlist_data/mpd.slice.2000-2999.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["playlists"]


# -------------------------------------------------------
# Scoring Logic
# -------------------------------------------------------

def score_playlist(generated, actual):
    """Simple overlap score: precision@k."""
    generated = {s.lower().strip() for s in generated}
    actual = {s.lower().strip() for s in actual}
    hits = sum(1 for g in generated if g in actual)
    return hits / len(generated)


# -------------------------------------------------------
# Model Loading
# -------------------------------------------------------

def load_bert_baseline(song_list, lyrics_list):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    return BertBaseline(song_list, lyrics_list, tokenizer, model)


def load_songbert_phase1(song_list, lyrics_list, model_dir="song_bert_output/final_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_bert = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

    phase1_model = SongBertModelPhase1(
        bert_model=base_bert,
        loss_fn=torch.nn.BCEWithLogitsLoss()
    )
    state_dict = load_file(f"{model_dir}/model.safetensors")
    phase1_model.load_state_dict(state_dict)
    phase1_model.eval()

    print("Phase-1 SongBERT Loaded.")
    return BertBaseline(song_list, lyrics_list, tokenizer, phase1_model.bert)


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_models(playlists, random_model, bert_baseline, bert_phase1, sample_limit=None):
    random_scores = []
    base_scores = []
    phase1_scores = []

    iterator = playlists if sample_limit is None else playlists[:sample_limit]

    for playlist in tqdm(iterator, desc="Evaluating playlists"):
        tracks = [t["track_name"] for t in playlist["tracks"]]
        k = len(tracks)
        playlist_name = playlist["name"]

        random_scores.append(score_playlist(random_model.generate_playlist(k), tracks))
        base_scores.append(score_playlist(bert_baseline.generate_playlist(k, playlist_name), tracks))
        phase1_scores.append(score_playlist(bert_phase1.generate_playlist(k, playlist_name), tracks))

    return random_scores, base_scores, phase1_scores


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

def plot_results(random_scores, bert_base_scores, phase1_scores, output="baseline_performance.png"):
    categories = ["Random", "ModernBERT Baseline", "SongBERT Phase-1"]
    values = [
        100 * sum(random_scores) / len(random_scores),
        100 * sum(bert_base_scores) / len(bert_base_scores),
        100 * sum(phase1_scores) / len(phase1_scores),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=["blue", "green", "red"])
    plt.ylabel("Similarity %")
    plt.title("Performance of Playlist Generation Models")
    plt.savefig(output)
    print(f"Saved plot -> {output}")


# -------------------------------------------------------
# Main Script
# -------------------------------------------------------

def main():
    print("Loading song data...")
    df = load_song_data()
    song_list = df["song"].tolist()
    lyrics_list = df["lyrics"].tolist()

    print("Loading playlists...")
    playlists = load_playlists()

    print("Loading models...")
    random_model = RandomBaseline(song_list)
    bert_baseline = load_bert_baseline(song_list, lyrics_list)
    bert_phase1 = load_songbert_phase1(song_list, lyrics_list)

    print("Evaluating...")
    random_scores, base_scores, phase1_scores = evaluate_models(
        playlists,
        random_model,
        bert_baseline,
        bert_phase1,
        sample_limit=1000  # match original code
    )

    print("\n=== AVERAGE SCORES ===")
    print(f"Random:        {sum(random_scores)/1000:.4f}")
    print(f"Baseline BERT: {sum(base_scores)/1000:.4f}")
    print(f"SongBERT P1:   {sum(phase1_scores)/1000:.4f}")

    plot_results(random_scores, base_scores, phase1_scores)


if __name__ == "__main__":
    main()
