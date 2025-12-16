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

from src.RandomBaselineInference import RandomBaseline
from src.BertInference import BertInferencer
from src.SongBertPhase2 import SongBertModelPhase2
from src.SongBertPhase1 import SongBertModelPhase1

from src.Metrics import LexicalPlaylistMetrics, EmbeddingPlaylistMetrics


def load_song_data(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip",
    )

    df = df[["song", "lyrics"]]
    df = df.dropna(subset=["song", "lyrics"])
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

# def analyze_playlist_coverage(playlists, song_list):
#     """
#     Reports how many playlist tracks are missing from the lyric corpus.
#     """
#     song_set = {s.lower().strip() for s in song_list}

#     total_tracks = 0
#     missing_tracks = 0

#     missing_by_playlist = []

#     for playlist in playlists:
#         tracks = [t["track_name"] for t in playlist["tracks"]]
#         total_tracks += len(tracks)

#         missing = [
#             t for t in tracks
#             if t.lower().strip() not in song_set
#         ]
#         missing_tracks += len(missing)

#         missing_by_playlist.append(len(missing))

#     print("\n=== PLAYLIST COVERAGE ANALYSIS ===")
#     print(f"Total playlist tracks: {total_tracks}")
#     print(f"Tracks missing lyrics: {missing_tracks}")
#     print(f"Coverage: {(1 - missing_tracks / max(total_tracks, 1)) * 100:.2f}%")

#     if missing_by_playlist:
#         avg_missing = sum(missing_by_playlist) / len(missing_by_playlist)
#         max_missing = max(missing_by_playlist)

#         print(f"Avg missing tracks / playlist: {avg_missing:.2f}")
#         print(f"Max missing tracks in a playlist: {max_missing}")

#     return {
#         "total_tracks": total_tracks,
#         "missing_tracks": missing_tracks,
#         "coverage_pct": (1 - missing_tracks / max(total_tracks, 1)) * 100,
#     }



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



def plot_results(results, output_prefix: str, LEXICAL_METRICS, EMBEDDING_METRICS):
    all_metrics = list(next(iter(results.values())).keys())

    lexical_metrics = [m for m in all_metrics if m in LEXICAL_METRICS]
    embedding_metrics = [m for m in all_metrics if m in EMBEDDING_METRICS]

    if lexical_metrics:
        plot_metric_group(
            results,
            lexical_metrics,
            title="Playlist Generation — Lexical Metrics (Higher is Better)",
            ylabel="Score",
            output_path=output_prefix.replace(".png", "_lexical.png"),
        )

    if embedding_metrics:
        plot_metric_group(
            results,
            embedding_metrics,
            title="Playlist Generation — Embedding Metrics (Lower is Better)",
            ylabel="Distance",
            output_path=output_prefix.replace(".png", "_embedding.png"),
        )
def plot_metric_group(
    results: Dict[str, Dict[str, List[float]]],
    metric_names: List[str],
    title: str,
    ylabel: str,
    output_path: str,
):
    model_names = list(results.keys())
    num_models = len(model_names)
    num_metrics = len(metric_names)

    # Compute means: metric → model → value
    means = {
        metric: [
            sum(results[model][metric]) / len(results[model][metric])
            for model in model_names
        ]
        for metric in metric_names
    }

    x = range(num_metrics)
    width = 0.8 / num_models

    plt.figure(figsize=(12, 6))

    for i, model in enumerate(model_names):
        offsets = [xi + i * width for xi in x]
        plt.bar(
            offsets,
            [means[m][i] for m in metric_names],
            width=width,
            label=model,
        )

    plt.xticks(
        [xi + width * (num_models - 1) / 2 for xi in x],
        metric_names,
        rotation=15,
    )

    plt.ylabel(ylabel)
    plt.title(title)
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
    parser.add_argument("--songbert-p2-dir", default="trained_models/songbert_p2/final_model")
    parser.add_argument("--songbert-final-dir", default="trained_models/songbert_p2/final_model")
    parser.add_argument("--plot-out", default="baseline_performance.png")

    return parser.parse_args()
    
def main():
    args = parse_args()

    print("Loading song data...")
    df = load_song_data(args.song_tsv)
    song_list = df["song"].tolist()
    lyrics_list = df["lyrics"].tolist()

    print("Loading playlists...")
    playlists = load_playlists(args.playlist_json)


    #analyze_playlist_coverage(playlists, song_list)

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
    models["songbert_p2"] = load_songbert(song_list, tokenized_lyrics, tokenizer, args.max_length, args.songbert_p2_dir, SongBertModelPhase2)
    models["songbert_final"] = load_songbert(song_list, tokenized_lyrics, tokenizer, args.max_length, args.songbert_final_dir, SongBertModelPhase2)

    # keywords = "gold chain bougie drink pregame pump going out rap"
    # for model_name in models:
    #     model = models[model_name]
    #     print(model_name, model.generate_playlist(10, keywords))
    # return

    metrics = {}

    p1_embedding_metrics = EmbeddingPlaylistMetrics(models["songbert_p1"])
    p2_embedding_metrics = EmbeddingPlaylistMetrics(models["songbert_p2"])
    final_embedding_metrics = EmbeddingPlaylistMetrics(models["songbert_final"])

    LEXICAL_METRICS = {
        "percentage_overlap",
        "jaccard",
    }

    EMBEDDING_METRICS = {
        "p1_chamfer",
        "p2_chamfer",
        "final_chamfer",
    }

    metrics["percentage_overlap"] = LexicalPlaylistMetrics.percentage_overlap
    metrics["jaccard"] = LexicalPlaylistMetrics.jaccard
    metrics["p1_chamfer"] = p1_embedding_metrics.chamfer_distance
    metrics["p2_chamfer"] = p2_embedding_metrics.chamfer_distance
    metrics["final_chamfer"] = final_embedding_metrics.chamfer_distance

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

    plot_results(results, args.plot_out, LEXICAL_METRICS, EMBEDDING_METRICS)


if __name__ == "__main__":
    main()
