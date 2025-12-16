# SongBERT: Playlist-Aware Song Embeddings

**CSE 595 Final Project**  
University of Michigan — Vrinda Desai, Rahul Ramesh

---

## Overview

- Uses **BERT** to encode song lyrics
- Trains models to score `(target song, playlist context)` pairs
- Supports **classification and ranking objectives**
- Evaluates using **lexical** and **embedding-based** playlist metrics

---


## Repository Structure

.
├── src/
│   ├── Inference/
│   │   ├── BertInference.py
│   │   ├── Metrics.py
│   │   └── RandomBaselineInference.py
│   │
│   └── Training/
│       ├── Phase1Dataset.py
│       ├── Phase2Dataset.py
│       ├── SongBertPhase1.py
│       └── SongBertPhase2.py
│
└── evaluation_scripts/
    ├── evaluate_inference.py
    └── evaluate_inference.sh



---

## Evaluation

**Lexical playlist metrics**
- Overlap-based scores (e.g., Jaccard, precision)

**Embedding playlist metrics**
- Cosine distance
- Playlist cohesion
- Target–context similarity


## License

Released for academic and research use.
