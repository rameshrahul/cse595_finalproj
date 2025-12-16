import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers import EvalPrediction
import numpy as np
from tqdm import tqdm
import random


class PlaylistDataset(Dataset):
    def __init__(self, adjacency_list, songid_to_tokenized_lyrics, max_length=512, num_context_songs=5, random_seed=42):
        self.data = []
        self.max_length = max_length
        self.songid_to_tokenized_lyrics = songid_to_tokenized_lyrics

        random.seed(random_seed)
        
        
        all_songs = list(adjacency_list.keys())
        
        for target_song, related_songs in tqdm(adjacency_list.items()):
            if len(related_songs) == 0:
                continue
            
            positive_songs = list(related_songs)
            num_positive = min(len(positive_songs), num_context_songs // 2)
            num_negative = num_context_songs - num_positive
            
            # Get negative examples (songs NOT in same playlists)
            negative_songs = []
            while len(negative_songs) != num_negative:
                random_song = random.choice(all_songs)
                if random_song not in related_songs:
                    negative_songs.append(random_song)
            
            
            context_songs = (
                random.sample(positive_songs, num_positive) +
                negative_songs
            )
            random.shuffle(context_songs)
            
            labels = [1.0 if song in related_songs else 0.0 for song in context_songs]
            
            self.data.append({
                'target': target_song,
                'context': context_songs,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        target_tokens = self.songid_to_tokenized_lyrics[item['target']]    
        context_tokens = [
            self.songid_to_tokenized_lyrics[song] 
            for song in item['context']
        ]
        
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        
        return {'target_tokens': target_tokens,
            'context_tokens': context_tokens,
            'labels': labels}


class PlaylistDataCollator:
    def __call__(self, batch):
        target_ids = torch.stack([b["target_tokens"]["input_ids"].squeeze(0) for b in batch])
        target_mask = torch.stack([b["target_tokens"]["attention_mask"].squeeze(0) for b in batch])

        context_ids = [torch.stack([c["input_ids"].squeeze(0) for c in b["context_tokens"]]) for b in batch]
        context_mask = [torch.stack([c["attention_mask"].squeeze(0) for c in b["context_tokens"]]) for b in batch]

        return {
            "target_input_ids": target_ids,
            "target_attention_mask": target_mask,
            "context_input_ids": torch.stack(context_ids),
            "context_attention_mask": torch.stack(context_mask),
            "labels": torch.stack([b["labels"] for b in batch])
        }
