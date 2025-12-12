import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random


class PlaylistDataset3(Dataset):
    def __init__(self, playlist_data, total_songs, tokenizer, songid_to_tokenized_lyrics, num_songs=5, num_samples=3, name_length=10):
        """ Args:
            adjacency_list: Dict where keys are song lyrics and values are sets of 
                            song lyrics that appear in the same playlist
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
            num_context_songs: Number of context songs per target song"""

        self.data = []
        self.tokenizer = tokenizer
        self.name_length = name_length
        self.songid_to_tokenized_lyrics = songid_to_tokenized_lyrics

        # playlist data is [(playlist_name, [song_ids])]
        for playlist in playlist_data:
            name = playlist[0]
            pos_population = playlist[1] # -> 5 samples from just these
            # number of songs in sample and how many samples

            total = num_songs * num_samples

            if len(pos_population) < total:
                pos_songs = random.choices(pos_population, k=total)
            else:
                pos_songs = random.sample(pos_population, k=total)

            pos_samples = [pos_songs[i:i + num_songs] for i in range(0, total, num_songs)]

            neg_songs = []
            for i in range(total):
                neg_song = random.randint(0, total_songs)
                while neg_song in pos_population:
                    neg_song = random.randint(0, total_songs)
                neg_songs.append(neg_song)

            neg_samples = [neg_songs[i:i + num_songs] for i in range(0, total, num_songs)]

            context_songs = pos_samples + neg_samples

            random.shuffle(context_songs)

            labels = [1.0 if song[0] in pos_population else 0.0 for song in context_songs]

            self.data.append({
                'target': name, # target is just name
                'context': context_songs, # context is list of list of songs
                'labels': labels
            })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize target song
        target_tokens = self.tokenizer(item['target'], truncation=True, padding="max_length",
        max_length=self.name_length, return_tensors="pt")

        context_tokens = [] # context tokens is a list of lists
        for sample in item['context']: # sample is 5 pos/neg songs
            songs = []
            for song in sample:
                songs.append(self.songid_to_tokenized_lyrics[song])
            context_tokens.append(songs)

        
        # Tokenize context songs
        # context_tokens = [
        #     self.songid_to_tokenized_lyrics[song] 
        #     for song in item['context']
        # ]
        
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        
        return {'target_tokens': target_tokens,
            'context_tokens': context_tokens,
            'labels': labels}
    

class PlaylistData3Collator:
    def __call__(self, batch):
        target_ids = torch.stack([b["target_tokens"]["input_ids"].squeeze(0) for b in batch])
        target_mask = torch.stack([b["target_tokens"]["attention_mask"].squeeze(0) for b in batch])


        batch_ids = []
        batch_masks = []

        for b in batch:
            context_ids = []
            context_mask = []
            for context_sample in b["context_tokens"]: # this is one list
                # stack all songs in the sample
                songs_in_sample = torch.stack([song["input_ids"].squeeze(0) for song in context_sample])
                context_ids.append(songs_in_sample)
            for context_sample in b["context_tokens"]:
                songs_in_sample = torch.stack([song["attention_mask"].squeeze(0) for song in context_sample])
                context_mask.append(songs_in_sample)
            batch_ids.append(torch.stack(context_ids))
            batch_masks.append(torch.stack(context_mask))

        # context_ids = [torch.stack([c["input_ids"].squeeze(0) for c in b["context_tokens"]]) for b in batch]
        # context_mask = [torch.stack([c["attention_mask"].squeeze(0) for c in b["context_tokens"]]) for b in batch]

        return {
            "target_input_ids": target_ids,
            "target_attention_mask": target_mask,
            "context_input_ids": torch.stack(batch_ids),
            "context_attention_mask": torch.stack(batch_masks),
            "labels": torch.stack([b["labels"] for b in batch])
        }





