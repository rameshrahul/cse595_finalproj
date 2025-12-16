import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from transformers import DataCollatorWithPadding


class BertInferencer:
    def __init__(self, song_list, tokenized_lyrics, model, tokenizer, max_length, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        torch.set_grad_enabled(False)

        self.model = model.to( self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else None).eval()

        self.tokenizer = tokenizer
        self.max_length = max_length

        song_list = [s.lower().strip() for s in song_list]

        self.index_to_song = dict(enumerate(song_list))
        self.song_to_index = {song:idx for idx, song in self.index_to_song.items()}

        cls_vectors = self.encode_tokenized(tokenized_lyrics)
        self.vector_matrix = F.normalize(cls_vectors, dim=1).cpu().numpy()

        self.nbrs = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nbrs.fit(self.vector_matrix)

        with torch.inference_mode():
            unk_cls = self.get_cls_vector(self.tokenizer.unk_token)
            unk_cls = F.normalize(unk_cls, dim=1)
            self.unk_embedding = unk_cls.squeeze(0).cpu().numpy()

    def encode_tokenized(self, tokenized_batches):
        hidden = self.model.config.hidden_size
        n = sum(batch["input_ids"].size(0) for batch in tokenized_batches)

        cls_out = torch.empty(
            (n, hidden),
            dtype=torch.float32,
            device="cpu",
            pin_memory=(self.device.type == "cuda"),
        )

        offset = 0

        for batch in tqdm(tokenized_batches, desc="Encoding lyrics"):
            if self.device.type == "cuda":
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**batch)
            else:
                with torch.inference_mode():
                    outputs = self.model(**batch)

            cls = outputs.last_hidden_state[:, 0, :]
            bs = cls.size(0)

            cls_out[offset:offset + bs].copy_(cls.float(), non_blocking=True)
            offset += bs

        return cls_out

    def get_cls_vector(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**inputs)
        else:
            with torch.inference_mode():
                outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def generate_playlist(self, k, keywords=""):
        query = F.normalize(self.get_cls_vector(keywords), dim=1)
        _, indices = self.nbrs.kneighbors(query.cpu(), n_neighbors=k)
        return [self.index_to_song[i] for i in indices[0]]

    def get_song_embedding(self, song_name: str):
        song_name = song_name.lower().strip()
        if song_name in self.song_to_index:
            idx = self.song_to_index[song_name]
            return self.vector_matrix[idx]
        else:
            return self.unk_embedding


