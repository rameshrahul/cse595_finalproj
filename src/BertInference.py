import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class BertInferencer:

    def __init__(self, song_list, lyrics_list, model, tokenizer, max_length, batch_size=32):
        # ---- Device & backend setup (A100 optimized) ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        torch.set_grad_enabled(False)

        # ---- Model ----
        if self.device.type == "cuda":
            self.model = model.to(self.device, dtype=torch.bfloat16).eval()
        else:
            self.model = model.to(self.device).eval()

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.index_to_song = dict(enumerate(song_list))

        # ---- Build index (hot path) ----
        cls_vectors = self.batch_encode(lyrics_list, batch_size)

        self.vector_matrix = F.normalize(cls_vectors, dim=1).cpu().numpy()

        #self.nbrs = NearestNeighbors(algorithm="ball_tree")
        self.nbrs = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nbrs.fit(self.vector_matrix)

    def batch_encode(self, lyrics_list, batch_size=16):
        n = len(lyrics_list)
        hidden = self.model.config.hidden_size

        # Preallocate output (CPU, pinned)
        cls_out = torch.empty(
            (n, hidden),
            dtype=torch.float32,
            device="cpu",
            pin_memory=(self.device.type == "cuda"),
        )

        collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
        )

        offset = 0

        for i in tqdm(range(0, n, batch_size), desc="Encoding lyrics"):
            batch_lyrics = lyrics_list[i : i + batch_size]

            # Tokenize (CPU)
            inputs = self.tokenizer(
                batch_lyrics,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = collator([dict(zip(inputs, t)) for t in zip(*inputs.values())])

            if self.device.type == "cuda":
                inputs = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in inputs.items()
                }

                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(**inputs)
                    cls = outputs.last_hidden_state[:, 0, :]
            else:
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    cls = outputs.last_hidden_state[:, 0, :]

            bs = cls.size(0)
            cls_out[offset : offset + bs].copy_(cls.float(), non_blocking=True)
            offset += bs

        return cls_out

    def get_cls_vector(self, song_lyrics):
        inputs = self.tokenizer(
            song_lyrics,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(**inputs)
        else:
            with torch.inference_mode():
                outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def generate_playlist(self, k, keywords=""):
        new_cls = F.normalize(self.get_cls_vector(keywords), dim=1)
        distances, indices = self.nbrs.kneighbors(new_cls.cpu(), n_neighbors=k)
        return [self.index_to_song[x] for x in indices[-1]]



class BertInferencerOld:


    def get_cls_vector(self, song_lyrics):
        inputs1 = self.tokenizer(song_lyrics, return_tensors="pt")
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
        return outputs1.last_hidden_state[:, 0, :]
    
    def __init__(self, song_list, lyrics_list, tokenizer, bert_model):
        print("starting init function")
        torch.set_grad_enabled(False)
        self.tokenizer = tokenizer
        self.model = bert_model

        self.model = bert_model.to("cpu")
        
        # Load tokenizer and model
        self.model.eval()

        vector_matrix = []
        index_to_song = {}
        for i, lyrics in enumerate(tqdm(lyrics_list)):
             index_to_song[i] = song_list[i]
             vector_matrix.append(self.get_cls_vector(lyrics))

        self.vector_matrix = np.squeeze(np.array(vector_matrix))
        self.index_to_song = index_to_song
        self.nbrs = NearestNeighbors(algorithm='ball_tree').fit(self.vector_matrix)


    def generate_playlist(self, k, keywords=''):
        new_cls = self.get_cls_vector(keywords)
        distances, indices = self.nbrs.kneighbors(new_cls, n_neighbors=k)
        return [self.index_to_song[x] for x in indices[-1]]

