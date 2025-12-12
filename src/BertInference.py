import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class BertInferencer:

    def batch_encode(self, lyrics_list, batch_size=64):
        vectors = []

        for i in tqdm(range(0, len(lyrics_list), batch_size)):
            batch = lyrics_list[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to("cpu")

            with torch.no_grad():
                outputs = self.model(**inputs)

            vectors.append(outputs.last_hidden_state[:, 0, :].cpu())

        return torch.cat(vectors)


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
        #self.vector_matrix = self.batch_encode(lyrics_list)
        self.index_to_song = index_to_song
        self.nbrs = NearestNeighbors(algorithm='ball_tree').fit(self.vector_matrix)
            


    def generate_playlist(self, k, keywords=''):
        new_cls = self.get_cls_vector(keywords)
        distances, indices = self.nbrs.kneighbors(new_cls, n_neighbors=k)
        return [self.index_to_song[x] for x in indices[-1]]

