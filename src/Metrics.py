import numpy as np

class LexicalPlaylistMetrics:
    @staticmethod
    def percentage_overlap(generated, actual, **kwargs):
        g = {s.lower().strip() for s in generated}
        a = {s.lower().strip() for s in actual}
        hits = sum(1 for s in g if s in a)
        return hits / max(len(g), 1)

    @staticmethod
    def jaccard(generated, actual, **kwargs):
        g = {s.lower().strip() for s in generated}
        a = {s.lower().strip() for s in actual}
        if not g and not a:
            return 1.0
        return len(g & a) / len(g | a)


class EmbeddingPlaylistMetrics:
    def __init__(self, embedding_model):
        self.model = embedding_model
    @staticmethod
    def _cosine_distance(a, b, eps=1e-8):
        a = a / (np.linalg.norm(a) + eps)
        b = b / (np.linalg.norm(b) + eps)
        return 1.0 - np.dot(a, b)

    def _get_embeddings(self, playlist):
        return np.stack([
            self.model.get_song_embedding(s)
            for s in playlist
        ])
    def chamfer_distance(self, generated, actual, **kwargs):
        g_emb = self._get_embeddings(generated)
        a_emb = self._get_embeddings(actual)

        g_norm = g_emb / np.linalg.norm(g_emb, axis=1, keepdims=True)
        a_norm = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)

        sim = g_norm @ a_norm.T
        dist = 1.0 - sim

        g_to_a = dist.min(axis=1).mean()
        a_to_g = dist.min(axis=0).mean()

        return 0.5 * (g_to_a + a_to_g)
