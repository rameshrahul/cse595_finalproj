import numpy as np

class RandomBaseline:
    def __init__(self, song_list, seed=42):
        self.song_list = song_list
        self.rng = np.random.default_rng(seed=seed)


    def generate_playlist(self, k, keywords=''):
        return self.rng.choice(self.song_list, size=k, replace=False)

