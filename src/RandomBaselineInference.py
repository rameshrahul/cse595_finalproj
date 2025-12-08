import numpy as np

class RandomBaseline:
    def __init__(self, song_list):
        self.song_list = song_list


    def generate_playlist(self, k, keywords=''):
        return np.random.choice(self.song_list, size=k, replace=False)

