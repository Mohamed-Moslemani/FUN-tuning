import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir, genres, sample_rate=22050, duration=30):
        self.data_dir = data_dir
        self.genres = genres
        self.sample_rate = sample_rate
        self.duration = duration

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def load_data(self):
        features = []
        labels = []
        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)
            for filename in os.listdir(genre_dir):
                file_path = os.path.join(genre_dir, filename)
                features.append(self.extract_features(file_path))
                labels.append(self.genres.index(genre))
        return np.array(features), np.array(labels)

    def split_data(self, features, labels, test_size=0.2):
        return train_test_split(features, labels, test_size=test_size, random_state=42)

