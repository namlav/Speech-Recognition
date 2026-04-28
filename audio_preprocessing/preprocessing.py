import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 200  # padding length


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio


def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # (time, features)
    return mfcc


def pad_features(features):
    if len(features) < MAX_LEN:
        pad_width = MAX_LEN - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)))
    else:
        features = features[:MAX_LEN]
    return features


def preprocess(file_path):
    audio = load_audio(file_path)
    features = extract_features(audio)
    features = pad_features(features)
    return features


if __name__ == "__main__":
    path = "../data/sample.wav"
    features = preprocess(path)
    print("Feature shape:", features.shape)
