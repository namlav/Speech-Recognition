import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 200


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.T  # (time, features)


def pad_or_truncate(features):
    if len(features) < MAX_LEN:
        pad_width = MAX_LEN - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)))
    else:
        features = features[:MAX_LEN]
    return features


def preprocess(file_path):
    audio = load_audio(file_path)
    mfcc = extract_mfcc(audio)
    mfcc = pad_or_truncate(mfcc)
    return mfcc


if __name__ == "__main__":
    path = "../data/vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav"
    features = preprocess(path)
    print("Shape:", features.shape)