import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 300


def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def extract_mfcc(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T  # (time, features)


def extract_mfcc_with_delta(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)  # (n_mfcc, time)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, time)
    return stacked.T  # (time, n_mfcc*3)


def pad_or_truncate(features, max_len=MAX_LEN, return_length=False):
    orig_len = len(features)
    if orig_len < max_len:
        pad_width = max_len - orig_len
        features = np.pad(features, ((0, pad_width), (0, 0)))
    else:
        features = features[:max_len]
    if return_length:
        return features, min(orig_len, max_len)
    return features


def speed_perturb_audio(audio, speed_factor):
    if speed_factor == 1.0:
        return audio
    new_length = int(len(audio) / speed_factor)
    old_indices = np.arange(len(audio), dtype=np.float64)
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


def apply_gain_augmentation(audio, max_gain_db=3.0):
    gain_db = np.random.uniform(-max_gain_db, max_gain_db)
    scale = 10 ** (gain_db / 20.0)
    return audio * scale


def preprocess(file_path, use_delta=True, max_len=MAX_LEN, apply_gain=False, return_length=False):
    audio = load_audio(file_path)
    if apply_gain:
        audio = apply_gain_augmentation(audio)
    if use_delta:
        features = extract_mfcc_with_delta(audio)
    else:
        features = extract_mfcc(audio)
    if return_length:
        features, length = pad_or_truncate(features, max_len, return_length=True)
        return features, length
    features = pad_or_truncate(features, max_len)
    return features


def preprocess_with_speed(file_path, use_delta=True, max_len=MAX_LEN,
                          speed_factor=1.0, apply_gain=True, return_length=False):
    audio = load_audio(file_path)
    if speed_factor != 1.0:
        audio = speed_perturb_audio(audio, speed_factor)
    if apply_gain:
        audio = apply_gain_augmentation(audio)
    if use_delta:
        features = extract_mfcc_with_delta(audio)
    else:
        features = extract_mfcc(audio)
    if return_length:
        features, length = pad_or_truncate(features, max_len, return_length=True)
        return features, length
    features = pad_or_truncate(features, max_len)
    return features


if __name__ == "__main__":
    path = "../data/vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav"
    features = preprocess(path, use_delta=True)
    print("Shape (with delta):", features.shape)
