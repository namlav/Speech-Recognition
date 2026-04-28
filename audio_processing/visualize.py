import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    return plt.gcf() # Trả về hình ảnh để Gradio hiển thị

def plot_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")
    return plt.gcf() # Trả về hình ảnh để Gradio hiển thị