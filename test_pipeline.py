import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works headless

from audio_processing.dataset import build_dataset
from audio_processing.preprocess import preprocess
from audio_processing.visualize import plot_waveform, plot_mfcc
import matplotlib.pyplot as plt

DATA_PATH = "data/vivos"

# Load dataset
dataset = build_dataset(DATA_PATH, "train")

print("Total samples:", len(dataset))

# Test 1 sample
sample = dataset[0]

print("Audio path:", sample["audio"])
print("Text:", sample["text"])

# Preprocess
features = preprocess(sample["audio"])
print("Feature shape:", features.shape)

# Visualize (save to file instead of showing)
plot_waveform(sample["audio"])
plt.savefig("waveform.png")
plt.close()

plot_mfcc(sample["audio"])
plt.savefig("mfcc.png")
plt.close()
print("Saved waveform.png and mfcc.png")
