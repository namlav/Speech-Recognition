from audio_processing.dataset import build_dataset
from audio_processing.preprocess import preprocess
from audio_processing.visualize import plot_waveform, plot_mfcc

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

# Visualize
plot_waveform(sample["audio"])
plot_mfcc(sample["audio"])
