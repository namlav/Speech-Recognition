import torch
from torch.utils.data import Dataset
from audio_processing.preprocess import preprocess


class VivosDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        features = preprocess(sample["audio"])
        text = sample["text"]

        features = torch.tensor(features, dtype=torch.float32)

        return features, text