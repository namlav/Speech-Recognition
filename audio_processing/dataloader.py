import torch
from torch.utils.data import Dataset
import numpy as np

class VivosDataset(Dataset):
    def __init__(self, dataset, precompute=True):
        self.dataset = dataset
        self.precompute = precompute
        
        if precompute:
            # Tính MFCC một lần, lưu vào RAM
            from audio_processing.preprocess import preprocess
            self.features = []
            self.texts = []
            for sample in dataset:
                feat = preprocess(sample["audio"])
                self.features.append(torch.tensor(feat, dtype=torch.float32))
                self.texts.append(sample["text"])
            print(f"[Dataset] Đã precompute {len(self.features)} mẫu MFCC")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.precompute:
            return self.features[idx], self.texts[idx]
        else:
            from audio_processing.preprocess import preprocess
            sample = self.dataset[idx]
            features = preprocess(sample["audio"])
            return torch.tensor(features, dtype=torch.float32), sample["text"]
