import torch
from torch.utils.data import Dataset
import numpy as np
from audio_processing.preprocess import preprocess_with_speed, MAX_LEN


class VivosDataset(Dataset):
    def __init__(self, dataset, precompute=True, use_delta=True,
                 speed_perturb=False, augment=True):
        self.dataset = dataset
        self.precompute = precompute
        self.use_delta = use_delta
        self.speed_perturb = speed_perturb
        self.augment = augment

        if precompute:
            self.features = []
            self.texts = []

            speeds = [0.9, 1.0, 1.1] if speed_perturb else [1.0]

            for sample in dataset:
                audio_path = sample["audio"]
                for sp in speeds:
                    feat = preprocess_with_speed(
                        audio_path,
                        use_delta=use_delta,
                        max_len=MAX_LEN,
                        speed_factor=sp,
                        apply_gain=augment and sp != 1.0,
                    )
                    self.features.append(torch.tensor(feat, dtype=torch.float32))
                    self.texts.append(sample["text"])

            print(f"[Dataset] Precomputed {len(self.features)} MFCC samples"
                  f" (speed_perturb={speed_perturb}, use_delta={use_delta})")

    def __len__(self):
        return len(self.features) if self.precompute else len(self.dataset)

    def __getitem__(self, idx):
        if self.precompute:
            return self.features[idx], self.texts[idx]
        else:
            from audio_processing.preprocess import preprocess_with_speed
            sample = self.dataset[idx]
            sp = 1.0
            if self.speed_perturb:
                sp = np.random.choice([0.9, 1.0, 1.1])
            features = preprocess_with_speed(
                sample["audio"],
                use_delta=self.use_delta,
                max_len=MAX_LEN,
                speed_factor=sp,
                apply_gain=self.augment,
            )
            return torch.tensor(features, dtype=torch.float32), sample["text"]
