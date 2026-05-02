import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=30,
                 n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        """
        x: (batch, time, freq)
        Chỉ áp dụng khi training (self.training=True).
        """
        if not self.training:
            return x

        B, T, F = x.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = min(self.freq_mask_param, F)
            if f > 0:
                f0 = torch.randint(0, F - f + 1, (1,)).item()
                x[:, :, f0:f0 + f] = 0

        # Time masking
        for _ in range(self.n_time_masks):
            t = min(self.time_mask_param, T)
            if t > 0:
                t0 = torch.randint(0, T - t + 1, (1,)).item()
                x[:, t0:t0 + t, :] = 0

        return x
