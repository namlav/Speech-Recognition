import torch
import torch.nn as nn

from model.specaugment import SpecAugment


class SpeechRecognitionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.4,
        specaug_freq_mask: int = 10,
        specaug_time_mask: int = 30,
        specaug_n_freq: int = 2,
        specaug_n_time: int = 2,
    ):
        super(SpeechRecognitionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # ---- SpecAugment ----
        self.specaugment = SpecAugment(
            freq_mask_param=specaug_freq_mask,
            time_mask_param=specaug_time_mask,
            n_freq_masks=specaug_n_freq,
            n_time_masks=specaug_n_time,
        )

        # ---- Bidirectional LSTM ----
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- Layer Normalization ----
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout)

        # ---- FC layers ----
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- SpecAugment (only during training) ----
        x = self.specaugment(x)

        # ---- Bidirectional LSTM ----
        lstm_out, _ = self.lstm(x)

        # ---- Layer Normalization ----
        lstm_out = self.layer_norm(lstm_out)

        # ---- Dropout ----
        lstm_out = self.dropout(lstm_out)

        # ---- FC Block ----
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # ---- Log-Softmax for CTC ----
        out = nn.functional.log_softmax(out, dim=-1)

        return out
