import os
import torch
import torch.nn as nn
import numpy as np

from .vocab import VOCAB_SIZE, BLANK_IDX, int_to_char

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'deepspeech2.pth')


class SequenceWise(nn.Module):
    """Collapses input of dim T*N*H to (T*N)*H, applies a module, then reshapes back.
    Used to apply BatchNorm1d / Linear across all time steps and batch items.
    The wrapped module is stored as self.module (matching checkpoint key names).
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x


class MaskConv(nn.Module):
    """Wraps a conv Sequential as self.seq_module (matching checkpoint key names)."""
    def __init__(self, seq_module):
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x):
        for module in self.seq_module:
            x = module(x)
        return x


class BatchRNN(nn.Module):
    """One bidirectional LSTM layer, optionally preceded by BatchNorm (via SequenceWise).
    After the bidirectional RNN, forward and backward directions are SUMMED
    (not concatenated), so output dim = hidden_size (not hidden_size*2).
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
            batch_first=False,
        )

    def forward(self, x):
        # x: (T, N, H)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            # Sum forward and backward directions: (T, N, H*2) -> (T, N, H)
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class DeepSpeech2(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=1024, num_rnn_layers=5, vocab_size=VOCAB_SIZE):
        super().__init__()

        # Conv stack matching the checkpoint architecture
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(1, 1), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(1, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        ))

        rnn_input_size = 32 * input_dim  # 32 * 41 = 1312

        # Stack of BatchRNN layers
        rnn_blocks = []
        for i in range(num_rnn_layers):
            if i == 0:
                rnn_blocks.append(BatchRNN(rnn_input_size, hidden_dim, batch_norm=False))
            else:
                rnn_blocks.append(BatchRNN(hidden_dim, hidden_dim, batch_norm=True))
        self.rnns = nn.Sequential(*rnn_blocks)

        # Fully-connected: BatchNorm1d -> Linear (no bias, matching checkpoint)
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False),
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        self.inference_log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (batch, time, freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, time, freq)
        else:
            # Already (batch, ch, time, freq) — ensure ch=1
            pass

        x = self.conv(x)  # (batch, 32, time, freq)

        batch, ch, time, freq = x.shape
        # Collapse channel & frequency: (batch, 32, time, freq) -> (batch, 32*freq, time)
        x = x.view(batch, ch * freq, time)
        # -> (batch, time, 32*freq)
        x = x.transpose(1, 2).contiguous()
        # -> (time, batch, 32*freq) for BatchRNN (expects T,N,H)
        x = x.transpose(0, 1)

        x = self.rnns(x)       # (time, batch, hidden_dim)
        x = self.fc(x)         # (time, batch, vocab_size)
        x = x.transpose(0, 1)  # (batch, time, vocab_size)

        x = self.inference_log_softmax(x)
        return x


_model_cache = None
_model_cache_path = None


def load_model(model_path=None):
    global _model_cache, _model_cache_path

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if _model_cache is not None and _model_cache_path == model_path:
        return _model_cache

    model = DeepSpeech2()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    _model_cache = model
    _model_cache_path = model_path
    return model


def ctc_greedy_decode(output):
    prev = BLANK_IDX
    result = []
    for idx in output:
        if idx != prev and idx != BLANK_IDX:
            result.append(int_to_char[idx])
        prev = idx
    return ''.join(result)


def predict(features, model_path=None):
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)

    if features.dim() == 2:
        features = features.unsqueeze(0)

    model = load_model(model_path)

    with torch.no_grad():
        output = model(features)
        output = output.squeeze(0)
        preds = torch.argmax(output, dim=-1).tolist()
        text = ctc_greedy_decode(preds)

    return text
