import os
import torch
import torch.nn as nn
import numpy as np

from .vocab import VOCAB_SIZE, BLANK_IDX, int_to_char

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'deepspeech2.pth')


class DeepSpeech2(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=512, num_rnn_layers=3, vocab_size=VOCAB_SIZE):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True),
        )

        rnn_input_size = 64 * input_dim

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_rnn_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)

        batch, ch, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, ch * freq)

        x, _ = self.rnn(x)
        x = self.classifier(x)
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
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)

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
