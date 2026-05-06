import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, output_size=30):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


VOCAB = list("abcdefghijklmnopqrstuvwxyz ")

_model = None  # cache global


def load_model():
    global _model
    if _model is None:
        _model = SimpleRNN()
        _model.eval()
    return _model


def decode(output):
    indices = torch.argmax(output, dim=-1)
    text = ""
    for i in indices[0]:
        text += VOCAB[i % len(VOCAB)]
    return text


def predict(features):
    model = load_model()

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    return decode(output)