import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True, **kwargs):
        super(MaskedConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs
        )

    def forward(self, x, seq_lens):
        batch_size, channels, height, width = x.shape

        output_seq_lens = self._compute_output_seq_len(seq_lens)

        conv_out = super().forward(x)

        mask = torch.zeros(batch_size, output_seq_lens.max(), device=x.device)
        for i, length in enumerate(output_seq_lens):
            mask[i, :length] = 1

        mask = mask.unsqueeze(1).unsqueeze(1)
        conv_out = conv_out * mask

        return conv_out, output_seq_lens

    def _compute_output_seq_len(self, seq_lens):
        return torch.floor(
            (seq_lens + (2 * self.padding[1]) - (self.kernel_size[1] - 1) - 1) // self.stride[1]
        ) + 1


class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(ConvolutionFeatureExtractor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size=(11, 21), stride=(2, 1), padding=(5, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.output_feature_dim = 20
        self.conv_output_features = self.output_feature_dim * self.out_channels

    def forward(self, x, seq_lens):
        x, seq_lens = self.conv1(x, seq_lens)
        x = self.bn1(x)
        x = torch.nn.functional.hardtanh(x)

        x, seq_lens = self.conv2(x, seq_lens)
        x = self.bn2(x)
        x = torch.nn.functional.hardtanh(x)

        x = x.permute(0, 3, 1, 2).flatten(2)

        return x, seq_lens


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(RNNLayer, self).__init__()

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.layernorm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x, seq_lens):
        batch, seq_len, embed_dim = x.shape

        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)

        out, _ = self.rnn(packed_x)

        x, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=seq_len, batch_first=True)

        x = self.layernorm(x)

        return x
