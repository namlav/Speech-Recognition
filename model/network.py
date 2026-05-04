import torch
import torch.nn as nn
from collections import OrderedDict

from model.specaugment import SpecAugment

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.GRU, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # RNN requires format (seq_len, batch, input_size)
        x, _ = self.rnn(x)
        
        if self.bidirectional:
            # (seq_len, batch, hidden_size * 2) -> (seq_len, batch, hidden_size)
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
            
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardtanh(0, 20, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpeechRecognitionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 39, # features
        hidden_dim: int = 512,
        num_layers: int = 5,
        num_classes: int = 100,
        dropout: float = 0.4,
        specaug_freq_mask: int = 10,
        specaug_time_mask: int = 30,
        specaug_n_freq: int = 2,
        specaug_n_time: int = 2,
        rnn_type=nn.GRU
    ):
        super(SpeechRecognitionModel, self).__init__()
        self.dropout_rate = dropout
        
        # SpecAugment
        self.specaugment = SpecAugment(
            freq_mask_param=specaug_freq_mask,
            time_mask_param=specaug_time_mask,
            n_freq_masks=specaug_n_freq,
            n_time_masks=specaug_n_time,
        )

        # CNN Layers (as in DeepSpeech 2)
        self.conv = nn.Sequential(
            CNNBlock(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            CNNBlock(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        )

        def calculate_conv_out_size(size):
            size = (size + 2 * 20 - 41) // 2 + 1
            size = (size + 2 * 10 - 21) // 2 + 1
            return size
            
        conv_out_dim = calculate_conv_out_size(input_dim)
        rnn_input_size = conv_out_dim * 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hidden_dim, rnn_type=rnn_type, 
                       bidirectional=True, batch_norm=False)
        rnns.append(("0", rnn))
        for x in range(num_layers - 1):
            rnn = BatchRNN(input_size=hidden_dim, hidden_size=hidden_dim, rnn_type=rnn_type, 
                           bidirectional=True, batch_norm=True)
            rnns.append((f"{x+1}", rnn))
            
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.fc = nn.Sequential(
            SequenceWise(nn.BatchNorm1d(hidden_dim)),
            nn.Linear(hidden_dim, num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor, output_lengths=None) -> torch.Tensor:
        # x expected to be (batch, time, feature)
        x = self.specaugment(x)
        
        # To CNN: (batch, 1, feature, time) -> Transpose to (batch, time, feature) inside if needed?
        # Standard input for DS2 is spectrograms (batch, 1, frequency, time)
        # Assuming x is (batch, time, freq)
        x = x.transpose(1, 2).unsqueeze(1) # (batch, 1, freq, time)
        
        x = self.conv(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dim
        x = x.transpose(1, 2) # (batch, time, feature)
        x = x.transpose(0, 1) # (time, batch, feature)
        
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
            
        x = self.fc(x) # (time, batch, num_classes)
        
        x = x.transpose(0, 1) # (batch, time, num_classes)
        out = nn.functional.log_softmax(x, dim=-1)
        return out
