import torch.nn as nn

from src.models.components import ConvolutionFeatureExtractor, RNNLayer
from src.config import CONV_IN_CHANNELS, CONV_OUT_CHANNELS, RNN_HIDDEN_SIZE, RNN_DEPTH


class DeepSpeech2(nn.Module):
    def __init__(self, conv_in_channels=CONV_IN_CHANNELS, conv_out_channels=CONV_OUT_CHANNELS,
                 rnn_hidden_size=RNN_HIDDEN_SIZE, rnn_depth=RNN_DEPTH, vocab_size=None):
        super(DeepSpeech2, self).__init__()

        self.feature_extractor = ConvolutionFeatureExtractor(
            conv_in_channels, conv_out_channels
        )

        self.output_hidden_features = self.feature_extractor.conv_output_features

        self.rnns = nn.ModuleList(
            [
                RNNLayer(
                    input_size=self.output_hidden_features if i == 0 else 2 * rnn_hidden_size,
                    hidden_size=rnn_hidden_size,
                )
                for i in range(rnn_depth)
            ]
        )

        if vocab_size is None:
            from transformers import Wav2Vec2CTCTokenizer
            from src.config import TOKENIZER_NAME
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_NAME)
            vocab_size = tokenizer.vocab_size

        self.head = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size),
            nn.Hardtanh(),
            nn.Linear(rnn_hidden_size, vocab_size),
        )

    def forward(self, x, seq_lens):
        x, final_seq_lens = self.feature_extractor(x, seq_lens)

        for rnn in self.rnns:
            x = rnn(x, final_seq_lens)

        x = self.head(x)

        return x, final_seq_lens
