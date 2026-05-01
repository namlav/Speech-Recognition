"""
model/network.py
================
Kiến trúc mô hình Nhận diện Giọng nói sử dụng Bidirectional LSTM + CTC.

Kiến thức áp dụng:
  - Chương 5: Mạng RNN (Bidirectional LSTM) cho dữ liệu chuỗi thời gian.
  - Chương 3.5: Kỹ thuật chống Overfitting – Dropout.
  - CTC (Connectionist Temporal Classification) cho bài toán sequence-to-sequence
    không cần alignment.

Input:  Tensor MFCC shape (batch, 200, 13)
Output: Log-probabilities shape (batch, 200, num_classes)  (chưa transpose)
"""

import torch
import torch.nn as nn


class SpeechRecognitionModel(nn.Module):
    """
    Mô hình nhận diện giọng nói dùng Bidirectional LSTM.

    Kiến trúc:
        Input (batch, 200, 13)
          │
          ▼
        Bidirectional LSTM (3 tầng, hidden_dim=256)
          │  ── Dropout chống overfitting giữa các tầng LSTM
          ▼
        LayerNorm  ── ổn định phân phối activation
          │
          ▼
        Dropout (p=0.3)  ── chống overfitting trước lớp FC
          │
          ▼
        Linear(hidden_dim*2, hidden_dim*2)
          │  ── ReLU + Dropout
          ▼
        Linear(hidden_dim*2, num_classes)
          │
          ▼
        Log-Softmax(dim=-1) → (batch, 200, num_classes)

    Tham số:
        input_dim (int):   Số đặc trưng MFCC đầu vào. Mặc định 13.
        hidden_dim (int):  Số unit ẩn mỗi chiều LSTM. Mặc định 256.
        num_layers (int):  Số tầng LSTM chồng lên nhau. Mặc định 3.
        num_classes (int): Số lượng ký tự đầu ra (gồm blank của CTC). Mặc định 100.
        dropout (float):   Xác suất Dropout. Mặc định 0.3.
    """

    def __init__(
        self,
        input_dim: int = 13,          # N_MFCC = 13 từ preprocess.py
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 100,       # Sẽ được set lại khi biết bộ từ điển
        dropout: float = 0.3,
    ):
        super(SpeechRecognitionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # ================================================================
        # Bidirectional LSTM – Chương 5: Mạng RNN
        #   - bidirectional=True: học ngữ cảnh cả hai chiều thời gian
        #   - batch_first=True:   tensor đầu vào/ra dạng (batch, seq, feature)
        #   - dropout trong LSTM: áp dụng giữa các tầng (trừ tầng cuối)
        # ================================================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ================================================================
        # Layer Normalization – ổn định huấn luyện, giảm internal covariate shift
        #   input: (batch, 200, hidden_dim * 2)  vì bidirectional
        # ================================================================
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # ================================================================
        # Dropout – Chương 3.5: Chống Overfitting
        #   Ngẫu nhiên tắt một tỉ lệ neuron trong quá trình training
        # ================================================================
        self.dropout = nn.Dropout(dropout)

        # ================================================================
        # Fully Connected – ánh xạ từ hidden states ra phân phối ký tự
        #   Dùng 2 lớp FC với ReLU + Dropout xen giữa để tăng khả năng biểu diễn
        # ================================================================
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến.

        Tham số:
            x: Tensor MFCC shape (batch, 200, 13)

        Trả về:
            Tensor log-probabilities shape (batch, 200, num_classes)
            (Dùng cho CTCLoss sau khi transpose về (T, N, C))
        """
        # ---- Bidirectional LSTM ----
        # lstm_out: (batch, 200, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # ---- Layer Normalization ----
        lstm_out = self.layer_norm(lstm_out)

        # ---- Dropout (chống overfitting) ----
        lstm_out = self.dropout(lstm_out)

        # ---- FC Block ----
        out = self.fc1(lstm_out)       # (batch, 200, hidden_dim*2)
        out = self.relu(out)
        out = self.dropout(out)         # Dropout lần 2 trước lớp output
        out = self.fc2(out)            # (batch, 200, num_classes)

        # ---- Log-Softmax cho CTC ----
        # CTCLoss yêu cầu log-probabilities
        out = nn.functional.log_softmax(out, dim=-1)

        return out
