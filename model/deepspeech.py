# model/deepspeech.py
import torch
import torch.nn as nn

class SimpleDeepSpeech(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super(SimpleDeepSpeech, self).__init__()
        
        # 1. Lớp tích chập (CNN) - Để rút trích đặc trưng cục bộ
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # 2. Mạng nơ-ron tái diễn (RNN/LSTM) hai chiều 
        self.rnn = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
        # 3. Lớp phân loại MLP 
        # Kích thước x2 vì dùng LSTM hai chiều (Bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Đầu vào x từ bạn của bạn có shape: (batch, time=200, features=13)
        # Conv1d của PyTorch yêu cầu shape: (batch, features, time)
        x = x.permute(0, 2, 1) 
        
        # Đi qua CNN
        x = self.cnn(x)
        
        # Đổi ngược lại shape cho LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # Đi qua LSTM
        x, _ = self.rnn(x)
        
        # Đi qua lớp Linear cuối cùng để ra xác suất từng ký tự
        x = self.fc(x)
        
        # CTCLoss của PyTorch mặc định yêu cầu shape (Time, Batch, Classes)
        x = x.permute(1, 0, 2) 
        return x