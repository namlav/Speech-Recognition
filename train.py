# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from audio_processing.dataset import build_dataset
from audio_processing.dataloader import VivosDataset
from model.vocab import CharVocabulary
from model.deepspeech import SimpleDeepSpeech

# 1. Cài đặt các siêu tham số (Chương 3.9)
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
DATA_PATH = "data/vivos" # Đổi lại đường dẫn đúng trên máy của nhóm

# 2. Khởi tạo Từ điển và Dữ liệu
vocab = CharVocabulary()
train_data = build_dataset(DATA_PATH, "train")
train_dataset = VivosDataset(train_data)

# Hàm collate để gom batch do độ dài text (label) khác nhau
def collate_fn(batch):
    features = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    
    # Stack features: shape (batch, 200, 13)
    features = torch.stack(features)
    
    # Chuyển text thành tensor và gộp lại
    target_lengths = [len(vocab.text_to_tensor(text)) for text in texts]
    targets = torch.cat([vocab.text_to_tensor(text) for text in texts])
    
    return features, targets, target_lengths

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 3. Khởi tạo Mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang huấn luyện trên: {device}")

model = SimpleDeepSpeech(
    input_features=13,  # Do preprocess n_mfcc = 13
    hidden_size=128, 
    num_classes=vocab.vocab_size
).to(device)

# 4. Khởi tạo Hàm mất mát và Tối ưu (Chương 3)
criterion = nn.CTCLoss(blank=0, zero_infinity=True) # Blank idx = 0 theo vocab
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. VÒNG LẶP HUẤN LUYỆN
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (features, targets, target_lengths) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(features) # Shape: (Time=200, Batch, Num_Classes)
        log_probs = nn.functional.log_softmax(outputs, dim=2)
        
        # CTCLoss yêu cầu input_lengths. 
        # Vì bạn của bạn đã fix cứng length=200 trong preprocess, nên ta tạo mảng toàn số 200
        input_lengths = torch.full(size=(features.size(0),), fill_value=200, dtype=torch.long)
        
        # Tính Loss
        loss = criterion(log_probs, targets, input_lengths, tuple(target_lengths))
        
        # Backward (BPTT - Mục 5.3)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
    print(f"--- Kết thúc Epoch {epoch+1} | Average Loss: {total_loss/len(train_loader):.4f} ---")

    #Lưu lại mô hình mỗi epoch
    torch.save(model.state_dict(), f"deepspeech_epoch_{epoch+1}.pth")
    print(f"Đã lưu mô hình tại: deepspeech_epoch_{epoch+1}.pth")