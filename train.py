"""
train.py
========
Script huấn luyện mô hình Nhận diện Giọng nói (Speech Recognition)
sử dụng dataset VIVOS và kiến trúc Bidirectional LSTM + CTC Loss.

Kiến thức áp dụng:
  - Chương 3.5: Thuật toán tối ưu AdamW.
  - Chương 3.6: Learning Rate Scheduler (ReduceLROnPlateau).
  - Chương 5:   CTC Loss cho bài toán sequence-to-sequence không alignment.
  - Kỹ thuật Dropout chống Overfitting (tích hợp trong model).

Cách chạy:
    python train.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Thêm thư mục gốc vào sys.path để import được các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from audio_processing.dataset import build_dataset
from audio_processing.dataloader import VivosDataset
from model.network import SpeechRecognitionModel


# ================================================================
# 1. BỘ TỪ ĐIỂN KÝ TỰ TIẾNG VIỆT (Character Map)
# ================================================================
# CTC Blank token luôn ở index 0 (quy ước mặc định của PyTorch CTCLoss).
# Các ký tự bắt đầu từ index 1.

VIETNAMESE_CHARS = [
    # Chữ thường không dấu
    "a", "b", "c", "d", "e", "g", "h", "i", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y",
    # Nguyên âm có dấu thanh: huyền (`), sắc ('), hỏi (?), ngã (~), nặng (.)
    "à", "á", "ả", "ã", "ạ",
    "è", "é", "ẻ", "ẽ", "ẹ",
    "ì", "í", "ỉ", "ĩ", "ị",
    "ò", "ó", "ỏ", "õ", "ọ",
    "ù", "ú", "ủ", "ũ", "ụ",
    "ỳ", "ý", "ỷ", "ỹ", "ỵ",
    # Nguyên âm có mũ + dấu thanh
    "â", "ầ", "ấ", "ẩ", "ẫ", "ậ",
    "ê", "ề", "ế", "ể", "ễ", "ệ",
    "ô", "ồ", "ố", "ổ", "ỗ", "ộ",
    # Nguyên âm có móc + dấu thanh
    "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ",
    "ơ", "ờ", "ớ", "ở", "ỡ", "ợ",
    "ư", "ừ", "ứ", "ử", "ữ", "ự",
    # Phụ âm đặc biệt
    "đ",
    # Khoảng trắng và dấu câu
    " ",
    ".", ",", "?", "!", "-", "/", ":", ";",
    # Chữ số
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Chữ hoa (nếu có trong dataset)
    "A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y",
    "À", "Á", "Ả", "Ã", "Ạ",
    "È", "É", "Ẻ", "Ẽ", "Ẹ",
    "Ì", "Í", "Ỉ", "Ĩ", "Ị",
    "Ò", "Ó", "Ỏ", "Õ", "Ọ",
    "Ù", "Ú", "Ủ", "Ũ", "Ụ",
    "Ỳ", "Ý", "Ỷ", "Ỹ", "Ỵ",
    "Â", "Ầ", "Ấ", "Ẩ", "Ẫ", "Ậ",
    "Ê", "Ề", "Ế", "Ể", "Ễ", "Ệ",
    "Ô", "Ồ", "Ố", "Ổ", "Ỗ", "Ộ",
    "Ă", "Ằ", "Ắ", "Ẳ", "Ẵ", "Ặ",
    "Ơ", "Ờ", "Ớ", "Ở", "Ỡ", "Ợ",
    "Ư", "Ừ", "Ứ", "Ử", "Ữ", "Ự",
    "Đ",
]


def build_char_map(chars: list) -> dict:
    """
    Xây dựng ánh xạ ký tự → số nguyên.
    Index 0 dành cho CTC blank token.
    Các ký tự bắt đầu từ index 1.

    Tham số:
        chars: Danh sách các ký tự tiếng Việt.

    Trả về:
        char_to_idx: dict {char: index}
        idx_to_char: dict {index: char}
        num_classes: tổng số class = len(chars) + 1 (blank)
    """
    # Blank token ở index 0
    char_to_idx = {"<blank>": 0}
    idx_to_char = {0: "<blank>"}

    for idx, char in enumerate(chars, start=1):
        # Tránh trùng lặp nếu có
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx_to_char[idx] = char

    # Đánh lại index liên tục
    unique_chars = list(char_to_idx.keys())
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    idx_to_char = {i: c for i, c in enumerate(unique_chars)}

    num_classes = len(char_to_idx)
    print(f"[CharMap] Tổng số class (gồm blank): {num_classes}")
    print(f"[CharMap] Ký tự: {unique_chars}")

    return char_to_idx, idx_to_char, num_classes


def encode_text(text: str, char_to_idx: dict) -> list:
    """
    Mã hóa chuỗi văn bản thành danh sách chỉ số nguyên.
    Bỏ qua các ký tự không có trong từ điển.

    Tham số:
        text:         Chuỗi văn bản tiếng Việt.
        char_to_idx:  Ánh xạ ký tự → index.

    Trả về:
        Danh sách các index tương ứng.
    """
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
        # Bỏ qua các ký tự không có trong từ điển (fallback: bỏ qua)
    return indices


def collate_fn(batch, char_to_idx: dict):
    """
    Hàm collate cho DataLoader – gom batch lại và chuẩn bị target cho CTCLoss.

    CTCLoss yêu cầu:
      - input_lengths:  độ dài thực tế của mỗi chuỗi input  (batch,)
      - target_lengths: độ dài mỗi chuỗi nhãn               (batch,)
      - targets:        các chỉ số nối liền nhau             (sum(target_lengths),)

    Tham số:
        batch:        List các tuple (features, text) từ VivosDataset.
        char_to_idx:  Ánh xạ ký tự → index.

    Trả về:
        features:       Tensor (batch, 200, 13)
        targets:        Tensor (sum(target_lengths),) – các index nối dài
        input_lengths:  Tensor (batch,) – độ dài input (200 cho tất cả)
        target_lengths: Tensor (batch,) – độ dài từng chuỗi nhãn
    """
    features_list = []
    target_list = []
    target_lengths_list = []

    for features, text in batch:
        features_list.append(features)

        encoded = encode_text(text, char_to_idx)
        target_list.extend(encoded)                      # Nối dài tất cả target
        target_lengths_list.append(len(encoded))

    features = torch.stack(features_list, dim=0)              # (batch, 200, 13)
    targets = torch.tensor(target_list, dtype=torch.long)    # (sum(target_len),)
    input_lengths = torch.full(
        (len(features_list),), 200, dtype=torch.long
    )                                                         # (batch,) – mọi input dài 200
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)  # (batch,)

    return features, targets, input_lengths, target_lengths


# ================================================================
# 2. HÀM HUẤN LUYỆN MỘT EPOCH
# ================================================================
def train_one_epoch(
    model, dataloader, ctc_loss_fn, optimizer, device, char_to_idx
):
    """
    Huấn luyện một epoch.

    Trả về:
        average_loss: Loss trung bình trên toàn bộ epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_data in dataloader:
        features, targets, input_lengths, target_lengths = batch_data

        features = features.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        # ---- Forward ----
        # log_probs: (batch, 200, num_classes)
        log_probs = model(features)

        # Chuyển về (T, N, C) = (200, batch, num_classes) cho CTCLoss
        log_probs = log_probs.permute(1, 0, 2)  # (200, batch, num_classes)

        # ---- Tính CTC Loss ----
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        # ---- Backpropagation ----
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping – tránh exploding gradient trong RNN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ================================================================
# 3. HÀM ĐÁNH GIÁ (VALIDATION)
# ================================================================
@torch.no_grad()
def validate(model, dataloader, ctc_loss_fn, device, char_to_idx):
    """
    Đánh giá mô hình trên tập validation.

    Trả về:
        average_loss: Loss trung bình trên tập validation.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_data in dataloader:
        features, targets, input_lengths, target_lengths = batch_data

        features = features.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs = model(features)
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ================================================================
# 4. HÀM CHÍNH – MAIN
# ================================================================
def main():
    # -------------------- Cấu hình --------------------
    DATA_PATH = "data/vivos"                 # Đường dẫn tới dataset VIVOS
    BATCH_SIZE = 128                          # Batch size
    NUM_EPOCHS = 50                          # Số epoch tối đa
    LEARNING_RATE = 1e-3                     # Learning rate khởi tạo
    HIDDEN_DIM = 256                         # Số unit ẩn LSTM
    NUM_LAYERS = 3                           # Số tầng LSTM
    DROPOUT = 0.3                            # Tỉ lệ Dropout
    WEIGHT_DECAY = 1e-4                      # Weight decay cho AdamW
    SAVE_DIR = "saved_models"                # Thư mục lưu model
    MODEL_PATH = os.path.join(SAVE_DIR, "model.pth")

    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Sử dụng: {device}")

    # -------------------- Tạo thư mục lưu model --------------------
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ================================================================
    # Bước 1: Xây dựng bộ từ điển ký tự
    # ================================================================
    char_to_idx, idx_to_char, num_classes = build_char_map(VIETNAMESE_CHARS)

    # ================================================================
    # Bước 2: Tải dữ liệu VIVOS
    # ================================================================
    print("\n[Data] Đang tải dữ liệu VIVOS...")
    train_data = build_dataset(DATA_PATH, "train")
    test_data = build_dataset(DATA_PATH, "test")  # VIVOS có split "test"

    print(f"[Data] Train samples: {len(train_data)}")
    print(f"[Data] Test samples:  {len(test_data)}")

    # Tạo Dataset PyTorch
    train_dataset = VivosDataset(train_data)
    test_dataset = VivosDataset(test_data)

    # Tạo DataLoader với collate_fn tùy chỉnh
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx),
        pin_memory=True,
        num_workers=2,   # Đặt 0 để tránh lỗi trên Windows
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx),
        pin_memory=True,
        num_workers=1,
    )

    # ================================================================
    # Bước 3: Khởi tạo mô hình
    # ================================================================
    print("\n[Model] Khởi tạo SpeechRecognitionModel...")
    model = SpeechRecognitionModel(
        input_dim=13,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    # In tổng quan kiến trúc
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Tổng tham số:     {total_params:,}")
    print(f"[Model] Tham số huấn luyện: {trainable_params:,}")

    # ================================================================
    # Bước 4: Hàm mất mát CTC (Connectionist Temporal Classification)
    #         Chương 5 – CTC Loss chuyên dụng cho bài toán nhận diện giọng nói.
    #         blank=0: token blank ở index 0 trong char_to_idx.
    #         zero_infinity=True: tránh loss = inf khi không có đường alignment hợp lệ.
    # ================================================================
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # ================================================================
    # Bước 5: Optimizer – AdamW (Chương 3.5)
    #         AdamW tách weight decay khỏi gradient update, cho hiệu quả
    #         regularization tốt hơn Adam truyền thống.
    # ================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ================================================================
    # Bước 6: Learning Rate Scheduler – ReduceLROnPlateau (Chương 3.6)
    #         Giảm learning rate khi loss trên validation plateau (không cải thiện).
    #         factor=0.5: giảm một nửa LR.
    #         patience=5:   chờ 5 epoch không cải thiện mới giảm.
    # ================================================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",         # Theo dõi loss (càng thấp càng tốt)
        factor=0.5,         # Giảm LR xuống 1/2
        patience=5,         # Chờ 5 epoch
        min_lr=1e-6,        # Ngưỡng dưới của LR
    )

    # ================================================================
    # Bước 7: Vòng lặp huấn luyện
    # ================================================================
    print("\n" + "=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10  # Dừng sớm nếu không cải thiện sau 10 epoch

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Huấn luyện ----
        train_loss = train_one_epoch(
            model, train_loader, ctc_loss_fn, optimizer, device, char_to_idx
        )

        # ---- Validation ----
        val_loss = validate(
            model, test_loader, ctc_loss_fn, device, char_to_idx
        )

        # ---- Cập nhật Scheduler ----
        scheduler.step(val_loss)

        # Lấy LR hiện tại để in
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ---- Lưu model tốt nhất ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "num_classes": num_classes,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                },
                MODEL_PATH,
            )
            print(f"  >>> Đã lưu model tốt nhất (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stopping] Dừng sau {epoch} epoch (không cải thiện).")
                break

    print("\n" + "=" * 60)
    print(f"HOÀN THÀNH HUẤN LUYỆN")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model đã lưu tại: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
