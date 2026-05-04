"""
train.py
========
Script huấn luyện mô hình Nhận diện Giọng nói (Speech Recognition)
sử dụng dataset VIVOS và kiến trúc Bidirectional LSTM + CTC Loss.

Cải tiến cho dataset nhỏ (15h):
  - Speed perturbation (0.9x, 1.0x, 1.1x) → dataset ×3
  - SpecAugment (time masking + frequency masking)
  - Delta + Delta-Delta features (13→39)
  - Warmup + Cosine Annealing LR schedule
  - Tăng dropout, giảm model size, giảm batch size
  - Volume augmentation
"""

import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from audio_processing.dataset import build_dataset
from audio_processing.dataloader import VivosDataset
from audio_processing.preprocess import MAX_LEN
from model.network import SpeechRecognitionModel


# ================================================================
# 1. BỘ TỪ ĐIỂN KÝ TỰ TIẾNG VIỆT
# ================================================================

VIETNAMESE_CHARS = [
    "a", "b", "c", "d", "e", "g", "h", "i", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y",
    "à", "á", "ả", "ã", "ạ",
    "è", "é", "ẻ", "ẽ", "ẹ",
    "ì", "í", "ỉ", "ĩ", "ị",
    "ò", "ó", "ỏ", "õ", "ọ",
    "ù", "ú", "ủ", "ũ", "ụ",
    "ỳ", "ý", "ỷ", "ỹ", "ỵ",
    "â", "ầ", "ấ", "ẩ", "ẫ", "ậ",
    "ê", "ề", "ế", "ể", "ễ", "ệ",
    "ô", "ồ", "ố", "ổ", "ỗ", "ộ",
    "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ",
    "ơ", "ờ", "ớ", "ở", "ỡ", "ợ",
    "ư", "ừ", "ứ", "ử", "ữ", "ự",
    "đ",
    " ",
    ".", ",", "?", "!", "-", "/", ":", ";",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
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
    char_to_idx = {"<blank>": 0}
    idx_to_char = {0: "<blank>"}

    for idx, char in enumerate(chars, start=1):
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx_to_char[idx] = char

    unique_chars = list(char_to_idx.keys())
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    idx_to_char = {i: c for i, c in enumerate(unique_chars)}

    num_classes = len(char_to_idx)
    print(f"[CharMap] Total classes (incl. blank): {num_classes}")
    return char_to_idx, idx_to_char, num_classes


def encode_text(text: str, char_to_idx: dict) -> list:
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
    return indices


def collate_fn(batch, char_to_idx: dict, max_len: int = MAX_LEN):
    features_list = []
    target_list = []
    target_lengths_list = []

    for features, text in batch:
        features_list.append(features)
        encoded = encode_text(text, char_to_idx)
        target_list.extend(encoded)
        target_lengths_list.append(len(encoded))

    features = torch.stack(features_list, dim=0)
    targets = torch.tensor(target_list, dtype=torch.long)
    input_lengths = torch.full((len(features_list),), max_len, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)

    return features, targets, input_lengths, target_lengths


# ================================================================
# 2. HÀM HUẤN LUYỆN MỘT EPOCH
# ================================================================
def train_one_epoch(model, dataloader, ctc_loss_fn, optimizer, device, char_to_idx):
    model.train()
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

        input_lengths = torch.full((features.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
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
        log_probs = log_probs.permute(1, 0, 2)

        input_lengths = torch.full((features.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ================================================================
# 4. WARMUP + COSINE ANNEALING SCHEDULER
# ================================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_factor=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ================================================================
# 5. HÀM CHÍNH – MAIN
# ================================================================
def main():
    # ==================== CẤU HÌNH HUẤN LUYỆN ====================
    DATA_PATH = "data/vivos"
    BATCH_SIZE = 128                    # Phù hợp với 16GB VRAM của Colab T4
    NUM_EPOCHS = 100                   # Tăng số epoch
    LEARNING_RATE = 1e-3               # LR khởi tạo
    HIDDEN_DIM = 512                   # Tăng số hidden dim cho mô hình sâu hơn
    NUM_LAYERS = 3                     # Tăng số lớp cho Colab T4
    DROPOUT = 0.3                      # Droput vừa phải
    WEIGHT_DECAY = 1e-4                # Weight decay cho AdamW
    WARMUP_EPOCHS = 5                  # Số epoch warmup
    USE_DELTA = True                   # 13 MFCC → 39 features
    SPEED_PERTURB = True               # 0.9x, 1.0x, 1.1x → dataset ×3
    AUGMENT = True                     # Volume augmentation
    INPUT_DIM = 39 if USE_DELTA else 13
    SAVE_DIR = "saved_models"
    MODEL_PATH = os.path.join(SAVE_DIR, "model.pth")
    MAX_SEQ_LEN = MAX_LEN             # 300 (tăng từ 200)

    # ==================== Thiết bị ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    print(f"[Config] BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}")
    print(f"[Config] HIDDEN_DIM={HIDDEN_DIM}, NUM_LAYERS={NUM_LAYERS}, DROPOUT={DROPOUT}")
    print(f"[Config] USE_DELTA={USE_DELTA}, SPEED_PERTURB={SPEED_PERTURB}, INPUT_DIM={INPUT_DIM}")
    print(f"[Config] MAX_SEQ_LEN={MAX_SEQ_LEN}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==================== Bước 1: Từ điển ký tự ====================
    char_to_idx, idx_to_char, num_classes = build_char_map(VIETNAMESE_CHARS)

    # ==================== Bước 2: Tải dữ liệu ====================
    print("\n[Data] Loading VIVOS dataset...")
    train_data = build_dataset(DATA_PATH, "train")
    test_data = build_dataset(DATA_PATH, "test")
    print(f"[Data] Train samples (original): {len(train_data)}")
    print(f"[Data] Test samples:        {len(test_data)}")

    # Train dataset: speed perturbation + augmentation
    train_dataset = VivosDataset(
        train_data,
        precompute=True,
        use_delta=USE_DELTA,
        speed_perturb=SPEED_PERTURB,
        augment=AUGMENT,
    )

    # Test dataset: KHÔNG augmentation
    test_dataset = VivosDataset(
        test_data,
        precompute=True,
        use_delta=USE_DELTA,
        speed_perturb=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx, MAX_SEQ_LEN),
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx, MAX_SEQ_LEN),
        pin_memory=True,
        num_workers=1,
    )

    # ==================== Bước 3: Khởi tạo mô hình ====================
    print("\n[Model] Initializing SpeechRecognitionModel...")
    model = SpeechRecognitionModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params:        {total_params:,}")
    print(f"[Model] Trainable params:    {trainable_params:,}")

    # ==================== Bước 4: CTC Loss ====================
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # ==================== Bước 5: Optimizer – AdamW ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ==================== Bước 6: Warmup + Cosine Annealing ====================
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=NUM_EPOCHS,
        min_factor=0.01,
    )

    # ==================== Bước 7: Vòng lặp huấn luyện ====================
    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15  # Tăng patience vì loss dao động hơn với augmentation

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, ctc_loss_fn, optimizer, device, char_to_idx
        )

        val_loss = validate(
            model, test_loader, ctc_loss_fn, device, char_to_idx
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

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
                    "input_dim": INPUT_DIM,
                },
                MODEL_PATH,
            )
            print(f"  >>> Saved best model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stopping] Stopped at epoch {epoch} (no improvement).")
                break

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETED")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
