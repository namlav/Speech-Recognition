"""
train.py
========
Refactored using SpeechBrain.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import jiwer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from audio_processing.dataset import build_dataset
from audio_processing.dataloader import VivosDataset
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


def collate_fn(batch, char_to_idx: dict):
    features_list = []
    target_list = []
    target_lengths_list = []
    input_lengths_list = []

    for features, text, length in batch:
        features_list.append(features)
        encoded = encode_text(text, char_to_idx)
        target_list.extend(encoded)
        target_lengths_list.append(len(encoded))
        input_lengths_list.append(length)

    features = torch.stack(features_list, dim=0)
    targets = torch.tensor(target_list, dtype=torch.long)
    # Time length after first conv layer (stride=2 on time axis): floor((T + 1) / 2)
    input_lengths = torch.tensor([(l + 1) // 2 for l in input_lengths_list], dtype=torch.long)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)

    return features, targets, input_lengths, target_lengths


# ================================================================
# 2. SPEECHBRAIN CLASS
# ================================================================
class ASRBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        features, targets, input_lengths, target_lengths = batch
        features = features.to(self.device)
        
        log_probs = self.modules.model(features)
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) for CTCLoss
        return log_probs

    def compute_objectives(self, predictions, batch, stage):
        features, targets, input_lengths, target_lengths = batch
        targets = targets.to(self.device)
        input_lengths = input_lengths.to(self.device)
        target_lengths = target_lengths.to(self.device)

        loss = self.hparams["ctc_loss"](predictions, targets, input_lengths, target_lengths)
        
        # Compute CER on validation/test
        if stage != sb.Stage.TRAIN:
            idx_to_char = self.hparams["idx_to_char"]
            pred_inds = torch.argmax(predictions, dim=-1)  # (T, N)
            
            offset = 0
            for n in range(pred_inds.size(1)):
                # Decode prediction (CTC collapse)
                prev = -1
                pred_chars = []
                for t in range(pred_inds.size(0)):
                    idx = int(pred_inds[t, n].item())
                    if idx == 0:
                        prev = -1
                        continue
                    if idx != prev:
                        char = idx_to_char.get(idx, "")
                        if char and char != "<blank>":
                            pred_chars.append(char)
                    prev = idx
                pred_text = "".join(pred_chars)
                
                # Decode target
                tgt_len = int(target_lengths[n].item())
                tgt_inds = targets[offset:offset + tgt_len].tolist()
                offset += tgt_len
                tgt_text = "".join(idx_to_char.get(i, "") for i in tgt_inds)
                
                self.cer_metrics.append(tgt_text)
                self.cer_metrics.append(pred_text)
        
        return loss

    def on_stage_start(self, stage, epoch):
        if stage == sb.Stage.TRAIN:
            if not hasattr(self, "scheduler_init"):
                self.scheduler = self.hparams["lr_scheduler"](
                    self.optimizer, 
                    self.hparams["warmup_epochs"], 
                    self.hparams["number_of_epochs"]
                )
                self.scheduler_init = True
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = []  # interleaved [ref, hyp, ref, hyp, ...]

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.VALID:
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            
            # Compute CER
            cer_str = "N/A"
            if hasattr(self, "cer_metrics") and len(self.cer_metrics) >= 2:
                refs = self.cer_metrics[0::2]
                hyps = self.cer_metrics[1::2]
                cer_val = jiwer.cer(refs, hyps)
                cer_str = f"{cer_val:.2%}"
            
            print(f"Epoch {epoch} | Val Loss: {stage_loss:.4f} | CER: {cer_str} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model based on validation loss
            if not hasattr(self, "best_val_loss") or stage_loss < self.best_val_loss:
                self.best_val_loss = stage_loss
                model_path = os.path.join("saved_models", "model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.modules.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": stage_loss,
                        "cer": cer_val if cer_str != "N/A" else None,
                        "char_to_idx": self.hparams["char_to_idx"],
                        "idx_to_char": self.hparams["idx_to_char"],
                        "num_classes": self.hparams["num_classes"],
                        "hidden_dim": self.hparams["hidden_dim"],
                        "num_layers": self.hparams["num_layers"],
                        "dropout": self.hparams["dropout"],
                        "input_dim": self.hparams["input_dim"],
                    },
                    model_path,
                )
                print(f"  >>> Saved best model (Val Loss: {stage_loss:.4f}, CER: {cer_str})")

# ================================================================
# 3. HÀM CHÍNH – MAIN
# ================================================================
def main():
    # ==================== CẤU HÌNH HUẤN LUYỆN ====================
    DATA_PATH = "data/vivos"
    BATCH_SIZE = 64                    # Phù hợp với 16GB VRAM của Colab T4
    NUM_EPOCHS = 100                   # Tăng số epoch
    LEARNING_RATE = 3e-4               # Lower LR for stability
    HIDDEN_DIM = 512                   # Tăng số hidden dim cho mô hình sâu hơn
    NUM_LAYERS = 3                     # Tăng số lớp cho Colab T4
    DROPOUT = 0.4                      # Droput vừa phải
    WEIGHT_DECAY = 1e-4                # Weight decay cho AdamW
    WARMUP_EPOCHS = 5                  # Số epoch warmup
    USE_DELTA = True                   # 13 MFCC → 39 features
    SPEED_PERTURB = True               # 0.9x, 1.0x, 1.1x → dataset ×3
    AUGMENT = True                     # Volume augmentation
    INPUT_DIM = 39 if USE_DELTA else 13
    SAVE_DIR = "saved_models"

    # ==================== Thiết bị ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[Device] Using: {device}")
    print(f"[Config] BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}")
    print(f"[Config] HIDDEN_DIM={HIDDEN_DIM}, NUM_LAYERS={NUM_LAYERS}, DROPOUT={DROPOUT}")
    print(f"[Config] USE_DELTA={USE_DELTA}, SPEED_PERTURB={SPEED_PERTURB}, INPUT_DIM={INPUT_DIM}")

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

    # Tự động lấy số luồng CPU (Colab thường có 2)
    num_cpus = os.cpu_count() or 2
    workers_train = min(4, num_cpus)
    workers_test = min(2, num_cpus)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx),
        pin_memory=True,
        num_workers=workers_train,
        persistent_workers=(workers_train > 0),
        prefetch_factor=2 if workers_train > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, char_to_idx),
        pin_memory=True,
        num_workers=workers_test,
        persistent_workers=(workers_test > 0),
        prefetch_factor=2 if workers_test > 0 else None,
    )

    # ==================== Bước 3: Đọc Hparams & Khởi tạo ====================
    print("\n[Settings] Reading hparams...")
    hparams_file = "hparams.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    # Set seed for full reproducibility (torch set in YAML, add numpy/cuda here)
    np.random.seed(hparams["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hparams["seed"])
        torch.backends.cudnn.deterministic = True
    
    print("\n[Model] Initializing SpeechRecognitionModel...")
    model = SpeechRecognitionModel(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        num_classes=num_classes,
        dropout=hparams["dropout"],
    ).to(device)
    
    # Loss
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    hparams["model"] = model
    hparams["ctc_loss"] = ctc_loss_fn
    hparams["epoch_counter"] = sb.utils.epoch_loop.EpochCounter(limit=hparams["number_of_epochs"])
    hparams["char_to_idx"] = char_to_idx
    hparams["idx_to_char"] = idx_to_char
    hparams["num_classes"] = num_classes
    
    # Pass lambda as opt_class for Speechbrain
    opt_class = lambda x: torch.optim.AdamW(
        x, 
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )
    
    def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_factor=0.01):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    hparams["lr_scheduler"] = get_warmup_cosine_scheduler
    
    # Initialize Brain
    asr_brain = ASRBrain(
        modules={"model": model},
        opt_class=opt_class,
        hparams=hparams,
        run_opts={"device": device, "auto_mix_prec": (device.type == "cuda")},
        checkpointer=sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=SAVE_DIR,
            recoverables={"model": model, "counter": hparams["epoch_counter"]}
        ),
    )
    
    # ==================== Bước 4: Vòng lặp huấn luyện ====================
    print("\n" + "=" * 60)
    print("START TRAINING (SPEECHBRAIN)")
    print("=" * 60)

    asr_brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_loader,
        valid_set=test_loader,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

