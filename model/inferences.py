"""
model/inferences.py
===================
Module dự đoán (Inference) cho bài toán Nhận diện Giọng nói.

Cung cấp hàm predict(audio_path) để file app.py import và sử dụng:
    from model.inferences import predict
    text = predict("path/to/audio.wav")

Quy trình:
    1. Load trọng số mô hình đã huấn luyện (saved_models/model.pth).
    2. Gọi preprocess(audio_path) để trích xuất đặc trưng MFCC (200, 13).
    3. Đưa tensor qua mô hình → log-probabilities (200, num_classes).
    4. Giải mã bằng thuật toán Greedy Decoder cho CTC → chuỗi văn bản.
    5. Trả về chuỗi văn bản tiếng Việt.
"""

import os
import sys
import torch
import numpy as np

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_processing.preprocess import preprocess, MAX_LEN
from model.network import SpeechRecognitionModel


# ================================================================
# ĐƯỜNG DẪN MẶC ĐỊNH ĐẾN FILE TRỌNG SỐ
# ================================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "saved_models", "model.pth"
)

# Cache toàn cục: chỉ load model một lần, tránh load lại mỗi lần gọi predict
_model = None
_idx_to_char = None
_device = None


def _load_model(model_path: str = None):
    """
    Load mô hình và ánh xạ ký tự từ checkpoint đã lưu.
    Sử dụng biến toàn cục để cache, chỉ load một lần duy nhất.

    Tham số:
        model_path: Đường dẫn tới file .pth (mặc định: saved_models/model.pth)

    Trả về:
        model:       PyTorch model đã load trọng số, ở chế độ eval().
        idx_to_char: Dict {index: char} để decode.
        device:      Thiết bị đang chạy (cpu/cuda).
    """
    global _model, _idx_to_char, _device

    if _model is not None:
        return _model, _idx_to_char, _device

    if model_path is None:
        model_path = MODEL_PATH

    # Chuẩn hóa đường dẫn
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy file model tại: {model_path}\n"
            f"Hãy chạy train.py trước để huấn luyện và lưu mô hình."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint ----
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)

    # Lấy các tham số từ checkpoint
    num_classes = checkpoint["num_classes"]
    hidden_dim = checkpoint.get("hidden_dim", 256)
    num_layers = checkpoint.get("num_layers", 3)
    dropout = checkpoint.get("dropout", 0.3)
    input_dim = checkpoint.get("input_dim", 13)
    _idx_to_char = checkpoint["idx_to_char"]

    # ---- Khởi tạo lại kiến trúc và load trọng số ----
    _model = SpeechRecognitionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(_device)

    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()  # Chế độ đánh giá (tắt dropout, batch norm)

    print(f"[Inference] Loaded model from: {model_path}")
    print(f"[Inference] Device: {_device}, Classes: {num_classes}")

    return _model, _idx_to_char, _device


# ================================================================
# THUẬT TOÁN GREEDY DECODER CHO CTC
# ================================================================
def greedy_decode(log_probs: torch.Tensor, idx_to_char: dict) -> str:
    """
    Giải mã đầu ra CTC bằng thuật toán Greedy Decoder.

    Thuật toán:
        1. Tại mỗi bước thời gian t, chọn ký tự có xác suất cao nhất (argmax).
        2. Nén các ký tự trùng lặp liên tiếp (CTC collapsing):
           - Ví dụ: "a a a" → "a"
        3. Loại bỏ blank token (index 0):
           - Blank token đại diện cho "không có ký tự" tại bước thời gian đó.

    Tham số:
        log_probs:   Tensor shape (T, num_classes) – log-probabilities.
        idx_to_char: Dict {index: char} để ánh xạ index → ký tự.

    Trả về:
        Chuỗi văn bản đã giải mã.
    """
    # Bước 1: Lấy index có xác suất cao nhất tại mỗi bước thời gian
    # argmax theo dim=1 (class) → shape (T,)
    best_indices = torch.argmax(log_probs, dim=1).cpu().numpy()

    # Bước 2 & 3: Collapse repeated chars & remove blanks
    decoded_chars = []
    prev_idx = -1  # Index trước đó (khởi tạo khác mọi index hợp lệ)

    for idx in best_indices:
        idx = int(idx)
        # Bỏ qua blank token (index 0)
        if idx == 0:
            prev_idx = -1
            continue
        # Chỉ thêm nếu khác với ký tự liền trước (CTC collapsing)
        if idx != prev_idx:
            char = idx_to_char.get(idx, "")
            if char and char != "<blank>":
                decoded_chars.append(char)
        prev_idx = idx

    return "".join(decoded_chars)


# ================================================================
# HÀM DỰ ĐOÁN CHÍNH – predict()
# ================================================================
def predict(audio_path: str) -> str:
    """
    Dự đoán văn bản từ file âm thanh .wav.

    Hàm này được import trực tiếp bởi app.py:
        from model.inferences import predict
        text = predict("path/to/audio.wav")

    Quy trình xử lý:
        1. Load mô hình đã huấn luyện (cache lại sau lần load đầu).
        2. Trích xuất đặc trưng MFCC từ file âm thanh (gọi preprocess).
        3. Chạy inference qua mô hình.
        4. Giải mã CTC Greedy → chuỗi văn bản.

    Tham số:
        audio_path: Đường dẫn tới file âm thanh .wav.

    Trả về:
        Chuỗi văn bản tiếng Việt đã nhận diện.
    """
    # ---- Bước 1: Load mô hình (cache) ----
    model, idx_to_char, device = _load_model()

    # ---- Bước 2: Trích xuất đặc trưng MFCC ----
    # Tự động chọn use_delta dựa trên input_dim của model
    input_dim = model.input_dim
    use_delta = (input_dim == 39)

    # preprocess() trả về numpy array shape (MAX_LEN, input_dim)
    features = preprocess(audio_path, use_delta=use_delta, max_len=MAX_LEN)

    # Chuyển thành tensor và thêm batch dimension
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # ---- Bước 3: Inference ----
    with torch.no_grad():
        # log_probs: (1, 200, num_classes)
        log_probs = model(features_tensor)

        # Bỏ batch dimension: (1, 200, num_classes) → (200, num_classes)
        log_probs = log_probs.squeeze(0)

    # ---- Bước 4: Giải mã CTC Greedy ----
    text = greedy_decode(log_probs, idx_to_char)

    return text


# ================================================================
# CHẠY THỬ NGHIỆM
# ================================================================
if __name__ == "__main__":
    # Test inference với một file âm thanh mẫu (nếu có)
    test_audio = "data/vivos/test/waves/VIVOSSPK01/VIVOSSPK01_R001.wav"

    if os.path.exists(test_audio):
        result = predict(test_audio)
        print(f"Recognition result: \"{result}\"")
    else:
        print(f"Test file not found: {test_audio}")
        print("Run train.py first and check the dataset path.")
