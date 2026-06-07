# Speech-Recognition

Vietnamese Speech-to-Text sử dụng kiến trúc **DeepSpeech2** huấn luyện trên dataset **LSVSC**.

## Kiến trúc mô hình (Model Architecture)

DeepSpeech2 gồm 3 khối chính:

```
Audio Input (raw waveform)
        │
        ▼
┌───────────────────────────────┐
│   ConvolutionFeatureExtractor │  ◄── 2× MaskedConv2D + BatchNorm + Hardtanh
│   (Mel Spectrogram → Features)│
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│   5× RNNLayer (Bidirectional  │  ◄── GRU + LayerNorm
│   GRU, hidden=512)            │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│   Classification Head         │  ◄── Linear(1024→512) + Hardtanh + Linear(512→vocab)
│   (Token prediction)          │
└───────────┬───────────────────┘
            │
            ▼
      CTC Decoding
            │
            ▼
      Transcript text
```

- **Feature Extractor**: 2 lớp `MaskedConv2d` tự động zero-out vùng padding để không ảnh hưởng đến feature map, sau đó reshape về sequence (time × features).
- **RNN Layers**: Stack 5 GRU bidirectional, mỗi tầng có `pack_padded_sequence` để bỏ qua padding khi tính toán.
- **CTC Loss**: Huấn luyyện với Connectionist Temporal Classification — không cần alignment frame-level giữa audio và text.
- **SpecAugment**: Frequency masking + Time masking áp dụng ngẫu nhiên trên Mel spectrogram khi training.

## Cấu trúc thư mục (Directory Structure)

```
Speech-Recognition/
├── src/                          # Mã nguồn chính (refactored từ fine-tuning.ipynb)
│   ├── config.py                 # Hyperparameters tập trung
│   ├── data/
│   │   ├── dataset.py            # LSVSCDataset: load audio → Mel spec → Augment
│   │   └── collate.py            # collate_fn: padding, packing cho CTC loss
│   ├── models/
│   │   ├── components.py         # MaskedConv2d, ConvolutionFeatureExtractor, RNNLayer
│   │   └── deepspeech2.py        # DeepSpeech2 model
│   ├── training/
│   │   └── trainer.py            # Training loop (train/eval, AMP, scheduler, save best)
│   ├── inference/
│   │   ├── predict.py            # inference() cho single sample
│   │   └── app.py                # Web UI với Gradio
│   └── utils/
│       └── visualization.py      # Vẽ loss curves
│
├── data/                         # Thư mục chứa dữ liệu âm thanh thô
├── weights/
│   └── best_weights.pt           # Weights đã train (DeepSpeech2)
│
├── fine-tuning.ipynb             # Notebook huấn luyện gốc
├── requirements.txt
└── README.md
```

## Hướng dẫn chạy (How to Run)

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện (Training)

```bash
python -m src.training.trainer
```

Hoặc mở `fine-tuning.ipynb` trên Colab/Kaggle với GPU.

### 3. Chạy Gradio app (Run app to use)

```bash
python -m src.inference.app
```

### 4. Inference thủ công

```python
from src.inference.predict import load_model, inference

model, tokenizer = load_model()
result = inference(audio_array, orig_sr=16000, model=model, tokenizer=tokenizer)
print(result)
```

### 5. Đánh giá WER

Xem cell cuối trong `fine-tuning.ipynb` hoặc dùng:

```python
from jiwer import wer
from src.inference.predict import load_model, inference
from datasets import load_dataset

model, tokenizer = load_model()
testset = load_dataset("doof-ferb/LSVSC")["test"]

errors = []
for item in testset:
    pred = inference(item["audio"]["array"], item["audio"]["sampling_rate"], model, tokenizer)
    errors.append(wer(item["transcription"].lower(), pred.lower()))

print("Mean WER:", sum(errors) / len(errors))
```

## Dataset

**LSVSC** (Large-Scale Vietnamese Speech Corpus): tải qua HuggingFace `doof-ferb/LSVSC`.
