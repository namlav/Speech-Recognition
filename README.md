# Speech-Recognition

Vietnamese speech recognition using DeepSpeech 2 + CTC + SpeechBrain.

## Structure

```text
speech_recognition_project/
│
├── train.py                 # Training pipeline (SpeechBrain)
├── test_pipeline.py         # Quick data pipeline test
├── hparams.yaml             # Hyperparameter config
├── requirements.txt
├── README.md
│
├── data/
│   └── vivos/               # VIVOS Vietnamese speech dataset
│       ├── train/
│       └── test/
│
├── audio_processing/
│   ├── dataset.py           # Build dataset from VIVOS metadata
│   ├── dataloader.py        # PyTorch Dataset (precompute, speed perturb, aug)
│   ├── preprocess.py        # MFCC extraction, padding, augmentation
│   └── visualize.py         # Waveform/MFCC plotting
│
├── model/
│   ├── network.py           # DeepSpeech 2 model (CNN + GRU + CTC)
│   ├── specaugment.py       # SpecAugment (frequency/time masking)
│   └── inferences.py        # Inference: predict(audio_path) -> text
│
├── app/
│   └── app.py               # Streamlit web app (upload + realtime mic)
│
└── saved_models/
    └── model.pth            # Trained weights (gitignored)
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py              # Train model
python -m streamlit run app/app.py  # Launch web demo
```
