import torch
import librosa
import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Cấu hình
#   - Tiếng Anh: "facebook/wav2vec2-base-960h" (hoặc large)
#   - Tiếng Việt: "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    """Load Wav2Vec2 model và processor (cached bởi Streamlit)."""
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return processor, model


def predict(audio_path: str) -> str:
    """
    Nhận đường dẫn file WAV, chạy Wav2Vec2 inference và trả về transcript.

    Parameters
    ----------
    audio_path : str
        Đường dẫn đến file âm thanh (WAV, 16kHz mono).

    Returns
    -------
    str
        Nội dung nhận dạng giọng nói.
    """
    processor, model = load_model()

    # Load audio
    speech, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Preprocess + inference
    inputs = processor(
        speech,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()
