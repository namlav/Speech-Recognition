import streamlit as st
import numpy as np
import tempfile
import sys, os

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_processing.preprocess import preprocess
from model.inference import predict

st.title("🎤 Speech Recognition Demo")

mode = st.radio("Chọn chế độ:", ["Upload file", "Realtime Mic"])

# =========================
# 📂 FILE MODE
# =========================
if mode == "Upload file":
    uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        st.audio(path)

        if st.button("Transcribe File"):
            features = preprocess(path)
            text = predict(features)
            st.success(text)

# =========================
# 🎤 REALTIME MODE
# =========================
else:
    st.write("🎙️ Nói vào mic...")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.buffer = []

        def recv(self, frame):
            audio = frame.to_ndarray().flatten()
            self.buffer.extend(audio.tolist())

            # xử lý mỗi ~2s audio
            if len(self.buffer) > 32000:
                data = np.array(self.buffer)
                self.buffer = []

                # save temp
                temp_path = "temp_realtime.wav"
                import soundfile as sf

                sf.write(temp_path, data, 16000)

                features = preprocess(temp_path)
                text = predict(features)

                print("Realtime:", text)

            return frame

    webrtc_streamer(
        key="speech",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
