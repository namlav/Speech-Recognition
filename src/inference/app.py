import io
import os
import traceback
import time

_GRADIO_TEMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gradio_temp")
os.makedirs(_GRADIO_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(_GRADIO_TEMP)

FFMPEG_PATHS = [
    r"C:\Users\lavan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin",
]
for _p in FFMPEG_PATHS:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional as aF
import torchaudio.transforms as T

from src.config import DEVICE, SAMPLING_RATE, N_MELS, TOP_DB
from src.inference.predict import load_model


model, tokenizer = load_model()
model.eval()

ERROR_LOG_PATH = "app_error.log"


def log_error(msg):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def to_float_audio(audio_data):
    if audio_data.dtype in (np.int16, np.int32, np.int64):
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    elif audio_data.dtype == np.float64:
        audio_data = audio_data.astype(np.float32)
    return audio_data


def get_num_samples(audio_data):
    return audio_data.shape[0]


def to_mono(audio_data):
    if audio_data.ndim == 1:
        return audio_data
    return np.mean(audio_data, axis=1)


def draw_plots(audio_data, sr):
    audio_data = to_float_audio(audio_data)
    audio_data = to_mono(audio_data)

    fig_wave = plt.figure(figsize=(8, 3))
    librosa.display.waveshow(audio_data, sr=sr, color="#3498db")
    plt.title("Biểu đồ sóng âm (Waveform)", fontsize=10)
    plt.xlabel("Thời gian (s)", fontsize=8)
    plt.ylabel("Biên độ", fontsize=8)
    plt.tight_layout()

    fig_spec = plt.figure(figsize=(8, 3))
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.title("Biểu đồ phổ Mel Spectrogram", fontsize=10)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    return fig_wave, fig_spec


def audio_upload_handler(audio):
    if audio is None:
        return None, None
    try:
        sr, audio_data = audio
        if get_num_samples(audio_data) < 256:
            return None, None
        fig_wave, fig_spec = draw_plots(audio_data, sr)
        return fig_wave, fig_spec
    except Exception as e:
        log_error(f"audio_upload_handler error: {traceback.format_exc()}")
        return None, None


def pipeline_giao_dien_moi(audio):
    if audio is None:
        return "Vui lòng cung cấp file âm thanh!", "", None, None

    try:
        start_time = time.time()

        orig_sr, audio_data = audio
        audio_data = to_float_audio(audio_data)

        if get_num_samples(audio_data) < 256:
            return "File âm thanh quá ngắn hoặc không đọc được!", "", None, None

        audio_mono = to_mono(audio_data)
        audio_tensor = torch.from_numpy(audio_mono).float().unsqueeze(0)

        if orig_sr != SAMPLING_RATE:
            audio_tensor = aF.resample(audio_tensor, orig_freq=orig_sr, new_freq=SAMPLING_RATE)

        if audio_tensor.shape[-1] < 256:
            return "Audio quá ngắn sau khi xử lý!", "", None, None

        audio_tensor = aF.highpass_biquad(audio_tensor, SAMPLING_RATE, cutoff_freq=85.0)
        audio_tensor = aF.lowpass_biquad(audio_tensor, SAMPLING_RATE, cutoff_freq=4000.0)

        audio2mels = T.MelSpectrogram(sample_rate=SAMPLING_RATE, n_mels=N_MELS)
        amp2db = T.AmplitudeToDB(top_db=TOP_DB)

        mel = audio2mels(audio_tensor)
        mel = amp2db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0)
        src_len = torch.tensor([mel.shape[-1]])

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_logits, _ = model(mel.to(DEVICE), src_len)

        pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist()
        pred_transcript = tokenizer.decode(pred_tokens)
        text_result = pred_transcript.lower()

        process_time = time.time() - start_time
        duration = audio_tensor.shape[-1] / SAMPLING_RATE

        info_md = f"""
        ### 📊 Chi tiết quá trình xử lý:
        * **Độ dài đoạn ghi âm:** `{duration:.2f} giây`
        * **Thời gian suy luận (Latency):** `{process_time:.2f} giây`
        * **Tốc độ xử lý (Real-time factor):** `{process_time/duration:.2f}x`
        """

        fig_wave, fig_spec = draw_plots(audio_data, orig_sr)

        return text_result, info_md, fig_wave, fig_spec

    except Exception as e:
        log_error(traceback.format_exc())
        return f"❌ Lỗi hệ thống: {str(e)}", "", None, None


def launch():
    with gr.Blocks(title="Vietnamese Speech-to-Text") as demo:

        gr.Markdown(
            """
            <h1 style='text-align: center; color: #2c3e50;'>🎙️ HỆ THỐNG NHẬN DIỆN GIỌNG NÓI TIẾNG VIỆT</h1>
            <p style='text-align: center; font-size: 15px; color: #7f8c8d;'>Kiến trúc: DeepSpeech2 Custom Trained (LSVSC Dataset)</p>
            <hr>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 1. Đầu vào âm thanh")
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Ghi âm từ Mic hoặc Tải file âm thanh lên",
                )
                submit_btn = gr.Button("🚀 BẮT ĐẦU NHẬN DIỆN GIỌNG NÓI", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📝 2. Kết quả nhận diện văn bản")
                text_out = gr.Textbox(
                    label="Văn bản tiếng Việt dịch được (Predicted Transcript):",
                    lines=4,
                    placeholder="Kết quả dịch...",
                    buttons=["copy"],
                )

                gr.Markdown("### ⚙️ 3. Phân tích chi tiết hiệu năng")
                info_out = gr.Markdown("*(Đang chờ bạn nạp file âm thanh để phân tích...)*")

        gr.Markdown("### 📈 4. Trực quan hóa đặc trưng tín hiệu âm thanh thực tế")
        with gr.Row():
            wave_out = gr.Plot(label="Biểu đồ sóng âm (Waveform)")
            spec_out = gr.Plot(label="Biểu đồ phổ Mel Spectrogram (80 kênh)")

        audio_in.change(
            fn=audio_upload_handler,
            inputs=[audio_in],
            outputs=[wave_out, spec_out],
        )

        submit_btn.click(
            fn=pipeline_giao_dien_moi,
            inputs=[audio_in],
            outputs=[text_out, info_out, wave_out, spec_out],
        )

    demo.launch(share=True, debug=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()
