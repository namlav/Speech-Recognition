import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import gradio as gr
import time
import random
from audio_processing.preprocess import preprocess
from audio_processing.visualize import plot_waveform, plot_mfcc

# ==========================================
# PHẦN GIẢ LẬP KÈM HIỆU ỨNG THỜI GIAN THỰC & VẼ BIỂU ĐỒ
# ==========================================
def mock_speech_to_text_stream(audio_filepath, speech_type):
    if audio_filepath is None:
        # Nếu không có file, trả về 5 giá trị rỗng/None tương ứng với 5 Output
        yield "Vui lòng ghi âm hoặc tải file lên.", "0%", gr.update(visible=False), None, None
        return

    # Giả lập thời gian model bắt đầu phân tích
    time.sleep(0.5) 
    
    if speech_type == "Đơn giọng nói (Single-speaker)":
        full_text = "Xin chào giảng viên, đây là bài kiểm tra nhận diện một người nói. Nhóm chúng em xin trình bày đồ án Deep Learning sử dụng mô hình RNN."
    else:
        full_text = "[Người 1]: Xin chào, nhóm bạn làm đồ án gì?\n[Người 2]: Nhóm mình làm nhận diện giọng nói bằng kiến trúc Mạng nơ-ron hồi quy."
        
    current_text = ""
    words = full_text.split(" ")
    
    # 1. Hiệu ứng Streaming: Chữ chạy ra từ từ
    for word in words:
        current_text += word + " "
        confidence = f"{random.uniform(88.5, 98.9):.1f}%" 
        
        # TRONG LÚC CHỮ ĐANG CHẠY: Chưa hiện biểu đồ (trả về None, None)
        yield current_text.strip(), confidence, gr.update(visible=False), None, None
        time.sleep(0.15) 

    # 2. Xử lý lưu file text kết quả
    file_path = "ket_qua_nhan_dien.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(current_text.strip())
        
    # 3. GỌI CODE CỦA PERSON A ĐỂ VẼ BIỂU ĐỒ
    try:
        fig_wave = plot_waveform(audio_filepath)
        fig_mfcc = plot_mfcc(audio_filepath)
    except Exception as e:
        print("Lỗi vẽ biểu đồ:", e)
        fig_wave, fig_mfcc = None, None

    # KHI HOÀN THÀNH: Trả về text cuối, độ tin cậy, nút tải file, VÀ 2 BIỂU ĐỒ
    final_confidence = f"{random.uniform(94.0, 98.9):.1f}%"
    yield current_text.strip(), final_confidence, gr.update(value=file_path, visible=True), fig_wave, fig_mfcc

# ==========================================
# GIAO DIỆN HIỆN ĐẠI
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

with gr.Blocks(title="AI Speech Recognition") as demo:
    
    gr.Markdown("<h1 style='text-align: center;'>🎙️ Hệ Thống Nhận Diện Giọng Nói</h1>")
    gr.Markdown("<p style='text-align: center;'><b>Đồ án Deep Learning</b> | <i>Sử dụng kiến trúc RNN</i></p>")
    
    with gr.Row():
        # CỘT TRÁI: INPUT
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Ghi âm hoặc Tải lên (.wav)",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#4f46e5", 
                    waveform_progress_color="#10b981"
                )
            )
            
            speech_type = gr.Radio(
                choices=["Đơn giọng nói (Single-speaker)", "Đa giọng nói (Multi-speaker)"],
                value="Đơn giọng nói (Single-speaker)",
                label="Chế độ nhận diện"
            )
            
            with gr.Row():
                submit_btn = gr.Button("Bắt đầu nhận diện", variant="primary")
                clear_btn = gr.Button("Làm mới", variant="secondary")
                
        # CỘT PHẢI: OUTPUT
        with gr.Column(scale=1):
            text_output = gr.Textbox(
                label="Văn bản nhận diện (Real-time)", 
                lines=5, 
                placeholder="Chữ sẽ hiện ra ở đây..."
            )
            
            with gr.Row():
                confidence_output = gr.Textbox(label="Độ tin cậy (Confidence)", lines=1)
                download_output = gr.File(label="Tải kết quả", visible=False)
                
    # ĐƯA BIỂU ĐỒ XUỐNG DƯỚI CÙNG ĐỂ RỘNG RÃI
    with gr.Accordion("Xem chi tiết phân tích âm thanh", open=True):
        with gr.Row():
            # Chia cột nhỏ để dễ quản lý tiêu đề cho từng hình
            with gr.Column():
                # Tự tạo tiêu đề bằng Markdown
                gr.Markdown("<h3 style='text-align: center; color: #4f46e5;'>Biểu đồ Sóng âm</h3>")
                waveform_plot = gr.Plot(show_label=False)
                
            with gr.Column():
                gr.Markdown("<h3 style='text-align: center; color: #4f46e5;'>Đặc trưng MFCC (Nạp vào Model)</h3>")
                mfcc_plot = gr.Plot(show_label=False)

    submit_btn.click(
        fn=mock_speech_to_text_stream, 
        inputs=[audio_input, speech_type], 
        outputs=[text_output, confidence_output, download_output, waveform_plot, mfcc_plot]
    )
    
    clear_btn.click(
        lambda: (None, "Đơn giọng nói (Single-speaker)", "", "", gr.update(visible=False), None, None), 
        inputs=None, 
        outputs=[audio_input, speech_type, text_output, confidence_output, download_output, waveform_plot, mfcc_plot]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=custom_theme)