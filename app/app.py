import gradio as gr
import time
import random
import os
# from audio_processing.preprocess import process_audio  
# from model.inference import predict_text

# ==========================================
# PHẦN GIẢ LẬP (MOCKING) KÈM HIỆU ỨNG THỜI GIAN THỰC
# ==========================================
def mock_speech_to_text_stream(audio_filepath, speech_type):
    """
    Hàm này dùng 'yield' thay vì 'return' để tạo hiệu ứng chữ hiện ra từng từ (streaming).
    Khi ghép code thật, mô hình RNN của bạn B cũng có thể xuất ra từng token một cách tương tự.
    """
    if audio_filepath is None:
        yield "Vui lòng ghi âm hoặc tải file lên.", "0%", gr.update(visible=False)
        return

    # Giả lập thời gian model bắt đầu phân tích
    time.sleep(0.5) 
    
    if speech_type == "Đơn giọng nói (Single-speaker)":
        full_text = "Xin chào giảng viên, đây là bài kiểm tra nhận diện một người nói. Nhóm chúng em xin trình bày đồ án Deep Learning sử dụng mô hình RNN."
    else:
        full_text = "[Người 1]: Xin chào, nhóm bạn làm đồ án gì?\n[Người 2]: Nhóm mình làm nhận diện giọng nói bằng kiến trúc Mạng nơ-ron hồi quy."
        
    current_text = ""
    words = full_text.split(" ")
    
    # 1. Hiệu ứng Streaming: Hiện từng chữ và cập nhật độ tin cậy liên tục
    for word in words:
        current_text += word + " "
        # Giả lập độ tin cậy dao động từ 88% đến 98%
        confidence = f"{random.uniform(88.5, 98.9):.1f}%" 
        
        # Trả về 3 giá trị tương ứng với 3 output: Text, Độ tin cậy, và Nút Tải file (ẩn)
        yield current_text.strip(), confidence, gr.update(visible=False)
        time.sleep(0.15) # Tốc độ gõ chữ

    # 2. Tạo file .txt để người dùng tải về khi đã nhận diện xong
    file_path = "ket_qua_nhan_dien.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(current_text.strip())
        
    # Khi hoàn thành, hiện nút tải file lên
    final_confidence = f"{random.uniform(94.0, 98.9):.1f}%"
    yield current_text.strip(), final_confidence, gr.update(value=file_path, visible=True)

# ==========================================
# GIAO DIỆN HIỆN ĐẠI (GRADIO 6.x)
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
                # Ô hiển thị độ tin cậy
                confidence_output = gr.Textbox(label="Độ tin cậy (Confidence)", lines=1)
                
            # Nút tải file (Mặc định ẩn, chỉ hiện khi chạy xong)
            download_output = gr.File(label="Tải kết quả", visible=False)

    # BẮT SỰ KIỆN: Kết nối Nút bấm với Hàm giả lập
    submit_btn.click(
        fn=mock_speech_to_text_stream, 
        inputs=[audio_input, speech_type], 
        outputs=[text_output, confidence_output, download_output]
    )
    
    # Reset toàn bộ giao diện
    clear_btn.click(
        lambda: (None, "Đơn giọng nói (Single-speaker)", "", "", gr.update(visible=False)), 
        inputs=None, 
        outputs=[audio_input, speech_type, text_output, confidence_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=custom_theme)