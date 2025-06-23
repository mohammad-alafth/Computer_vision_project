import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import tempfile
import time
import re

# === LOAD MODEL ===
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("✅ Model berhasil dimuat.")
print(f"Classes: {model.names}")

# === EXTRACT NOMINAL DARI NAMA KELAS ===
def extract_nominal(class_name):
    match = re.search(r'coin_(\d+)', class_name)
    return int(match.group(1)) if match else 0

COIN_VALUES = {name: extract_nominal(name) for name in model.names.values()}

def format_currency(amount):
    return f"Rp {amount:,}".replace(",", ".")

# === DETEKSI GAMBAR ===
def detect_image(image):
    if image is None:
        return None, "Tidak ada gambar yang diupload"

    img_array = np.array(image)
    results = model(img_array)[0]
    annotated = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    class_counts = {}
    total_nominal = 0

    if hasattr(results, 'boxes') and results.boxes is not None:
        classes = results.boxes.cls.cpu().numpy()
        for cls in classes:
            class_name = model.names[int(cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            nominal = COIN_VALUES.get(class_name, 0)
            total_nominal += nominal * count

    summary = f"🔴 {sum(class_counts.values())} objek terdeteksi\n"
    for class_name, count in class_counts.items():
        summary += f"- {class_name}: {count}\n"
    summary += f"\n💰 Total: {format_currency(total_nominal)}"

    return annotated_rgb, summary

# === DETEKSI VIDEO ===
def detect_video(video_path):
    if video_path is None:
        return None, "Tidak ada video yang diupload"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Gagal membuka video"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    class_counts = {}
    total_nominal = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        out.write(results.plot())

        if hasattr(results, 'boxes') and results.boxes is not None:
            classes = results.boxes.cls.cpu().numpy()
            for cls in classes:
                class_name = model.names[int(cls)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_nominal += COIN_VALUES.get(class_name, 0)

    cap.release()
    out.release()

    summary = f"🎞️ Total objek: {sum(class_counts.values())}\n"
    for class_name, count in class_counts.items():
        summary += f"- {class_name}: {count}\n"
    summary += f"\n💰 Total: {format_currency(total_nominal)}"

    return temp_output.name, summary

# === WEBCAM STREAM ===
def webcam_inference(image):
    if image is None:
        return None
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    results = model(img_array)[0]
    annotated = results.plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def get_detection_summary(image):
    if image is None:
        return "Tidak ada frame"
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    results = model(img_array)[0]

    class_counts = {}
    total_nominal = 0

    if hasattr(results, 'boxes') and results.boxes is not None:
        classes = results.boxes.cls.cpu().numpy()
        for cls in classes:
            class_name = model.names[int(cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in class_counts.items():
            total_nominal += COIN_VALUES.get(class_name, 0) * count

    summary = f"🔴 {sum(class_counts.values())} objek terdeteksi\n"
    for class_name, count in class_counts.items():
        summary += f"- {class_name}: {count}\n"
    summary += f"\n💰 Total: {format_currency(total_nominal)}\n⏰ {time.strftime('%H:%M:%S')}"
    return summary

def detect_from_webcam_frame(image):
    if image is not None:
        return detect_image(Image.fromarray(image))
    return None, "Tidak ada frame"

# === CSS UNTUK RAPIH ===
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
"""

# === GRADIO UI ===
with gr.Blocks(css=custom_css, title="Deteksi Koin Rupiah") as app:
    gr.Markdown("# 💰 Deteksi Koin Rupiah Otomatis dengan YOLOv8")

    with gr.Tabs():
        with gr.TabItem("📷 Gambar"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Gambar")
                    detect_btn = gr.Button("🔍 Deteksi")
                with gr.Column():
                    image_output = gr.Image(label="Hasil Deteksi")
                    summary_output = gr.Textbox(lines=8, label="Ringkasan")
            detect_btn.click(fn=detect_image, inputs=image_input, outputs=[image_output, summary_output])

        with gr.TabItem("🎥 Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    process_btn = gr.Button("🎬 Proses")
                with gr.Column():
                    video_output = gr.Video(label="Hasil Deteksi")
                    video_summary = gr.Textbox(lines=8, label="Ringkasan")
            process_btn.click(fn=detect_video, inputs=video_input, outputs=[video_output, video_summary])

        with gr.TabItem("📹 Real-time"):
            with gr.Row():
                with gr.Column():
                    webcam_live = gr.Interface(fn=webcam_inference,
                        inputs=gr.Image(source="webcam", streaming=True),
                        outputs=gr.Image(label="Output Real-time"),
                        live=True)
                with gr.Column():
                    webcam_summary = gr.Interface(fn=get_detection_summary,
                        inputs=gr.Image(source="webcam", streaming=True),
                        outputs=gr.Textbox(lines=10, label="Ringkasan Deteksi"),
                        live=True)

        with gr.TabItem("📸 Manual Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam_input_manual = gr.Image(source="webcam", streaming=True, label="Stream Webcam")
                    detect_webcam_btn = gr.Button("📸 Ambil & Deteksi")
                with gr.Column():
                    webcam_result_img = gr.Image(label="Hasil")
                    webcam_result_text = gr.Textbox(lines=8, label="Ringkasan")
            detect_webcam_btn.click(fn=detect_from_webcam_frame, inputs=webcam_input_manual, outputs=[webcam_result_img, webcam_result_text])

if __name__ == '__main__':
    app.launch(share=True, debug=True)
