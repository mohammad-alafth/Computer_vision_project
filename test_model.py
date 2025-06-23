from ultralytics import YOLO

def main():
    # Load model YOLOv8n (versi ringan, bisa ganti dengan yolov8s/m/l/x)
    model = YOLO("yolov8n.pt")

    # Training model
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        # device='cuda',     # Gunakan 'cuda' jika tersedia, atau 'cpu'
        batch=8,           # Sesuaikan dengan kapasitas VRAM
        # workers=2,         # Aman untuk Windows (hindari crash karena multi-proses)
        # name="yolov8_coin_detection"  # Nama folder runs/train
    )

    # Evaluasi model setelah training
    metrics = model.val()
    print("\n=== EVALUASI MODEL (RATA-RATA) ===")
    print(f"mAP50      : {metrics.map50():.4f}")
    print(f"mAP50-95   : {metrics.map():.4f}")
    print(f"Precision  : {metrics.mp():.4f}")
    print(f"Recall     : {metrics.mr():.4f}")

    # Nama kelas sesuai data.yaml
    class_names = ['100', '1000', '200', '500']

    print("\n=== EVALUASI PER KELAS ===")
    for i, name in enumerate(class_names):
        p, r, ap50, ap = metrics.class_result(i)
        print(f"{name:<10} | Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {ap50:.4f} | mAP50-95: {ap:.4f}")

if __name__ == '__main__':
    main()
