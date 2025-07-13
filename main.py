from ultralytics import YOLO
from pathlib import Path

def train():
    # Yol ayarları
    base_dir = Path(__file__).parent
    model_path = base_dir / "yolov8n.pt"
    data_path = base_dir / "data.yaml"

    # Modeli yükle
    model = YOLO(str(model_path))

    # Eğitimi başlat
    model.train(
        data=str(data_path),
        epochs=10,
        imgsz=640,
        batch=8
    )

# Bu blok sadece doğrudan çalıştırıldığında devreye girer
if __name__ == "__main__":
    train()
