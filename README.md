📘 Plate-project1 – README.md Taslağı
markdown
Kopyala
Düzenle
# Plate-project1

Bu proje, YOLOv8 ve CNN tabanlı yöntemlerle plaka tespiti ve veri artırımı (data augmentation) işlemlerini kapsamaktadır.

## 📂 Proje Yapısı

plate-project1/
│
├── data_augmentation.py # Görüntüler için veri artırımı kodları
├── main.py # Ana model çalıştırma dosyası
├── cnn/ # CNN modeli dosyaları
├── yolov8n.pt # Eğitilmiş YOLOv8 model ağırlığı
├── data.yaml # YOLO için veri konfigürasyon dosyası
├── runs/ # YOLO sonuç çıktıları (train, val görselleri)
└── .gitignore # Git tarafından göz ardı edilecek dosyalar

markdown
Kopyala
Düzenle

## 🚀 Özellikler

- ✅ YOLOv8 ile plaka tespiti
- ✅ Albumentations ile gelişmiş augmentasyon teknikleri
- ✅ CNN ile sınıflandırma yapısı
- ✅ .gitignore ile veri dosyalarının dışarıda tutulması

## 🔧 Gereksinimler

- Python 3.10+
- OpenCV
- Albumentations
- Ultralytics YOLOv8
- TensorFlow / Keras

Kurulum için:

```bash
pip install -r requirements.txt
requirements.txt dosyasını henüz oluşturmadıysan, iste dersen onu da hazırlarım.

📌 Notlar
data/images/ ve data/labels/ klasörleri .gitignore dosyasıyla dışlanmıştır.

Eğitilmiş ağırlık dosyaları yolov8n.pt olarak kullanılabilir.
