import cv2
import numpy as np
import os

image_dir = "data/images"
label_dir = "data/labels"

X = []
y = []

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        print(f"Etiket dosyası eksik: {label_path}")
        continue

    # Görseli oku ve yeniden boyutlandır
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = image / 255.0  # normalize et
    X.append(image)

    # Etiket dosyasını oku (ilk bbox alınıyor - tek nesne varsayımı)
    with open(label_path, "r") as f:
        line = f.readline()
        parts = line.strip().split()
        bbox = list(map(float, parts[1:]))  # x_center, y_center, width, height
        y.append(bbox)

# NumPy dizilerine çevir
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Kaydet
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Dataset dosyaları oluşturuldu: X.npy, y.npy")
