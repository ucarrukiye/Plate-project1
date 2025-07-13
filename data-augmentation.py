import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os

# Augmentasyon tanımları
augmentations = [
    ("Rotate", A.Rotate(limit=20, p=1.0)),
    ("Horizontal_Flip", A.HorizontalFlip(p=1.0)),
    ("Brightness", A.RandomBrightnessContrast(p=1.0)),
    ("Zoom", A.RandomScale(scale_limit=0.4, p=1.0)),
    ("Gauss_Noise", A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)),
    ("Motion_Blur", A.MotionBlur(blur_limit=5, p=1.0)),
    ("Median_Blur", A.MedianBlur(blur_limit=3, p=1.0)),
    ("RandomCrop", A.RandomCrop(height=500, width=500, p=1.0)),
]

# Tüm görselleri işle
image_dir = "data/images"
label_dir = "data/labels"
image_paths = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

for image_file in image_paths:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

    # Görseli oku
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Etiket dosyasını oku
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

    # Görselin temel adı
    base_img_name = os.path.splitext(image_file)[0]
    h, w, _ = image.shape  # RandomCrop için boyut bilgisi

    for i, (title, aug) in enumerate(augmentations, start=1):
        img_out_path = f"{image_dir}/{base_img_name}_{title}.jpg"
        label_out_path = f"{label_dir}/{base_img_name}_
