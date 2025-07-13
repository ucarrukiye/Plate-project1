
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

image_path = "data/test/test.jpg"
image = cv2.imread(image_path)

results = model.predict(image)

# Tahmin sonuçlarındaki kutuları al
boxes = results[0].boxes

# Her kutuyu çiz
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    
    if conf > 0.75:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label = f"Plaka - {conf:.2f}"
    
    # Güven skorunu yaz
    cv2.putText(image, label, (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Görseli göster
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
