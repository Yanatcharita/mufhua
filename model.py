from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# โหลดโมเดล (ใช้ path สั้น เช่น best.pt)
model = YOLO("best.pt")

def detect_muffin_or_chihuahua(image: Image.Image):
    # บันทึกภาพชั่วคราว
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # ทำนาย
    results = model.predict(source=temp_path, conf=0.3, save=False)
    result_img = results[0].plot()
    result_pil = Image.fromarray(result_img)
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]

    os.remove(temp_path)
    return labels, result_pil
