from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# โหลดโมเดลเพียงครั้งเดียว
model = YOLO("best.pt")  # ต้องแน่ใจว่าไฟล์นี้อยู่ใน path เดียวกัน หรือระบุ path ชัดเจน

def detect_muffin_or_chihuahua(pil_image):
    # ตรวจสอบว่าภาพเป็น PIL
    if not isinstance(pil_image, Image.Image):
        raise ValueError("Input must be a PIL.Image")

    # แปลง PIL image เป็น NumPy array
    image_np = np.array(pil_image)

    # ทำนาย
    results = model.predict(source=image_np, conf=0.3, save=False)

    # ดึงภาพพร้อมกล่องที่ตรวจพบ
    result_img = results[0].plot()

    # แปลง BGR เป็น RGB
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_img_rgb)

    # ดึง label ที่ตรวจพบ
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]

    return labels, result_pil
