from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# โหลดโมเดลเพียงครั้งเดียว
model = YOLO("best.pt")

def detect_muffin_or_chihuahua(pil_image):
    # แปลง PIL image เป็น NumPy array
    image_np = np.array(pil_image)

    # ทำนาย
    results = model.predict(source=image_np, conf=0.3, save=False)

    # วาดผลลัพธ์
    result_img = results[0].plot()

    # แปลง BGR เป็น RGB
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # แปลงเป็น PIL image
    result_pil = Image.fromarray(result_img_rgb)

    # ดึง label
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]

    return labels, result_pil
