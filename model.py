from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import cv2
import numpy as np

# load model
model = YOLO("best.pt")

def detect_muffin_or_chihuahua(image: Image.Image):
    # save images
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # predict
    results = model.predict(source=temp_path, conf=0.5, save=False)
    result_img = results[0].plot()

    # แปลงจาก BGR เป็น RGB (ผลลัพธ์จาก .plot() เป็น BGR)
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    # แปลงกลับเป็น RGB ก่อนแสดงผล
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    result_pil = Image.fromarray(result_rgb)
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]

    os.remove(temp_path)
    return labels, result_pil
