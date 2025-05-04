import cv2
from PIL import Image
import numpy as np

def detect_muffin_or_chihuahua(uploaded_file):
    # โหลดโมเดล
    model = YOLO("best.pt")
    
    # บันทึกไฟล์อัปโหลดลงเครื่อง
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ทำนาย
    results = model.predict(source=file_path, conf=0.3, save=False)

    # ดึงภาพพร้อมกล่องผลลัพธ์
    result_img = results[0].plot()

    # แปลง BGR → RGB
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # แปลงเป็น PIL image เพื่อให้ Streamlit แสดงได้ถูกสี
    return Image.fromarray(result_img)
