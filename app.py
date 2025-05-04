import streamlit as st
from PIL import Image
from model import detect_muffin_or_chihuahua

st.set_page_config(page_title="Muffin or Chihuahua Detector")

st.title("🐶 Muffin or Chihuahua Detector 🧁")
st.write("อัปโหลดภาพเพื่อให้ AI ตรวจว่าเป็นชิวาวาหรือมัฟฟิน")

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปต้นฉบับ", use_container_width=True)  

    with st.spinner("กำลังตรวจสอบ..."):
        result_labels, result_image = detect_muffin_or_chihuahua(image)


    st.success("เสร็จแล้ว!")
    st.image(result_image, caption="ผลลัพธ์จากโมเดล")
    st.write("🔍 ตรวจพบ:", ", ".join(result_labels) if result_labels else "ไม่พบวัตถุ")

