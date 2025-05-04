%%writefile app.py
import streamlit as st
from PIL import Image
from model import detect_muffin_or_chihuahua

st.set_page_config(page_title="Muffin or Chihuahua Detector")

st.title("🐶 Muffin or Chihuahua Detector 🧁")
st.write("Upload an image to let the AI detect whether it's a chihuahua or a muffin.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Detecting..."):
        result_labels, result_image = detect_muffin_or_chihuahua(image)

    st.success("Detection complete!")
    st.image(result_image, caption="Detection Result", use_container_width=True)
    st.write("🔍 Detected:", ", ".join(result_labels) if result_labels else "No objects detected.")
