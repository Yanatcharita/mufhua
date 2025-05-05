# Project Overview
This project originated from my interest in applying Deep Learning to image classification, particularly in scenarios where images look deceptively similar but represent entirely different contexts — like chihuahuas and muffins.

Initially, I planned to explore Convolutional Neural Networks (CNNs) to tackle this challenge. However, I wanted to make the project more enchanted and technically interesting, so I decided to upgrade it into an object detection task, where the model needs to identify and localize multiple objects in a single image using bounding boxes.

At first, I experimented with Faster R-CNN due to its strong accuracy, but it turned out to be too slow for real-time or interactive use cases. I later switched to YOLOv8, which offered significantly better speed and efficiency. I retrained and fine-tuned the model with a custom dataset I curated, containing images of muffins and chihuahuas.

After obtaining a well-performing model, I deployed it as a Streamlit web application, allowing users to upload an image and instantly see whether it contains a muffin or a chihuahua — complete with bounding boxes around detected objects.

This project not only deepened my understanding of object detection techniques, but also let me combine personal interest (I truly love chihuahuas!) with real-world AI implementation in a fun, meaningful way.
