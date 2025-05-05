# Project Overview
This project originated from my interest in applying Deep Learning to image classification, particularly in scenarios where images look deceptively similar but represent entirely different contexts — like chihuahuas and muffins.

![image](https://github.com/user-attachments/assets/a8b73ef5-82ce-401b-8df5-87a04078393a)

Figure 1. Chihuahua vs muffin



Initially, I planned to explore Convolutional Neural Networks (CNNs) to tackle this challenge. However, I wanted to make the project more enchanted and technically interesting, so I decided to upgrade it into an object detection task, where the model needs to identify and localize multiple objects in a single image using bounding boxes.

At first, I experimented with Faster R-CNN due to its strong accuracy, but it turned out to be too slow for real-time or interactive use cases. I later switched to YOLOv8, which offered significantly better speed and efficiency. I retrained and fine-tuned the model with a custom dataset I curated, containing images of muffins and chihuahuas.

After obtaining a well-performing model, I deployed it as a Streamlit web application, allowing users to upload an image and instantly see whether it contains a muffin or a chihuahua — complete with bounding boxes around detected objects.

This project not only deepened my understanding of object detection techniques, but also let me combine personal interest (I truly love chihuahuas!) with real-world AI implementation in a fun, meaningful way.
# Workflow Summary
This section outlines the step-by-step process I followed to build, expand, and improve the object detection model.

## Dataset 
I collected a total of 1,500 images, consisting of:

500 images of muffins

1,000 images of chihuahuas

These were split into training and validation sets using an 80/20 ratio.

## Manual Annotation
Out of the full dataset, I manually labeled:

300 muffin images

400 chihuahua images

This gave me a total of 700 annotated images used for the initial model training.

## Initial YOLOv8 Training
I trained a YOLOv8 model (yolov8n.pt) using this small annotated dataset.

The training was run for 50 epochs with an image size of 640x640.

## Auto-labeling Unannotated Images
I used the trained model (best.pt) to detect objects in the remaining unlabeled images.

This step generated YOLO-format .txt annotation files for the rest of the dataset using model.predict().

## Dataset Expansion
I combined the auto-labeled images with the manually labeled ones, resulting in a much larger training dataset.

The dataset was restructured to include all images and labels for retraining.

## Model Retraining
With the expanded dataset, I retrained the YOLOv8 model, improving its performance and robustness in detecting both muffins and chihuahuas.

## Try Uploading an Image to Test the Model
![Screenshot 2025-05-03 151524](https://github.com/user-attachments/assets/ab438277-f1a3-4904-9553-5dd8097d4032)

Figure 2. Test the Model

## Model Evaluation
### Box Loss: measures bounding box regression error.

![image](https://github.com/user-attachments/assets/7227b173-b99d-4da0-9a5c-e78a72cc1081)

Figure 3. Training vs Validation Box Loss

### mAP@0.5 (mean Average Precision): measures object detection accuracy at an IoU threshold of 0.5.


![image](https://github.com/user-attachments/assets/20eb8a59-1f4f-42a2-991e-33dff4eecb84)

Figure 4. Train vs  Validation mAP50


# 🔗 Demo Web App:
https://mufhua-hrjlzdbjbdv8jnvyndrrhr.streamlit.app/

![Screenshot 2025-05-04 153248](https://github.com/user-attachments/assets/ac3e4845-0799-4597-8441-c554a207009c)

Figure 5. Web App Detection Result

# Possible Improvements and Future Work
## Add More Classes

The model currently detects only two classes. Expanding to include other confusing objects (e.g., cupcakes, dogs of other breeds) could improve its real-world robustness.

## Improve Dataset Diversity

Including more varied backgrounds, lighting conditions, and object orientations could help generalize the model further.

## Experiment with Other Architectures

Trying other models like YOLO-World, DETR, or even fine-tuning Vision Transformers could be interesting for comparison.

