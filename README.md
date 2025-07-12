# AI-powered-traffic-congestion-detection-and-prediction
This project leverages Computer Vision and Deep Learning to automatically detect traffic congestion levels and accidents from video footage. It combines YOLOv8 segmentation, occupancy ratio logic, and image classification models to assist urban planners, law enforcement, and citizens in improving road safety and traffic flow.

![APP1](https://github.com/user-attachments/assets/60913bf4-076c-4d7c-b5f5-ee1a79910a5a)

##  Features

- Vehicle detection and segmentation using **YOLOv8**
- Occupancy ratio calculation to measure congestion
- Classification of congestion level: **Low**, **Moderate**, **High**, **Unusual**
- Accident detection using a **CNN-based classifier (MobileNetV2)**
- Web-based UI built with **Flask**
- Upload traffic videos and get instant predictions

  ##  Project Objective

- To **automatically detect** traffic congestion levels using AI-based video analysis.
- To **classify and highlight** accident-prone or anomalous traffic conditions.
- To build a **deployed and interactive web system** for real-time traffic monitoring.

###  Web & Backend
- **Python 3.10+**
- **Flask 3.1**
##  Demo

**Upload a traffic video to instantly detect congestion levels and identify possible accidents using our AI-powered system.**
![APP6](https://github.com/user-attachments/assets/b8a4743f-a48f-4c36-a147-f8de87fe9d8a)
**Predicted congestion level and accident status based on real-time AI analysis of the uploaded traffic footage.**
![APP7](https://github.com/user-attachments/assets/93941446-1a3a-4ff6-8bc9-8e5576a82bdf)
## Model Summary

### 1. Congestion Detection
- Trained a custom YOLOv8 model for vehicle segmentation.
- Calculated occupancy ratio of vehicle pixels to road pixels.
- Random Forest Classifier trained on these features.

### 2. Accident Detection
- MobileNetV2 used as base CNN for accident classification.
- Dataset organized as `Accident` and `Non-Accident` images.
- Predictions made on video frames (1 frame every second approx.).

---
##  Technologies Used

| Category         | Tools / Libraries                       |
|------------------|------------------------------------------|
| Language         | Python 3.10+                             |
| Deep Learning    | YOLOv8 (Ultralytics), TensorFlow 2.19    |
| ML Classifier    | Random Forest (for congestion prediction)|
| Framework        | Flask 3.1, Gunicorn                      |
| Image Processing | OpenCV 4.12                              |
| Training         | Google Colab, Jupyter Notebook           |
---


## ▶ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/traffic-congestion-detector.git
cd traffic-congestion-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
