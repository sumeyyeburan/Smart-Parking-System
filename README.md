# Smart Parking Lot Occupancy Detection System

A graduation thesis project focused on real-time parking lot occupancy detection using a hybrid computer vision approach. 
This system analyzes surveillance camera feeds, localizes vehicles, classifies parking slot availability, and generates statistical reports via a web-based dashboard.

## Overview

The project aims to develop a cost-effective, highly scalable vision-based parking monitoring system that can be used in smart city infrastructures and commercial parking lot management. 
The system processes static images and live video streams in real time. Instead of relying solely on one model, it utilizes a dual-verification hybrid decision engine. It evaluates bounding box overlaps from YOLOv8 alongside patch-level classification probabilities from a custom-trained MobileNetV2 CNN. Based on these inferences and spatial rules, the system categorizes each parking slot as either occupied or empty, updating a relational database to support historical utilization reporting.

## Key Features

* Real-time image and video feed processing
* Hybrid vehicle detection and slot classification (YOLOv8 + MobileNetV2 CNN)
* Interactive dynamic ROI (Region of Interest) drawing via web dashboard
* Automated camera source identification and coordinate mapping
* Multi-stage decision tree to mitigate blind spots, shadows, and occlusions
* Hourly occupancy logging and statistical reporting via SQLite
* Role-based access control (Admin and User)
* Modular Python project structure integrated with Streamlit

## Occupancy Decision Logic

The occupancy state is estimated based on a hybrid algorithmic decision tree combining CNN confidence and YOLOv8 bounding box overlap (IoU).

| Condition | YOLOv8 Overlap | CNN Confidence | Result |
| :--- | :--- | :--- | :--- |
| Standard Detection | > 30% Overlap | > Threshold (0.4/0.5) | **Occupied** |
| Clear Space | < 30% Overlap | < Threshold (0.4/0.5) | **Empty** |
| Camera Blind Spot (Bottom Frame) | Not Detected | > Threshold (0.4/0.5) | **Occupied** |
| Heavy Occlusion (e.g., Trees) | Not Detected | Overwhelming (> 0.99) | **Occupied** |

If the conditions do not meet the occupancy thresholds, the system classifies the slot as empty.

## System Architecture

The general pipeline of the system is as follows:

Camera Input (Image/Video)
     ↓
Interactive Coordinate Scaling & Camera Identification
     ↓
YOLOv8 Full-Frame Vehicle Localization
     ↓
MobileNetV2 CNN Patch-Level Probability Analysis
     ↓
Hybrid Decision Tree Resolution
     ↓
Database Tracing & Streamlit Dashboard Visualization

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLOv8
* TensorFlow / Keras
* Pandas
* SQLite3
* Streamlit & Streamlit-Drawable-Canvas
* Git & GitHub

## Project Structure

```text
Smart-Parking-System/
│
├── models/               # Trained CNN models (.h5) and YOLO weights
├── data/                 # CSV coordinates, uploaded assets, and dataset files
├── output/               # Processed output images/frames
├── app_streamlit.py      # Main Streamlit web dashboard application
├── database.py           # SQLite database initialization and query operations
├── main.py               # Core logic for hybrid inference
├── roi_manager.py        # Coordinate extraction and bounding box logic
├── train_cnn.py          # Script for training the MobileNetV2 classifier
├── draw_rois.py          # Manual OpenCV coordinate drawing tool
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files and folders
└── README.md             # Project documentation
```

## Installation
Clone the repository:

## Bash
git clone [https://github.com/sumeyyeburan/Smart-Parking-System.git](https://github.com/sumeyyeburan/Smart-Parking-System.git)

cd Smart-Parking-System

Create and activate a virtual environment:

python -m venv venv

On Windows: venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Usage
Run the Streamlit dashboard application:

streamlit run app_streamlit.py

Scalability and Deployment Approach

This project follows a hardware-minimalist design principle. By utilizing existing surveillance camera feeds and deep learning, the system eliminates the need for expensive physical ground sensors for every parking slot. It only records statistical outputs such as timestamp, active camera, total spots, and calculated occupancy rates to the database.

## Thesis Information
Project Title: Real-Time Parking Lot Occupancy Detection System

Type: Graduation Thesis Project

## Department: Computer Engineering

University: Çukurova University

## Author
Sümeyye Buran, Computer Engineering Student, Çukurova University

## Advisor
Assoc. Prof. Dr. Serkan Kartal

## License
This project is developed for academic purposes. License information can be added later depending on the publication status of the project.
