# Smoke Vehicle Detection and Dashboard

This project provides a web-based dashboard for uploading videos, processing them using AI models (YOLO for object detection, EasyOCR for number plate recognition), and generating analytics reports and processed videos. It leverages Django for the web framework, Celery for asynchronous task processing, and RabbitMQ as the message broker.

---

## 🚀 Features

- **Video Upload**: Easily upload video files through a user-friendly web interface.
- **Asynchronous Processing**: Videos are processed in the background using Celery, preventing UI blocking.
- **Real-time Progress Tracking**: A progress bar updates in real-time to show the status of video processing.
- **Smoke and Vehicle Detection**: Utilizes YOLOv8 for detecting smoke emissions, smoke-emitting vehicles, and non-smoke vehicles.
- **Number Plate Recognition (OCR)**: Integrates EasyOCR to identify and extract number plates from detected vehicles.
- **Processed Video Output**: Generates a new video with bounding boxes and labels for detected objects.
- **Detailed Reports**: Produces CSV and Excel summary reports of detections.
- **Visual Analytics**: Generates bar and pie charts summarizing vehicle types and plate detection statistics.
- **Email Integration**: Option to send processed reports via email.
- **Data Management**: Clear all uploaded and processed video data with a single click, or delete individual video logs.

---
NOTE
Optional but effective use a GPU 

## 🛠️ Technologies Used

### Backend
- **Django** – Web framework
- **Celery** – Asynchronous task queue
- **RabbitMQ** – Message broker for Celery
- **Django-Celery-Results** – Stores Celery task results in the Django database
- **OpenCV (cv2)** – Video processing and frame manipulation
- **Ultralytics YOLOv8** – Object detection (`best.pt` pre-trained model)
- **EasyOCR** – Optical character recognition for number plates
- **Pandas** – Data manipulation and CSV/Excel generation
- **Matplotlib** – Chart generation
- **ffmpeg** – Video re-encoding for web compatibility

### Frontend
- **HTML**, **CSS**, **JavaScript**
- **Font Awesome** – Icons

---
🔎 Note: This project supports both GPU and CPU. If an NVIDIA GPU is available, PyTorch will use it for faster processing. Otherwise, the system will automatically fall back to CPU execution.

## ⚙️ Setup and Installation

Follow these steps to get the project running locally.

### ✅ Prerequisites

Ensure the following are installed:

- **Python 3.8+**
- **pip** (Python package installer)
- **RabbitMQ Server**
- **ffmpeg**
- A **YOLOv8 model file** (`best.pt`)

### 🔧 RabbitMQ Installation

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
