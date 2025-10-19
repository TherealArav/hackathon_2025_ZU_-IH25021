⚡ Fresh-Gens: Multi-Modal Image Analyzer

Project Overview

Fresh-Gens is a powerful, multi-modal Streamlit web application designed to analyze images using various computer vision and natural language processing techniques. It provides an all-in-one solution for extracting text, translating it, synthesizing speech, and performing object classification on road signs using a high-performance YOLOv8 model.

This project is containerized using Docker to ensure all complex dependencies, especially those related to CUDA/GPU acceleration for YOLO, are easily managed.

Features

Optical Character Recognition (OCR): Extracts English text from uploaded images using EasyOCR.

Language Translation: Translates the extracted English text into Arabic (العربية) using the googletrans library.

Text-to-Speech (TTS): Generates high-quality audio for both the original English and the translated Arabic text using gTTS.

Road Sign Classification (CV): Uses a pre-trained YOLOv8n model (from ultralytics) to detect and classify road signs in the image, leveraging CUDA for GPU-accelerated inference.

Prerequisites

To run this application, you must have the following installed:

Docker: Required for building and running the container.

NVIDIA Container Toolkit (nvidia-docker): Essential for providing the container access to your NVIDIA GPU (e.g., RTX 4070) and CUDA drivers.

CUDA Drivers: Must be properly installed on your host machine to support the base image's CUDA version (12.1 in the provided Dockerfile).

Installation and Setup (Recommended: Docker)

The easiest way to get the application running, including the GPU-accelerated YOLO model, is by using the provided Dockerfile.

1. Build the Docker Image

Navigate to the root directory of your project (where Dockerfile, main.py, cv_logic.py, and yolov8n.pt reside) and run the following command:

docker build -t fresh-gens-analyzer .


2. Run the Container (with GPU Access)

You must include the --gpus all flag to enable the container to utilize your CUDA-enabled GPU for the YOLO model.

docker run --gpus all -p 8501:8501 fresh-gens-analyzer


3. Access the Application

Once the container is running, open your web browser and navigate to:

http://localhost:8501


Local Installation (Alternative)

If you prefer to run the application directly in a Python virtual environment, follow these steps:

Create and Activate Environment:

python -m venv combined_env
source combined_env/bin/activate


Install PyTorch with CUDA:

Find the correct command for your specific CUDA version on the official PyTorch website. Example for CUDA 12.1:

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


Install Other Dependencies:

pip install -r requirements.txt


Run Streamlit:

streamlit run main.py


Project File Structure

File

Purpose

main.py

The core Streamlit UI and orchestration logic.

cv_logic.py

Dedicated microservice containing the YOLOv8 road sign classification function (classify_road_sign).

yolov8n.pt

The pre-trained YOLO model weights used for object detection.

requirements.txt

List of all Python dependencies.

Dockerfile

Instructions for building the Docker image.

Dependencies

streamlit

ultralytics (for YOLOv8)

torch (for GPU acceleration)

opencv-python

easyocr

gTTS

google-trans-new

Pillow

numpy

Note: You must run the following application on a Linux/MacOS system as OpenCV does not have great support with Windows
