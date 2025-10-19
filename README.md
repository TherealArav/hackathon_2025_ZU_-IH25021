<h1>⚡ Fresh-Gens: Multi-Modal Image Analyzer </h1>  

<h2> Project Overview </h2>  

<p>Visi-Tech is a powerful, multi-modal Streamlit web application designed to analyze images using various <strong>computer vision</strong> and <strong>natural language processing</strong> techniques. It provides an all-in-one solution for extracting text, translating it, synthesizing speech, and performing object classification on road signs using a high-performance YOLOv8 model.</p>

<h2>Features</h2>

1. Optical Character Recognition (OCR): Extracts English text from uploaded images using EasyOCR.
2. Language Translation: Translates the extracted English text into Arabic (العربية) using the googletrans library.
3. Text-to-Speech (TTS): Generates high-quality audio for both the original English and the translated Arabic text using gTTS.
4. Road Sign Classification (CV): Uses a pre-trained YOLOv8n model (from ultralytics) to detect and classify road signs in the 5. image, leveraging CUDA for GPU-accelerated inference.

<h2>Prerequisites</h2>

<p>To run this application, you must have the following installed:</P>

1. NVIDIA Container Toolkit (nvidia-docker): Essential for providing the container access to your NVIDIA GPU (e.g., RTX 4070) 2. and CUDA drivers.
3. CUDA Drivers: Must be properly installed on your host machine to support the base image's CUDA version (12.1).

<h2>Installation and Setup</h2>

Local Installation 

If you prefer to run the application directly in a Python virtual environment, follow these steps:

1. Create and Activate Environment:
2. python -m venv combined_env3. 
3. source combined_env/bin/activate


4. Install PyTorch with CUDA:

<p>Find the correct command for your specific CUDA version on the official PyTorch website. Example for CUDA 12.1:</p>

<code>pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)</code>


<h2>Install Other Dependencies:</h2>

1. pip install -r requirements.txt


2. Run Streamlit:

3. streamlit run main.py


<h2>Project File Structure</h2>

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

<h2>Dependencies</h2>
1. streamlit
2. ultralytics (for YOLOv8)
3. torch (for GPU acceleration)
4. opencv-python
5. easyocr
6. gTTS
7. google-trans-new
8. Pillow
9. numpy

<em>Note: You must run the following application on a Linux/MacOS system as OpenCV does not have great support with Windows</em>
