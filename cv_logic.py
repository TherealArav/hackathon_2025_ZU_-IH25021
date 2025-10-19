import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from ultralytics import YOLO

# --- Model Initialization ---

# NOTE ON MODEL: For a general demonstration, we use yolov8n.pt. 
# For true, production-grade road sign classification, you would replace 
# 'yolov8n.pt' with a model trained specifically on a traffic sign dataset (like GTSDB).
MODEL_PATH = 'yolov8n.pt' 

# Determine the device (CPU or CUDA)
try:
    # Set the device based on CUDA availability, as requested by the user
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'
    
YOLO_MODEL = None
try:
    # Load the YOLO model globally (this is equivalent to caching)
    YOLO_MODEL = YOLO(MODEL_PATH)
    print(f"YOLO Model loaded successfully on device: {DEVICE}")
except Exception as e:
    # Print a warning if the model fails to load (e.g., if ultralytics isn't installed)
    print(f"Warning: Could not load YOLO model ({MODEL_PATH}). Error: {e}. Falling back to error message.")

def classify_road_sign(
    image_np: np.ndarray
) -> Tuple[str, Optional[np.ndarray]]:
    """
    Performs road sign classification using the pre-loaded YOLO model.
    
    It accepts an RGB NumPy array, runs detection, draws bounding boxes, 
    and returns the classification summary text and the modified image array.

    Args:
        image_np: The input image as a NumPy array (RGB format).

    Returns:
        A tuple: (classification_result_string, processed_image_np)
        The processed_image_np is the image with detections drawn.
    """
    if image_np is None:
        return "Error: No image data provided.", None
        
    if YOLO_MODEL is None:
         return "Error: YOLO Model failed to load. Ensure 'ultralytics' and 'torch' are installed.", image_np

    # Run YOLO detection. The image_np is passed directly.
    # The device is set automatically to 'cuda' if available.
    results = YOLO_MODEL(image_np, device=DEVICE, verbose=False)
    
    processed_image = image_np.copy()
    detection_summary = []
    
    # Process results from YOLO
    if results and results[0].boxes:
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) # Bounding boxes (x1, y1, x2, y2)
        confs = results[0].boxes.conf.cpu().numpy()             # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs
        names = results[0].names                                # Class names map
        
        # Convert image to BGR for OpenCV drawing functions
        processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        for box, conf, class_id in zip(boxes, confs, class_ids):
            class_name = names[class_id]
            label = f"{class_name}: {conf:.2f}"
            
            # Simple color assignment for drawing boxes (BGR format)
            if 'stop' in class_name.lower():
                color = (0, 0, 255) # Red
            elif 'traffic light' in class_name.lower() or 'traffic sign' in class_name.lower():
                 color = (0, 255, 255) # Yellow
            else:
                color = (255, 0, 0) # Blue (General)

            # Draw bounding box
            cv2.rectangle(processed_image_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label background
            (w_label, h_label), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(processed_image_bgr, (box[0], box[1] - h_label - baseline), (box[0] + w_label, box[1]), color, -1)
            
            # Put label text (Black text on colored background)
            cv2.putText(processed_image_bgr, label, (box[0], box[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add to summary
            detection_summary.append(f"- {class_name} (Confidence: {conf:.2f})")

        # Convert back to RGB for Streamlit display
        processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
        
        classification_text = "Detected Road Signs/Objects:\n" + "\n".join(detection_summary)
    else:
        classification_text = "No road signs or relevant objects detected by YOLO model."

    return classification_text, processed_image
