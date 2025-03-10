import cv2
import numpy as np
import time
import pyttsx3
import threading
from ultralytics import YOLO
import torch

# Allow SegmentationModel in PyTorch (for PyTorch 2.6+)
try:
    from ultralytics.nn.tasks import SegmentationModel
    torch.serialization.add_safe_globals([SegmentationModel])
except ImportError:
    pass  # For older versions, this is not needed

# Load YOLO model
model = YOLO("best.pt")  # Ensure this file is correct
class_names = model.names

# Open video file
cap = cv2.VideoCapture('p.mp4')

frame_count = 0  # Frame counter

# 🚀 Constants for Distance Calculation & Filtering
FOCAL_LENGTH = 700  # Example focal length in pixels (adjustable)
REAL_WORLD_HEIGHT = 0.5  # Approximate average pothole height in meters
MIN_POTHOLE_SIZE = 30  # Ignore potholes smaller than this size

# Text-to-Speech Engine Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust voice speed

# Track last announcement time
last_alert_time = 0

def speak(text):
    """Function to play voice alert asynchronously."""
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Function to apply Gamma Correction
def adjust_gamma(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Stop if video ends

    frame_count += 1
    if frame_count % 1 != 0:  # Process every frame
        continue  

    img = cv2.resize(img, (1120, 600))
    h, w, _ = img.shape

    # 🔥 Apply Enhanced Color Correction
    img = apply_clahe(img)  # Contrast enhancement
    img = adjust_gamma(img, gamma=1.2)  # Adjust brightness

    # Perform inference
    results = model(img)

    pothole_detected = False  # Flag to check if pothole is detected
    detected_direction = None  # Store detected pothole direction

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu().numpy()

            for seg, box in zip(masks, boxes):
                seg = cv2.resize(seg, (w, h))

                # Apply Morphological Closing to refine mask
                kernel = np.ones((3, 3), np.uint8)
                seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
                seg = cv2.GaussianBlur(seg, (5, 5), 0)

                conf = box.conf[0].item()
                if conf < 0.5:  # Lower threshold for better recall
                    continue

                contours, _ = cv2.findContours((seg > 0.5).astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)

                    if w < MIN_POTHOLE_SIZE or h < MIN_POTHOLE_SIZE:  # Ignore small potholes
                        continue

                    # **🔥 Improved Distance Estimation** (Using Height Instead of Width)
                    distance = (FOCAL_LENGTH * REAL_WORLD_HEIGHT) / h
                    distance = round(distance, 2)

                    # Determine Pothole Direction
                    center_x = x + w // 2
                    if center_x < img.shape[1] // 3:
                        direction = "Left"
                    elif center_x > 2 * img.shape[1] // 3:
                        direction = "Right"
                    else:
                        direction = "Center"

                    pothole_detected = True  # Mark pothole detected
                    detected_direction = direction  # Store latest detected direction

                    # 🔲 **Draw Bounding Box (Square Shape)**
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Green Box

                    # 🔲 **Keep Polygon Contours (Default)**
                    cv2.polylines(img, [contour], True, (0, 0, 255), 2)  # Red Contour

                    # 🚀 **Display text with separate colors**
                    text_class = f"{c}"  # Pothole class name
                    text_conf = f"({conf:.2f})"  # Confidence level
                    text_distance = f"{distance}m"  # Estimated distance
                    text_direction = f"{direction}"  # Pothole direction

                    # Define colors
                    color_class = (0, 255, 0)       # Green for class name
                    color_confidence = (255, 0, 0)  # Blue for confidence
                    color_distance = (0, 165, 255)  # Orange for distance
                    color_direction = (255, 255, 0) # Yellow for direction

                    # Base position for text
                    text_x, text_y = x, y - 10
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2

                    # Display each part separately with its own color
                    cv2.putText(img, text_class, (text_x, text_y), font, font_scale, color_class, thickness, cv2.LINE_AA)
                    text_x += cv2.getTextSize(text_class, font, font_scale, thickness)[0][0] + 5

                    cv2.putText(img, text_conf, (text_x, text_y), font, font_scale, color_confidence, thickness, cv2.LINE_AA)
                    text_x += cv2.getTextSize(text_conf, font, font_scale, thickness)[0][0] + 5

                    cv2.putText(img, text_distance, (text_x, text_y), font, font_scale, color_distance, thickness, cv2.LINE_AA)
                    text_x += cv2.getTextSize(text_distance, font, font_scale, thickness)[0][0] + 5

                    cv2.putText(img, text_direction, (text_x, text_y), font, font_scale, color_direction, thickness, cv2.LINE_AA)

    # 🚨 **Voice Alert** - "Go Slow" every 10 seconds
    current_time = time.time()
    if pothole_detected and (current_time - last_alert_time > 10):
        speak("Go slow, maintain 30 Kilometer speed")
        last_alert_time = current_time  # Reset timer

    # 🚨 **Voice Alert for Pothole Direction** (Each Time a Pothole is Detected)
    if detected_direction:
        speak(f"Pothole ahead on the {detected_direction}")

    # Display the output frame
    cv2.imshow('Pothole Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  # Exit if 'q' is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
