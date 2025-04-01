from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load the custom YOLOv8 model
model = YOLO("weights/best.pt")  # make sure this path points to your trained model

# Color detection helper (HSV-based)
def detect_signal_color(image_crop):
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
    mask_red = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

    red_pixels = np.sum(mask_red)
    green_pixels = np.sum(mask_green)

    if red_pixels > green_pixels:
        return "Red"
    elif green_pixels > red_pixels:
        return "Green"
    else:
        return "Unknown"

# Main detection pipeline
def analyze_image(image: Image.Image):
    np_image = np.array(image.convert("RGB"))
    results = model(np_image)[0]

    labels = []
    signal_color = "Unknown"
    hazard_level = "Low"
    direction = "Straight"  # Optional logic for future use

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = results.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        labels.append(label)

        # Analyze signal color if traffic signal found
        if label in ["signal", "traffic_light"]:
            x1, y1, x2, y2 = xyxy
            roi = np_image[y1:y2, x1:x2]
            if roi.size > 0:
                signal_color = detect_signal_color(roi)

    # Rule-based hazard level
    if "vehicle_on_track" in labels and signal_color == "Red":
        hazard_level = "High"
    elif "rock" in labels or "rail_damage" in labels:
        hazard_level = "High"
    elif "person" in labels and signal_color == "Red":
        hazard_level = "Medium"
    elif "vehicle_on_track" in labels:
        hazard_level = "Medium"

    summary = f"Detected: {', '.join(labels)}"

    return {
        "summary": summary,
        "labels": labels,
        "signal_color": signal_color,
        "hazard_level": hazard_level,
        "direction": direction
    }