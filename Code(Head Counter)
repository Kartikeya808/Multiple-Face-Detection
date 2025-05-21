import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load Face Detection Model (Ensure you have a face-specific model)
model_path = r"C:\Users\Hp\OneDrive\Desktop\DSA LBFAT\yolov8n.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Read the Image
image_path = r"C:\Users\Hp\OneDrive\Desktop\DSA LBFAT\INPUT2.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Resize Image (Optional for Faster Processing)
resize_factor = 0.5
image = cv2.resize(image, (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor)))

# Perform Face Detection with Confidence & IOU Filtering
results = model(image, conf=0.5, iou=0.4)  # Adjust these values based on performance

# Extract Detections
detections = results[0].boxes.data.cpu().numpy()

# Draw Bounding Boxes
face_count = 0
if len(detections) == 0:
    print("No faces detected.")
else:
    for (x1, y1, x2, y2, conf, cls) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        conf = round(float(conf), 2)
        face_count += 1

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw semi-transparent label background
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Draw text label
        cv2.putText(image, f"Face {face_count}: {conf}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display Results
print(f"Total faces detected: {face_count}")
cv2.imshow("YOLOv8 Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Processed Image
cv2.imwrite("yolov8_detected_faces.jpg", image)
print("Image saved as 'yolov8_detected_faces.jpg'.")
