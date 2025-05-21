import cv2
import torch
import numpy as np
import face_recognition
import os
from ultralytics import YOLO

# === Load YOLOv8 Face Detection Model ===
model_path = r"C:\Users\Hp\OneDrive\Desktop\DSA PROJECT REVAMPED\yolov8n.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# === Load Known Faces ===
known_encodings = []
known_labels = []

known_faces_dir = r"C:\Users\Hp\OneDrive\Desktop\known_faces"
for file in os.listdir(known_faces_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_dir, file)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            name_reg = os.path.splitext(file)[0]  # e.g., Ravi_21BCE0942
            known_encodings.append(encoding)
            known_labels.append(name_reg)
        else:
            print(f"Warning: No face found in {file}")

print(f"\nLoaded {len(known_encodings)} known face(s):")
print("Labels:", known_labels)

# === Load Image to Process ===
image_path = r"C:\Users\Hp\OneDrive\Desktop\DSA PROJECT REVAMPED\INPUT2.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

resize_factor = 0.5
image = cv2.resize(image, (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor)))

# === Run YOLOv8 Face Detection ===
results = model(image, conf=0.5, iou=0.4)
detections = results[0].boxes.data.cpu().numpy()

face_count = 0

if len(detections) == 0:
    print("No faces detected.")
else:
    for (x1, y1, x2, y2, conf, cls) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Expand bounding box slightly
        pad = 20
        h, w, _ = image.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = image[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        face_encodings = face_recognition.face_encodings(face_rgb)
        name = "Unknown"
        regno = ""

        if face_encodings:
            match_results = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])

            if True in match_results:
                best_match_index = np.argmin(face_distances)
                label_full = known_labels[best_match_index]

                if '_' in label_full:
                    name, regno = label_full.split('_', 1)
                else:
                    name = label_full
                    regno = ""

                print(f"Matched: {name} ({regno})")
            else:
                print("Face detected but no match found.")
        else:
            print("No encoding found for face region.")

        # === Draw Bounding Box ===
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === Write Labels Inside Box ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Position text near the bottom-left inside the box
        text_y_offset = 15
        cv2.putText(image, name, (x1 + 5, y1 + text_y_offset), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, regno, (x1 + 5, y1 + text_y_offset + 18), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

        face_count += 1

print(f"\nTotal faces detected: {face_count}")
cv2.imshow("YOLOv8 Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("yolov8_face_recognition_output.jpg", image)
print("Image saved as 'yolov8_face_recognition_output.jpg'.")
