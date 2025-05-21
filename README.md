# Face Detection & Recognition using YOLOv8

This project implements face detection and recognition on static images using YOLOv8, OpenCV, and the `face_recognition` and `dlib` libraries.

## 📌 Features

- Face **detection** using YOLOv8 (`yolov8n.pt` model)
- Face **recognition** using deep face embeddings via `face_recognition`
- Annotates images with **bounding boxes**, **names**, and **confidence scores**
- Supports adding new known faces easily
- Lightweight and runs locally

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (via `ultralytics`)
- OpenCV
- Dlib
- face_recognition

---



## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-recognition-yolov8.git
cd face-recognition-yolov8
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed, then:

```bash
pip install -r requirements.txt
```

### 3. Download the YOLOv8 Model
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy
```

Or manually download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics).

### 4. Run the Script
```bash
python face_recognition_yolov8.py
```

---

## 📁 Project Structure

```
face-recognition-yolov8/
├── known_faces/
│   └── person1.jpg
├── test_images/
│   └── group_photo.jpg
├── face_recognition_yolov8.py
├── requirements.txt
└── README.md
```

---

## ✅ To-Do / Future Enhancements

- Support video stream input (webcam or video file)
- Create GUI using Tkinter or Streamlit
- Train custom YOLOv8 face detector for higher accuracy

---

## 🙋‍♂️ About Me

I’m a CSE + Business Systems undergrad at VIT Vellore with a passion for AI and Fintech.  
This project was a deep dive into computer vision, and I'm eager to learn more and collaborate.  
Feel free to connect on [LinkedIn]: www.linkedin.com/in/kartikeya-singh-3b898931b

---

## 🏷️ Tags

#YOLOv8 #ComputerVision #Python #FaceRecognition #OpenCV #StudentProject #Ultralytics
