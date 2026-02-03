[README.md](https://github.com/user-attachments/files/25011829/README.md)
# 🚗 Car Damage Detection System

AI-powered car damage detection system using YOLO11 and YOLOv8 models to automatically identify and classify vehicle damage.

## 📋 Overview

This project uses state-of-the-art deep learning models to:
- Detect vehicles in images
- Identify the main vehicle in focus
- Detect and classify different types of damage (scratches, dents, cracks, etc.)
- Provide visual annotations with bounding boxes

## 🎯 Features

- **Automatic Vehicle Detection**: Uses YOLOv8 or custom Roboflow model to detect cars, buses, and trucks
- **Smart Focus Detection**: Automatically identifies the main vehicle in the image (largest/closest)
- **Damage Classification**: Detects multiple damage types using YOLO11m model
- **Visual Results**: Generates annotated images with color-coded bounding boxes
  - 🔵 Blue = Main vehicle
  - ⚪ Gray = Other vehicles  
  - 🔴 Red = Damage areas
- **Detailed Reports**: Provides damage count and type breakdown for each image

## 🛠️ Technologies Used

- **Python 3.x**
- **Ultralytics YOLO** - Object detection framework
- **OpenCV** - Image processing
- **Matplotlib** - Visualization
- **Roboflow** - Dataset management (optional)

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/car-damage-detection.git
cd car-damage-detection
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python matplotlib roboflow
```

### 3. Download models

**YOLO11 Damage Detection Model:**
```bash
git clone https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector.git
```

**YOLOv8 Vehicle Detection (automatic download):**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically on first run
```

## 🚀 Usage

### Basic Usage

```python
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load models
damage_model = YOLO('path/to/trained.pt')
vehicle_model = YOLO('yolov8n.pt')

# Process image
image_path = 'path/to/your/image.jpg'
results = damage_model.predict(image_path, conf=0.10)

# View results
results[0].show()
```

### Advanced Usage (Main Vehicle Focus)

Run the main detection script:

```python
python main.py
```

This will:
1. Detect all vehicles in the image
2. Identify the main vehicle (largest/closest)
3. Analyze only the main vehicle for damage
4. Generate annotated output images
5. Print detailed damage report along with cost recovery per image according to the damage categories, specified later on under damage types detected

## 📁 Project Structure

```
car-damage-detection/
├── main.py                          # Main detection script
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── damaged_pics/                    # Input images folder
├── YOLO11m-Car-Damage-Detector/    # YOLO11 model
│   └── trained.pt                   # Trained weights
└── outputs/                         # Annotated results
```

## 🎨 Output Examples

Each processed image shows:
- **Blue box**: Main vehicle being analyzed
- **Red boxes**: Detected damage areas with labels
- **Gray boxes**: Other vehicles in the image (not analyzed)

## 📊 Damage Types Detected

The model can detect various damage types including:
- Scratches
- Dents
- Cracks
- Paint damage
- Broken parts
- And more (depending on model training)

## ⚙️ Configuration

### Adjust Detection Confidence

```python
# Lower confidence = more detections (may include false positives)
# Higher confidence = fewer detections (more accurate)

# Vehicle detection
vehicle_results = vehicle_model.predict(image, conf=0.3)

# Damage detection
damage_results = damage_model.predict(image, conf=0.10)
```

### Change Focus Selection Method


# Method 1: Largest vehicle (default)
main_box = max(boxes, key=lambda box: box.area)

# Method 2: Closest to center
center_x, center_y = w // 2, h // 2
main_box = min(boxes, key=lambda box: distance_to_center(box))

# Method 3: Highest confidence
main_box = max(boxes, key=lambda box: box.conf)
```

## 🔧 Training Your Own Model

To train on custom data:

1. **Prepare dataset** using Roboflow:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("project-name")
dataset = project.version(1).download("yolov8")
```

2. **Train the model**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

## 📝 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
roboflow>=1.0.0
Pillow>=8.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8 Model**: [ReverendBayes/YOLO11m-Car-Damage-Detector](https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector)
- **Ultralytics**: For the amazing YOLO framework
- **Roboflow Model**: For dataset management tools (https://universe.roboflow.com/curacel-ai/car-damage-detection-5ioys)
- Yunus Karaman, Nimet Karagoz (For research idea and Feedback)
- Gonul Beril Aksu (For Collaboration on the Design Process)
## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔮 Future Improvements

- [ ] Add damage severity classification
- [ ] Create web interface for easy usage
- [ ] Support for video processing
- [ ] Multi-angle damage analysis


---

**Made with ❤️ for automated vehicle damage assessment**
