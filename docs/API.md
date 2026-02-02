[API.md](https://github.com/user-attachments/files/25021484/API.md)
# API Documentation

Complete reference for all functions and classes.

## Core Functions

### `detect_vehicles(image_path, conf=0.3)`

Detects vehicles in an image.

**Parameters:**
- `image_path` (str): Path to the image file
- `conf` (float, optional): Confidence threshold (default: 0.3)

**Returns:**
- `results`: YOLO detection results object

**Example:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict('car.jpg', conf=0.25)
print(f"Found {len(results[0].boxes)} vehicles")
```

---

### `detect_damage(image_path, conf=0.1)`

Detects damage in an image or image region.

**Parameters:**
- `image_path` (str): Path to the image file
- `conf` (float, optional): Confidence threshold (default: 0.1)

**Returns:**
- `results`: YOLO detection results object with damage boxes

**Example:**
```python
damage_model = YOLO('trained.pt')
results = damage_model.predict('damaged_car.jpg', conf=0.15)

for box in results[0].boxes:
    damage_type = damage_model.names[int(box.cls)]
    print(f"Found: {damage_type}")
```

---

### `get_main_vehicle(boxes, method='largest')`

Identifies the main vehicle from multiple detections.

**Parameters:**
- `boxes` (list): List of YOLO bounding boxes
- `method` (str): Selection method - 'largest', 'center', or 'confidence'

**Returns:**
- `box`: The selected main vehicle bounding box

**Example:**
```python
# Get largest vehicle
main_vehicle = max(results[0].boxes, 
                   key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
```

---

## Model Properties

### YOLO Results Object

**Attributes:**
- `boxes`: Bounding boxes (xyxy format)
- `conf`: Confidence scores
- `cls`: Class IDs
- `names`: Class names dictionary

**Methods:**
- `show()`: Display annotated image
- `save()`: Save annotated image
- `plot()`: Get plotted image array

**Example:**
```python
results = model.predict('image.jpg')

# Access boxes
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf
    class_id = box.cls
    class_name = model.names[int(class_id)]
```

---

## Configuration

### Model Configuration
```python
config = {
    'vehicle_model': 'yolov8n.pt',
    'damage_model': 'trained.pt',
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'imgsz': 640,
    'device': 'cuda'  # or 'cpu'
}
```

### Detection Parameters
```python
params = {
    'conf': 0.25,           # Confidence threshold
    'iou': 0.45,            # NMS IoU threshold
    'imgsz': 640,           # Input image size
    'max_det': 300,         # Maximum detections
    'classes': [2, 5, 7],   # Filter specific classes
    'device': '0',          # GPU device
    'verbose': False        # Print results
}
```

---

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | FPS | Inference Time |
|----------|-----|----------------|
| CPU (i7) | 2-5 | 200-500ms |
| GPU (GTX 1660) | 30-50 | 20-30ms |
| GPU (RTX 3090) | 100+ | 10ms |

*Results vary based on image size and model complexity*
