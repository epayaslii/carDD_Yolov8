# Usage Guide

Complete guide for using the Car Damage Detection System.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download models
git clone https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector.git
```

### 2. Run Detection
```python
python main.py
```

## Basic Usage

### Single Image Detection

```python
from ultralytics import YOLO
import cv2

# Load models
damage_model = YOLO('YOLO11m-Car-Damage-Detector/trained.pt')
vehicle_model = YOLO('yolov8n.pt')

# Detect damage
image_path = 'path/to/car.jpg'
results = damage_model.predict(image_path, conf=0.10)

# Display results
results[0].show()
```

### Batch Processing

```python
from pathlib import Path

# Process multiple images
image_folder = 'damaged_pics/'
for img_path in Path(image_folder).glob('*.jpg'):
    results = damage_model.predict(str(img_path), save=True)
    print(f"Processed: {img_path.name}")
```

## Advanced Features

### 1. Main Vehicle Focus Detection

Automatically detects and analyzes only the primary vehicle:

```python
# Method 1: Largest vehicle (default)
main_box = max(boxes, key=lambda b: b.area)

# Method 2: Closest to center
center_x, center_y = width // 2, height // 2
main_box = min(boxes, key=lambda b: distance_to_center(b))

# Method 3: Highest confidence
main_box = max(boxes, key=lambda b: b.conf)
```

### 2. Damage Classification

```python
# Get detailed damage information
for box in results[0].boxes:
    damage_type = damage_model.names[int(box.cls)]
    confidence = box.conf
    print(f"{damage_type}: {confidence:.2%}")
```

### 3. Custom Annotations

```python
import cv2

# Draw custom bounding boxes
image = cv2.imread('car.jpg')
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
cv2.imwrite('annotated.jpg', image)
```

## Configuration

### Confidence Thresholds

```python
# Lower threshold = more detections (may include false positives)
results = model.predict(image, conf=0.05)

# Higher threshold = fewer detections (more accurate)
results = model.predict(image, conf=0.50)
```

**Recommended values:**
- Vehicle detection: `conf=0.25` to `0.40`
- Damage detection: `conf=0.10` to `0.25`

### Image Size

```python
# Smaller = faster, less accurate
results = model.predict(image, imgsz=320)

# Larger = slower, more accurate
results = model.predict(image, imgsz=1280)
```

**Recommended:** `imgsz=640` (default)

### Device Selection

```python
# Use GPU (if available)
results = model.predict(image, device='cuda')

# Use CPU
results = model.predict(image, device='cpu')

# Auto-detect
results = model.predict(image, device='0')  # First GPU or CPU
```

## Output Formats

### Save Annotated Images

```python
# Save with default name
results = model.predict(image, save=True)

# Save to specific directory
results = model.predict(image, save=True, project='outputs', name='run1')
```

### Export Results to JSON

```python
import json

damage_report = {
    'image': image_path,
    'vehicles': len(vehicle_boxes),
    'damages': []
}

for box in damage_boxes:
    damage_report['damages'].append({
        'type': model.names[int(box.cls)],
        'confidence': float(box.conf),
        'bbox': box.xyxy[0].tolist()
    })

with open('report.json', 'w') as f:
    json.dump(damage_report, f, indent=2)
```

### Export to CSV

```python
import csv

with open('damages.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Damage Type', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        writer.writerow([
            image_path,
            model.names[int(box.cls)],
            float(box.conf),
            x1, y1, x2, y2
        ])
```

## Troubleshooting

### Common Issues

**1. Model not found**
```
Error: FileNotFoundError: trained.pt not found
```
**Solution:** Download the model or check the path
```python
# Verify model exists
import os
print(os.path.exists('YOLO11m-Car-Damage-Detector/trained.pt'))
```

**2. Out of memory**
```
Error: CUDA out of memory
```
**Solution:** Reduce batch size or image size
```python
results = model.predict(image, imgsz=320)  # Smaller size
```

**3. No detections**
```
Warning: No damage detected
```
**Solution:** Lower confidence threshold
```python
results = model.predict(image, conf=0.05)  # Lower threshold
```

**4. Too many false positives**
```
Warning: Many incorrect detections
```
**Solution:** Increase confidence threshold
```python
results = model.predict(image, conf=0.40)  # Higher threshold
```

### Performance Tips

1. **Use GPU** if available (10-50x faster)
2. **Resize large images** before processing
3. **Batch process** multiple images together
4. **Use appropriate confidence** thresholds
5. **Cache models** in memory (don't reload each time)

### Getting Help

- Check [Issues](https://github.com/YOUR_USERNAME/car-damage-detection/issues)
- Read [FAQ](#faq)
- Open a new issue with details

## FAQ

**Q: What image formats are supported?**  
A: JPG, JPEG, PNG, BMP, TIFF, WEBP

**Q: Can I use videos?**  
A: Yes, but you need to extract frames first or modify the code

**Q: How accurate is the model?**  
A: Accuracy depends on image quality and damage visibility. Test with your specific use case.

**Q: Can I train on my own data?**  
A: Yes! See the Training section in README.md

---

For more examples, check the [examples/](../examples/) folder.
