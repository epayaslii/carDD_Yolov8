# ========================================
# INSTALLATION AND SETUP
# ========================================

!pip install ultralytics
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="scJ5N349SaurBYQn080w")
project = rf.workspace("car-damage-detection-yolo").project("car-damage-detection-5ioys-pqnzm")
version = project.version(1)
dataset = version.download("yolov8")

import os

print("📥 Cloning repository...")
!git clone https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector.git

print("\n📁 Searching for model files...")
!find /content/YOLO11m-Car-Damage-Detector -name "*.pt"

print("\n📂 Repository structure:")
!ls -lh /content/YOLO11m-Car-Damage-Detector/

if os.path.exists('/content/YOLO11m-Car-Damage-Detector/weights'):
    print("\n📦 Weights folder contents:")
    !ls -lh /content/YOLO11m-Car-Damage-Detector/weights/


# ========================================
# BASIC DAMAGE DETECTION
# ========================================

!pip install ultralytics
from ultralytics import YOLO
import os

model_path = '/content/YOLO11m-Car-Damage-Detector/trained.pt'

if os.path.exists(model_path):
    model = YOLO(model_path)
    
    image_source = '/content/damaged_pics'
    
    results = model.predict(source=image_source, save=True, conf=0.07)
    print("\n✅ PROCESS COMPLETE! Results are in 'runs/detect/predict' folder.")
else:
    print("❌ Model file not found.")


# ========================================
# VISUALIZE RESULTS
# ========================================

import matplotlib.pyplot as plt
import cv2

for result in results:
    res_plotted = result.plot()
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()


# ========================================
# MODEL TRAINING
# ========================================

!pip install ultralytics
from ultralytics import YOLO

model = YOLO('/content/YOLO11m-Car-Damage-Detector/trained.pt')

model.train(
    data="/content/Car-Damage-Detection-1/data.yaml",
    epochs=35,
    imgsz=320,
    batch=4,
    workers=8,
    cache=True,
    exist_ok=True,
    optimizer='SGD',
    amp=True,
    lr0=0.01,
    project='car_damage_project',
    name='experiment_v1'
)


# ========================================
# COST ESTIMATION ANALYSIS
# ========================================

import cv2
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load trained model
model_path = '/content/runs/detect/car_damage_project/experiment_v1/weights/best.pt'
new_model = YOLO(model_path)

# Damage type pricing dictionary (in USD)
price_dictionary = {
    'doorouter-dent': 150.0,
    'fender-dent': 120.0,
    'front-bumper-dent': 200.0,
    'Headlight-Damage': 350.0,
    'Front-Windscreen-Damage': 500.0,
    'doorouter-scratch': 50.0,
    'bonnet-dent': 250.0,
    'rear-bumper-dent': 180.0,
    'default': 100.0
}

# Get all image paths
image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
    image_paths.extend(glob.glob(f'/content/damaged_pics/{ext}'))

cost_list = []

# Process each image
for path in image_paths:
    filename = os.path.basename(path)
    results = new_model.predict(source=path, conf=0.05, save=False, verbose=False)
    result = results[0]
    img = result.orig_img.copy()
    
    damage_count = len(result.boxes)
    
    if damage_count == 0:
        cost_list.append({
            "Image": filename,
            "Damage Count": 0,
            "Detected Damage": "No Damage Detected",
            "Cost ($)": 0.0
        })
        continue
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = new_model.names[cls_id]
        cost = price_dictionary.get(class_name, price_dictionary['default'])
        
        cost_list.append({
            "Image": filename,
            "Damage Count": damage_count,
            "Detected Damage": class_name,
            "Cost ($)": cost
        })
        
        # Draw bounding box and label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{class_name}: ${cost}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display annotated image
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{filename} - ({damage_count} Damages)")
    plt.axis('off')
    plt.show()

# Create detailed report
df = pd.DataFrame(cost_list)
final_report = df.groupby("Image").agg(
    Damage_Count=("Damage Count", "first"), 
    Damage_Details=("Detected Damage", lambda x: ", ".join(x)),
    Total_Cost_USD=("Cost ($)", "sum")
).reset_index()

print("\n--- DETAILED VEHICLE DAMAGE REPORT ---")
display(final_report)


# ========================================
# MAIN VEHICLE FOCUS DETECTION
# ========================================

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Load models
damage_model = YOLO('/content/YOLO11m-Car-Damage-Detector/trained.pt')

# Try to load custom vehicle model, fallback to YOLOv8n
try:
    vehicle_model = YOLO('/content/runs/detect/car_damage_project/experiment_v1/weights/best.pt')
    print("✅ Roboflow vehicle model loaded")
except:
    vehicle_model = YOLO('yolov8n.pt')
    print("✅ YOLOv8n vehicle model loaded")

# Image folder
image_folder = '/content/damaged_pics'

image_list = list(Path(image_folder).glob('*.jpg')) + list(Path(image_folder).glob('*.jpeg'))

print(f"\n📂 Folder: {image_folder}")
print(f"🖼️ {len(image_list)} images will be processed\n")
print("="*80)

for idx, image_file in enumerate(image_list, 1):
    print(f"\n🔍 [{idx}/{len(image_list)}] {image_file.name}")
    print("-"*80)
    
    # Load image
    image = cv2.imread(str(image_file))
    if image is None:
        print("❌ Could not read image\n")
        continue
    
    h, w = image.shape[:2]
    image_draw = image.copy()
    
    # DETECT VEHICLES
    vehicle_result = vehicle_model.predict(str(image_file), conf=0.10, verbose=False)
    
    vehicle_count = len(vehicle_result[0].boxes)
    
    if vehicle_count == 0:
        print("⚠️ No vehicle detected\n")
        continue
    
    print(f"✅ {vehicle_count} vehicle(s) detected")
    
    # FIND MAIN VEHICLE (largest one)
    largest_box = max(vehicle_result[0].boxes,
                      key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
    
    x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
    
    print(f"🎯 Main vehicle selected (size: {x2-x1}x{y2-y1} pixels)")
    
    # Draw main vehicle box (BLUE - thick)
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.putText(image_draw, 'MAIN VEHICLE', (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    # Mark other vehicles with light gray
    for box in vehicle_result[0].boxes:
        if box != largest_box:
            ox1, oy1, ox2, oy2 = map(int, box.xyxy[0])
            cv2.rectangle(image_draw, (ox1, oy1), (ox2, oy2), (128, 128, 128), 2)
            cv2.putText(image_draw, 'other', (ox1, oy1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    
    # Crop vehicle region
    vehicle_crop = image[y1:y2, x1:x2]
    
    if vehicle_crop.size == 0:
        print("❌ Invalid vehicle region\n")
        continue
    
    # Save temporarily
    temp_path = f'/content/temp_vehicle_main.jpg'
    cv2.imwrite(temp_path, vehicle_crop)
    
    # DAMAGE DETECTION (only on main vehicle)
    damage_result = damage_model.predict(temp_path, conf=0.10, verbose=False)
    damage_count = len(damage_result[0].boxes)
    
    print(f"🔴 {damage_count} damage(s) found on main vehicle (YOLO11 Model)")
    
    # Draw damages on original image
    damage_types = {}
    for damage_box in damage_result[0].boxes:
        hx1, hy1, hx2, hy2 = map(int, damage_box.xyxy[0])
        
        # Adjust coordinates to original image
        original_hx1 = x1 + hx1
        original_hy1 = y1 + hy1
        original_hx2 = x1 + hx2
        original_hy2 = y1 + hy2
        
        # Draw damage box (RED)
        cv2.rectangle(image_draw,
                    (original_hx1, original_hy1),
                    (original_hx2, original_hy2),
                    (0, 0, 255), 3)
        
        # Write damage type
        if damage_box.cls is not None:
            damage_type = damage_model.names[int(damage_box.cls)]
            
            # Count damage types
            if damage_type not in damage_types:
                damage_types[damage_type] = 0
            damage_types[damage_type] += 1
            
            cv2.putText(image_draw, damage_type,
                      (original_hx1, original_hy1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    os.remove(temp_path)
    
    # Print damage details
    if damage_types:
        print("\n📋 Damage Details:")
        for type_name, count in damage_types.items():
            print(f"  - {type_name}: {count} piece(s)")
    
    # Result
    print(f"\n📊 RESULT: 1 main vehicle, {damage_count} total damage(s)")
    
    # Display image
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
    
    # Create title
    title = f'{image_file.name} (YOLO11 Damage Model)\nMAIN VEHICLE: {damage_count} Damage(s)'
    if damage_types:
        title += '\n' + ', '.join([f'{k}: {v}' for k, v in damage_types.items()])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save result
    output_path = f'/content/yolo11_result_{image_file.stem}.jpg'
    cv2.imwrite(output_path, image_draw)
    print(f"✅ Saved: {output_path}")
    print("="*80)

print("\n✅ COMPLETED!")
print(f"\n📊 Models Used:")
print(f"  - Damage Detection: YOLO11m (trained.pt)")
print(f"  - Vehicle Detection: {'Roboflow' if 'best.pt' in str(vehicle_model.ckpt_path) else 'YOLOv8n'}")

