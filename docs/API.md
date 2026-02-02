# 

This API provides car damage detection using a deep learning model.
Users can upload a car image and receive detected damage types and bounding boxes.
## Base URL

http://localhost:8000
## Authentication

No authentication is required.
## Endpoints

| Method | Endpoint        | Description                  |
|------|-----------------|------------------------------|
| POST | /predict        | Detect car damages from image |
| GET  | /health         | API health check              |
## POST /predict

Detects damages on a car image.
curl -X POST http://localhost:8000/predict \
  -F "file=@car.jpg"
{
  "damages": [
    {
      "type": "scratch",
      "confidence": 0.92,
      "bbox": [120, 45, 300, 200]
    },
    {
      "type": "dent",
      "confidence": 0.88,
      "bbox": [400, 100, 520, 260]
    }
  ]
}
| Field | Type | Description |
|-----|-----|-------------|
| type | string | Damage type |
| confidence | float | Model confidence |
| bbox | array | Bounding box [x1, y1, x2, y2] |

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid image format"
}
{
  "error": "Model inference failed"
}
## Model Information

- Architecture: YOLOv8
- Framework: PyTorch
- Input size: 640x640
- Trained on: Custom car damage dataset
## Performance

- Average inference time: ~120ms (GPU)
- mAP@0.5: 0.87
## Notes

- Works best with clear daylight images
- Not optimized for motorcycles
