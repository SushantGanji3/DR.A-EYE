# API Documentation

## Overview
The Flask API provides endpoints for diabetic retinopathy prediction from retinal images.

## Base URL
- Local: `http://localhost:5000`
- Production: (configure as needed)

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict from Image Upload
**POST** `/predict`

Upload an image file for prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with key `image` containing the image file

**Response:**
```json
{
  "predicted_class": "No_DR",
  "predicted_label": 0,
  "confidence": 0.9876,
  "confidence_percentage": 98.76,
  "all_probabilities": {
    "No_DR": 0.9876,
    "Mild": 0.0089,
    "Moderate": 0.0023,
    "Severe": 0.0010,
    "Proliferate_DR": 0.0002
  },
  "message": "No diabetic retinopathy detected. Continue regular eye checkups.",
  "recommendation": "Continue regular eye examinations."
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "details": "Detailed error information"
}
```

### 3. Predict from URL
**POST** `/predict_url`

Predict from an image URL.

**Request:**
```json
{
  "url": "https://example.com/image.png"
}
```

**Response:** Same as `/predict` endpoint

## Class Labels

- `0`: No_DR - No diabetic retinopathy
- `1`: Mild - Mild diabetic retinopathy
- `2`: Moderate - Moderate diabetic retinopathy
- `3`: Severe - Severe diabetic retinopathy
- `4`: Proliferate_DR - Proliferative diabetic retinopathy

## Running the API

### Local Development
```bash
cd api
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -f api/Dockerfile -t dr-a-eye-api .
docker run -p 5000:5000 dr-a-eye-api
```

## Testing

### Using curl
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/image.png"
```

### Using Python
```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('image.png', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Notes

- The model must be trained first and saved to `model/best_resnet18.pth`
- Images are automatically resized to 224x224 pixels
- Supported formats: PNG, JPG, JPEG
- Maximum file size: 10MB (can be configured)

