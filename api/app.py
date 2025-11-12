"""
Flask API for Diabetic Retinopathy prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import sys
from pathlib import Path
import traceback

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / 'model'))

from model import DiabeticRetinopathyModel

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
model = None
device = None
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Image transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


def load_model():
    """Load the trained model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = DiabeticRetinopathyModel(num_classes=5, pretrained=False)
    
    # Load trained weights
    model_path = Path(__file__).parent.parent / 'model' / 'best_resnet18.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")


def preprocess_image(image_bytes):
    """Preprocess image for inference"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict diabetic retinopathy from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            load_model()
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        predicted_class = int(predicted.item())
        confidence_score = float(confidence.item())
        class_name = class_names[predicted_class]
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_probs = {
            class_names[i]: float(all_probs[i]) 
            for i in range(len(class_names))
        }
        
        # Determine severity message
        severity_messages = {
            'No_DR': 'No diabetic retinopathy detected. Continue regular eye checkups.',
            'Mild': 'Mild diabetic retinopathy detected. Consult an ophthalmologist.',
            'Moderate': 'Moderate diabetic retinopathy detected. Immediate consultation recommended.',
            'Severe': 'Severe diabetic retinopathy detected. Urgent medical attention required.',
            'Proliferate_DR': 'Proliferative diabetic retinopathy detected. Emergency medical attention required.'
        }
        
        result = {
            'predicted_class': class_name,
            'predicted_label': predicted_class,
            'confidence': confidence_score,
            'confidence_percentage': round(confidence_score * 100, 2),
            'all_probabilities': class_probs,
            'message': severity_messages[class_name],
            'recommendation': 'Consult an ophthalmologist for professional diagnosis.' if class_name != 'No_DR' else 'Continue regular eye examinations.'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Predict from image URL"""
    try:
        data = request.get_json()
        
        if 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Download image from URL
        import requests
        from urllib.parse import urlparse
        
        response = requests.get(data['url'], timeout=10)
        response.raise_for_status()
        
        image_bytes = response.content
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Make prediction (same as /predict endpoint)
        if model is None:
            load_model()
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = int(predicted.item())
        confidence_score = float(confidence.item())
        class_name = class_names[predicted_class]
        
        all_probs = probabilities[0].cpu().numpy()
        class_probs = {
            class_names[i]: float(all_probs[i]) 
            for i in range(len(class_names))
        }
        
        severity_messages = {
            'No_DR': 'No diabetic retinopathy detected. Continue regular eye checkups.',
            'Mild': 'Mild diabetic retinopathy detected. Consult an ophthalmologist.',
            'Moderate': 'Moderate diabetic retinopathy detected. Immediate consultation recommended.',
            'Severe': 'Severe diabetic retinopathy detected. Urgent medical attention required.',
            'Proliferate_DR': 'Proliferative diabetic retinopathy detected. Emergency medical attention required.'
        }
        
        result = {
            'predicted_class': class_name,
            'predicted_label': predicted_class,
            'confidence': confidence_score,
            'confidence_percentage': round(confidence_score * 100, 2),
            'all_probabilities': class_probs,
            'message': severity_messages[class_name],
            'recommendation': 'Consult an ophthalmologist for professional diagnosis.' if class_name != 'No_DR' else 'Continue regular eye examinations.'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    print("Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will be loaded on first prediction request.")
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

