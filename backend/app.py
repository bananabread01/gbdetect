import os
import torch
import numpy as np
import cv2
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import Attention  
from heatmaps import visualize_attention, visualize_gradcam, visualize_gradcam_plus

app = Flask(__name__)
# CORS to allow specific origins
# CORS(app, origins=[
#     "http://localhost:5173",  # Local development
#     "https://bananabread01.github.io"  # GitHub Pages
# ])

CORS(app, origins="*")

# Configure CORS explicitly
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Add CORS headers to all responses
# @app.after_request
# def add_cors_headers(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
#     response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
#     return response

# # Handle OPTIONS requests explicitly
# @app.route('/predict', methods=['OPTIONS'])
# def handle_options():
#     response = make_response()
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
#     response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
#     return response

ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/MILprototype3.pth"
model = Attention(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Check if the uploaded file is in an allowed format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Function to preprocess image
def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = transform(image)
    return image  # Add batch dimension

def generate_attention_heatmap(image_tensor):
    heatmap = visualize_attention(model, image_tensor)
    return heatmap

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(image_tensor, predicted_class):
    heatmap = visualize_gradcam(model, image_tensor, predicted_class)
    return heatmap

def generate_gradcam_plus_heatmap(image_tensor, predicted_class):
    heatmap = visualize_gradcam_plus(model, image_tensor, predicted_class)
    return heatmap

# Function to encode heatmap as Base64
def encode_heatmap(heatmap):
    _, buffer = cv2.imencode(".png", heatmap)
    return base64.b64encode(buffer).decode("utf-8")

# Simple test endpoint
@app.route('/test', methods=['GET', 'OPTIONS'])
def test():
    if request.method == 'OPTIONS':
        return handle_options()
    return jsonify({'status': 'ok', 'message': 'API is working'})

# api endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print("Received predict request")
    print("Request headers:", request.headers)
    print("Request method:", request.method)
    print("Request files:", request.files)
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        print(f"Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid file format. Only JPG, JPEG, and PNG are allowed.'}), 400

    image_tensor = preprocess_image(file)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Run inference
    # with torch.no_grad():
    #     output = model(image_tensor)
    #     probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    #     predicted_class = np.argmax(probabilities)
    #     confidence = probabilities[predicted_class]

    model.eval()
    with torch.no_grad():
        output, _, _ = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

    attention_heatmap = generate_attention_heatmap(image_tensor)
    gradcam_heatmap = generate_gradcam_heatmap(image_tensor, predicted_class)
    gradcam2plus_heatmap = generate_gradcam_plus_heatmap(image_tensor, predicted_class)

    attention_encoded = encode_heatmap(attention_heatmap)
    gradcam_encoded = encode_heatmap(gradcam_heatmap)
    gradcam2plus_encoded = encode_heatmap(gradcam2plus_heatmap)

    # JSON response
    return jsonify({
        'predicted_class': int(predicted_class),
        'probabilities': probabilities.tolist(),
        'confidence': float(confidence),
        'attention_heatmap': attention_encoded,
        'gradcam_heatmap': gradcam_encoded,
        'gradcam2plus_heatmap': gradcam2plus_encoded
        
    })
if __name__ == '__main__':
    print("Starting Flask app with CORS enabled for all origins (*)")
    app.run(debug=True)