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
from model import MILModel  

app = Flask(__name__)
CORS(app)

ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "prototype4.pth"
model = MILModel(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#Check if the uploaded file is in an allowed format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Function to preprocess image
def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension

# Function to generate attention heatmap
def generate_attention_heatmap(image_tensor):
    model.eval()
    with torch.no_grad():
        att_map = model.get_attention_map(image_tensor)  # Get attention map
    att_map = att_map[0].cpu().numpy()
    
    # Resize heatmap to match original image size
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h, w, _ = img_np.shape
    att_map_resized = cv2.resize(att_map, (w, h))

    # Convert heatmap to color
    heatmap = cv2.applyColorMap(np.uint8(255 * att_map_resized), cv2.COLORMAP_JET)

    return heatmap

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(image_tensor, predicted_class):
    target_layer = None
    for module in reversed(list(model.base.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        return None

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Convert grayscale heatmap to color
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    
    return heatmap

# Function to encode heatmap as Base64
def encode_heatmap(heatmap):
    _, buffer = cv2.imencode(".png", heatmap)
    return base64.b64encode(buffer).decode("utf-8")

# api endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only JPG, JPEG, and PNG are allowed.'}), 400

    image_tensor = preprocess_image(file)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

    attention_heatmap = generate_attention_heatmap(image_tensor)
    gradcam_heatmap = generate_gradcam_heatmap(image_tensor, predicted_class)

    attention_encoded = encode_heatmap(attention_heatmap)
    gradcam_encoded = encode_heatmap(gradcam_heatmap)

    # JSON response
    return jsonify({
        'predicted_class': int(predicted_class),
        'probabilities': probabilities.tolist(),
        'confidence': float(confidence),
        'attention_heatmap': attention_encoded,
        'gradcam_heatmap': gradcam_encoded
        
    })
if __name__ == '__main__':
    app.run(debug=True)