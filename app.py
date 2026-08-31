# full stack backend file

import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from src.model import UNet

app = Flask(__name__)
UPLOAD_FOLDER = 'data/web_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
weights_path = 'models/unet_anatomical_epoch_1.pth'

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"--> [SUCCESS] Streamlined Engine loaded weights from: {weights_path}")
else:
    raise FileNotFoundError(f"Fatal: Premium model checkpoint missing at: {weights_path}")
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    input_path = os.path.join(UPLOAD_FOLDER, 'input_latest.jpg')
    file.save(input_path)
    
    color_img = cv2.imread(input_path)
    if color_img is None:
        return jsonify({'error': 'Invalid format'}), 400
    # Extract clinical greenscale channel layer
    green_channel = color_img[:, :, 1]
    resized_input = cv2.resize(green_channel, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'input_greenscale.jpg'), resized_input)
    tensor_input = torch.tensor(resized_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        model_output = model(tensor_input)
    deblurred_array = (model_output.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    # filter mapping localized pixel intensities to recover blood vessel boundaries
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_output = clahe.apply(deblurred_array)
    
    input_green_file = cv2.merge([
        np.zeros_like(resized_input), # Blue = 0
        resized_input,                # Green = Data
        np.zeros_like(resized_input)  # Red = 0
    ])
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'input_greenscale.jpg'), input_green_file)
    output_green_file = cv2.merge([
        np.zeros_like(enhanced_output), # Blue = 0
        enhanced_output,                # Green = Data
        np.zeros_like(enhanced_output)  # Red = 0
    ])
    output_path = os.path.join(UPLOAD_FOLDER, 'output_deblurred.jpg')
    cv2.imwrite(output_path, output_green_file)
    
    return jsonify({
        'input_url': '/uploads/input_greenscale.jpg',
        'output_url': '/uploads/output_deblurred.jpg'
    })

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5001, debug=True)

