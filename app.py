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

# defining paths for all 7 available weight files
ensemble_weights = {
    'sliding_window_ep1': 'models/unet_anatomical_epoch_1.pth',
    'legacy_ep2': 'models/unet_anatomical_epoch_2.pth',
    'legacy_ep3': 'models/unet_anatomical_epoch_3.pth',
    'legacy_ep4': 'models/unet_anatomical_epoch_4.pth',
    'legacy_ep5': 'models/unet_anatomical_epoch_5.pth',
    'legacy_ep6': 'models/unet_anatomical_epoch_6.pth',
    'legacy_ep7': 'models/unet_anatomical_epoch_7.pth'
}

# ensemble model approach for using all weight files
ensemble_models = []

print("Initializing 7-Way Deep Learning Ensemble Engine...")
for key, path in ensemble_weights.items():
    if not os.path.exists(path):
        print(f"--> [CRITICAL GAP] File missing at path: {path}. Skipping checkpoint.")
        continue
        
    # Create a fresh, isolated model container instance
    model_instance = UNet(in_channels=1, out_channels=1).to(device)
    model_instance.load_state_dict(torch.load(path, map_location=device))
    model_instance.eval()
    
    ensemble_models.append(model_instance)
    print(f"--> [LOADED] Successfully mounted weight file: {key}")

if len(ensemble_models) == 0:
    raise FileNotFoundError("Fatal: The engine could not locate any of your .pth weight files in the models/ directory.")

print(f"\nSystem online: running consensus inference pool with {len(ensemble_models)} active checkpoints.")

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
    
    accumulated_outputs = None
    with torch.no_grad():
        output_sliding_window = ensemble_models[0](tensor_input)
        legacy_accumulation = None
        legacy_count = 0
        
        for model in ensemble_models[1:]:
            output_tensor = model(tensor_input)
            if legacy_accumulation is None:
                legacy_accumulation = output_tensor
            else:
                legacy_accumulation += output_tensor
            legacy_count += 1
        legacy_consensus = legacy_accumulation / float(legacy_count)
        
        # Biasing the math 85% to the generalized model and 15% to the legacy filter
        blended_output = (output_sliding_window * 0.85) + (legacy_consensus * 0.15)
        
    # reconstructing combined tensor array back to standard image bytes
    input_green_file = cv2.merge([
        np.zeros_like(resized_input), # Blue = 0
        resized_input,                # Green = Data
        np.zeros_like(resized_input)  # Red = 0
    ])
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'input_greenscale.jpg'), input_green_file)
    deblurred_array = (blended_output.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    output_green_file = cv2.merge([
        np.zeros_like(deblurred_array), # Blue = 0
        deblurred_array,                # Green = Data
        np.zeros_like(deblurred_array)  # Red = 0
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

