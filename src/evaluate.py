import os
import csv
import cv2
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from src.model import UNet

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio cleanly using native NumPy mathematics."""
    mse_val = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def evaluate_model():
    print("Initializing U-Net Anatomical Loss Evaluation Pipeline...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    MODEL_WEIGHTS = 'models/unet_anatomical_epoch_7.pth'
    PROCESSED_TEST_DIR = 'data/processed/test'
    OUTPUT_DIR = 'data/evaluation_outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Weights file not found at {MODEL_WEIGHTS}. Run train.py first.")
        return
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("Successfully loaded trained U-Net model weights.")

    manifest_path = os.path.join(PROCESSED_TEST_DIR, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"Error: Missing test manifest file at {manifest_path}. Run simulator.py first.")
        return
        
    with open(manifest_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        first_row = next(reader)
        idx_num, blurry_abs_path, raw_abs_path = first_row

    blurry_img = cv2.imread(blurry_abs_path, cv2.IMREAD_UNCHANGED)
    color_clean = cv2.imread(raw_abs_path)
    if blurry_img is None or color_clean is None:
        print("Error: Failed to read file targets from manifest memory address links.")
        return
    
    green_clean = color_clean[:, :, 1]
    clean_512 = cv2.resize(green_clean, (512, 512), interpolation=cv2.INTER_AREA)
    inputs = torch.tensor(blurry_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        outputs = model(inputs)
    
    # Convert tensor array back to standard pixel map format
    deblurred_img = (outputs.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    
    # Calculate structural and peak noise benchmark metrics
    current_ssim = ssim(clean_512, deblurred_img, data_range=255)
    current_psnr = calculate_psnr(clean_512, deblurred_img)
    print(f"\nEvaluation Complete! Target Index Asset Slot: {idx_num}")
    print(f"System SSIM: {current_ssim:.4f} | System PSNR: {current_psnr:.2f} dB")
    
    # Three-panel validation comparison plot using a true green colormap
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(clean_512, cmap='Greens_r')
    ax1.set_title("Ground Truth (Clean Green)")
    ax1.axis('off')
    
    ax2.imshow(blurry_img, cmap='Greens_r')
    ax2.set_title("Input Blur (+ Gaussian Noise)")
    ax2.axis('off')
    
    ax3.imshow(deblurred_img, cmap='Greens_r')
    ax3.set_title(f"U-Net Deblurred\nSSIM: {current_ssim:.4f} | PSNR: {current_psnr:.2f}dB")
    ax3.axis('off')
    
    plt.tight_layout()
    output_plot_path = os.path.join(OUTPUT_DIR, 'deblur_comparison.png')
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Comparison visual plot successfully saved to: {output_plot_path}")

if __name__ == '__main__':
    evaluate_model()
