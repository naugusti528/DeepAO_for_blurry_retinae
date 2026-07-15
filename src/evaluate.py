import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import your custom model architecture
from src.model import UNet

def evaluate_model():
    print("Initializing U-Net Evaluation Pipeline...")
    
    # 1. Device configuration (Match training hardware)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Path allocations matching project architecture layout
    MODEL_WEIGHTS = 'models/unet_deblur_epoch_10.pth'
    RAW_TEST_DIR = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TEST_DIR = 'data/processed/train'
    OUTPUT_RESULTS_DIR = 'data/evaluation_outputs'
    
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
    
    # 3. Load the saved model weights
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS}. Make sure your epochs finished saving!")
        return
        
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("Successfully loaded trained U-Net model weights.")

    # 4. Grab a synthetically blurred test file to evaluate
    blurry_paths = sorted(glob.glob(os.path.join(PROCESSED_TEST_DIR, '*.jpg')))
    if not blurry_paths:
        print("Error: No blurred images found in processed folder. Run simulator.py first!")
        return
        
    # Grab the first blurry image available
    test_blurry_path = blurry_paths[0]
    filename = os.path.basename(test_blurry_path).replace('blurry_', '')
    
    # Locate its matching original clean ground truth file
    raw_search = glob.glob(os.path.join(RAW_TEST_DIR, f'**/{filename}'), recursive=True)
    if not raw_search:
        print(f"Error: Could not locate clean matching image for {filename}")
        return
    test_clean_path = raw_search[0]

    # 5. Load and process the images for neural network feed
    clean_img = cv2.imread(test_clean_path, cv2.IMREAD_GRAYSCALE)
    blurry_img = cv2.imread(test_blurry_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert blurry image to PyTorch tensor [Batch=1, Channel=1, H, W]
    input_tensor = torch.tensor(blurry_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # 6. Pass blurry image through the U-Net for inference
    print("Running deep model inference deblurring...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # Convert output tensor back to standard 8-bit NumPy image array
    deblurred_img = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 7. Calculate Official Medical Metric Benchmarks
    # Compare Blurry vs. Ground Truth
    psnr_before = psnr(clean_img, blurry_img)
    ssim_before = ssim(clean_img, blurry_img)
    
    # Compare Model Sharpened Output vs. Ground Truth
    psnr_after = psnr(clean_img, deblurred_img)
    ssim_after = ssim(clean_img, deblurred_img)

    print("\n================ EVALUATION METRICS ================")
    print(f"Original Blurry Image   -> PSNR: {psnr_before:.2f} dB | SSIM: {ssim_before:.4f}")
    print(f"U-Net Sharpened Output -> PSNR: {psnr_after:.2f} dB | SSIM: {ssim_after:.4f}")
    print("====================================================")

    # 8. Save Before/After Visual Side-by-Side Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (Clean)")
    plt.imshow(clean_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Input Blur\nSSIM: {ssim_before:.4f}")
    plt.imshow(blurry_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"U-Net Deblurred\nSSIM: {ssim_after:.4f}")
    plt.imshow(deblurred_img, cmap='gray')
    plt.axis('off')
    
    plot_output_path = os.path.join(OUTPUT_RESULTS_DIR, 'deblur_comparison.png')
    plt.savefig(plot_output_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisual comparison plot successfully saved on your Mac at: {plot_output_path}")

if __name__ == '__main__':
    evaluate_model()

