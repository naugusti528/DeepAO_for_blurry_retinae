import cv2
import os
import glob
import numpy as np

RAW_TRAIN_DIR = 'data/raw/extracted_images/augmented_resized_V2/train'
OUTPUT_DIR = 'data/synthetic_blurred_train'

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob(os.path.join(RAW_TRAIN_DIR, '**/*.jpg'), recursive=True)

print(f"Found {len(image_paths)} clean ground-truth images.")
print("Adding synthetic blur with Gaussian noise application...")

for count, input_path in enumerate(image_paths[:50]):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue

    # Normalize image pixel values to 0.0 - 1.0 for accurate float math
    image_normalized = image.astype(np.float32) / 255.0

    # 4. Apply Gaussian Blur (Simulating camera defocus blur 'k')
    # 15x15 kernel size creates a noticeable blur. Must be an odd number.
    kernel_size = 15
    sigma = 3.0
    blurred_image = cv2.GaussianBlur(image_normalized, (kernel_size, kernel_size), sigma)

    # 5. Add Gaussian Noise (Simulating camera sensor noise 'n')
    noise_sigma = 0.02  # Higher numbers make the image grainier
    gaussian_noise = np.random.normal(0.0, noise_sigma, blurred_image.shape)
    
    # 6. Mathematical implementation: y = k*x + n
    synthetic_image = blurred_image + gaussian_noise
    synthetic_image = np.clip(synthetic_image, 0.0, 1.0) # Lock pixels within valid bounds

    # 7. Convert back to standard 8-bit image format and save
    filename = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, f"blurry_{filename}")
    cv2.imwrite(output_path, (synthetic_image * 255).astype(np.uint8))

    if (count + 1) % 10 == 0:
        print(f"Processed {count + 1}/50 images...")

print(f"\nSuccess! Processed images are saved on your computer at: {OUTPUT_DIR}")
