import os
import glob
import cv2
import numpy as np
import csv

def simulate_optical_distortion():
    splits = ['train', 'test']
    
    for split in splits:
        INPUT_DIR = f'data/raw/extracted_images/augmented_resized_V2/{split}'
        OUTPUT_DIR = f'data/processed/{split}'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*.jpg'))) + sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg')))
        if not image_paths:
            print(f"Warning: No raw images found for split: {split}. Skipping.")
            continue

        print(f"Found {len(image_paths)} source files in [{split}]. Processing...")
        
        manifest_path = os.path.join(OUTPUT_DIR, 'manifest.csv')
        with open(manifest_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'blurry_absolute_path', 'raw_absolute_path']) # Header
            
            for idx, path in enumerate(image_paths):
                color_img = cv2.imread(path)
                if color_img is None:
                    continue
                green_img = color_img[:, :, 1]
                clean_512 = cv2.resize(green_img, (512, 512), interpolation=cv2.INTER_AREA)
                
                blurry_img = cv2.GaussianBlur(clean_512, (15, 15), 3.0)
                row, col = blurry_img.shape
                gauss_noise = np.random.normal(0, 5, (row, col)).astype(np.float32)
                noisy_blurry = np.clip(blurry_img.astype(np.float32) + gauss_noise, 0, 255).astype(np.uint8)
                blurry_filename = f'blurry_{idx}.jpg'
                blurry_abs_path = os.path.abspath(os.path.join(OUTPUT_DIR, blurry_filename))
                cv2.imwrite(blurry_abs_path, noisy_blurry)
                writer.writerow([idx, blurry_abs_path, os.path.abspath(path)])
                
                if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                    print(f"[{split}] Baked and Index-Mapped [{idx + 1}/{len(image_paths)}] frames.")
                    
    print("Data simulation pipeline successfully completed across all splits!")

if __name__ == '__main__':
    simulate_optical_distortion()
