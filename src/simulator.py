import os
import glob
import cv2
import numpy as np

def simulate_optical_distortion():
    splits = ['train', 'test']
    
    for split in splits:
        INPUT_DIR = f'data/raw/extracted_images/augmented_resized_V2/{split}'
        OUTPUT_DIR = f'data/processed/{split}'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '**/*.jpg'), recursive=True))
        if not image_paths:
            print(f"Warning: No raw images found for split: {split}. Skipping.")
            continue

        print(f"Found {len(image_paths)} source files in [{split}]. Processing...")
        
        for idx, path in enumerate(image_paths):
            color_img = cv2.imread(path)
            if color_img is None:
                continue
            # greenscale + 512x512 resizing
            green_img = color_img[:, :, 1]
            clean_512 = cv2.resize(green_img, (512, 512), interpolation=cv2.INTER_AREA)
            filename = os.path.basename(path)
            # 15x15 Gaussian Blur + noise
            blurry_img = cv2.GaussianBlur(clean_512, (15, 15), 3.0)
            row, col = blurry_img.shape
            mean = 0
            sigma = 5
            gauss_noise = np.random.normal(mean, sigma, (row, col)).astype(np.float32)
            noisy_blurry_img = np.clip(blurry_img.astype(np.float32) + gauss_noise, 0, 255).astype(np.uint8)
            
            output_path = os.path.join(OUTPUT_DIR, f'blurry_{filename}')
            cv2.imwrite(output_path, noisy_blurry_img)
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                print(f"[{split}] Processed [{idx + 1}/{len(image_paths)}] retinal images.")

    print("Data simulation pipeline successfully completed across all splits!")

if __name__ == '__main__':
    simulate_optical_distortion()
