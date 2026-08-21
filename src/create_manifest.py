# creating csv file to store images to make parsing easier

import os
import glob
import csv

def generate_manifest_from_existing_files():
    print("Indexing existing processed files to assemble manifest spreadsheet...")
    splits = ['train', 'test']
    
    for split in splits:
        RAW_DIR = f'data/raw/extracted_images/augmented_resized_V2/{split}'
        PROCESSED_DIR = f'data/processed/{split}'
        
        blurry_paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'blurry_*.jpg')))
        raw_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.jpg')))
        if not blurry_paths or not raw_files:
            print(f"Warning: Missing files in split: {split}. Skipping.")
            continue    
        print(f"Found {len(blurry_paths)} existing processed files in [{split}]. Mapping...")
        
        min_length = min(len(blurry_files), len(raw_files))
        print(f"[{split}] Aligning {min_length} index slots sequentially.")
        manifest_path = os.path.join(PROCESSED_DIR, 'manifest.csv')
        with open(manifest_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'blurry_absolute_path', 'raw_absolute_path'])
            
            for i in range(min_length):
                writer.writerow([i, os.path.abspath(blurry_files[i]), os.path.abspath(raw_files[i])])
                
        print(f"--> [SUCCESS] Mapped spreadsheet locked down at: {manifest_path}")

if __name__ == '__main__':
    generate_manifest_from_existing_files()
