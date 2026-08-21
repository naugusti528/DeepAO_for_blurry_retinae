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
        if not blurry_paths:
            print(f"Warning: No blurry files found in {PROCESSED_DIR}. Skipping.")
            continue    
        print(f"Found {len(blurry_paths)} existing processed files in [{split}]. Mapping...")
        
        manifest_path = os.path.join(PROCESSED_DIR, 'manifest.csv')
        with open(manifest_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['blurry_file', 'raw_absolute_path']) # Header
            
            for path in blurry_paths:
                blurry_filename = os.path.basename(path)
                clean_name = blurry_filename.replace('blurry_', '')
                for tag in ['-600', 'GF-', '-GF', 'FA-', '-FA']:
                    clean_name = clean_name.replace(tag, '')
                raw_absolute_path = os.path.abspath(os.path.join(RAW_DIR, clean_name))
                writer.writerow([blurry_filename, raw_absolute_path])
                
        print(f"--> [SUCCESS] Mapped spreadsheet locked down at: {manifest_path}")

if __name__ == '__main__':
    generate_manifest_from_existing_files()
