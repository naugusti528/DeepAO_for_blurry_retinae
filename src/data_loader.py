# Scripts to load and augment image batches

import os
import csv
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class RetinalDeblurDataset(Dataset):
    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.samples = []
        
        manifest_path = os.path.join(processed_dir, 'manifest.csv')
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(os.path.dirname(processed_dir), 'manifest.csv')
            
        with open(manifest_path, mode='r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                if len(row) >= 2:
                    blurry_abs_path = row[-2]
                    raw_abs_path = row[-1]
                    self.samples.append((blurry_abs_path, raw_abs_path))
        print(f"Database Loaded! Handshaked {len(self.samples)} uncompromised pipeline paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blurry_path, raw_match_path = self.samples[idx]
        blurry_img = cv2.imread(blurry_path, cv2.IMREAD_UNCHANGED)
        color_clean = cv2.imread(raw_match_path)
        if color_clean is None:
            raise FileNotFoundError(f"Database sync failure: File missing at {raw_match_path}")
            
        green_clean = color_clean[:, :, 1]
        clean_512 = cv2.resize(green_clean, (512, 512), interpolation=cv2.INTER_AREA)
        gaussian_blur = cv2.GaussianBlur(clean_512, (5, 5), 1.0)
        clean_sharpened = cv2.addWeighted(clean_512, 1.5, gaussian_blur, -0.5, 0)
        blurry_tensor = torch.tensor(blurry_img, dtype=torch.float32).unsqueeze(0) / 255.0
        clean_tensor = torch.tensor(clean_sharpened, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return blurry_tensor, clean_tensor

def get_deblur_dataloader(raw_dir, processed_dir, batch_size=4, shuffle=True):
    dataset = RetinalDeblurDataset(raw_dir, processed_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
