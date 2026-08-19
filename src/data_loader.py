# Scripts to load and augment image batches

import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class RetinalDeblurDataset(Dataset):
    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.blurry_paths = sorted(glob.glob(os.path.join(processed_dir, 'blurry_*.jpg')))
        print("Indexing raw source dataset directory once for high-speed lookups...")
        self.raw_map = {}
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.raw_map[file] = os.path.join(root, file)
        print(f"Successfully mapped {len(self.raw_map)} clean target references.")

    def __len__(self):
        return len(self.blurry_paths)

    def __getitem__(self, idx):
        blurry_path = self.blurry_paths[idx]
        filename = os.path.basename(blurry_path).replace('blurry_', '')
        if filename not in self.raw_map:
            raise FileNotFoundError(f"Could not find matching clean image for: {filename}")
        raw_match_path = self.raw_map[filename]
        
        blurry_img = cv2.imread(blurry_path, cv2.IMREAD_UNCHANGED)
        color_clean = cv2.imread(raw_match_path)
        green_clean = color_clean[:, :, 1]
        clean_512 = cv2.resize(green_clean, (512, 512), interpolation=cv2.INTER_AREA)
        gaussian_blur = cv2.GaussianBlur(clean_512, (5, 5), 1.0)
        clean_sharpened = cv2.addWeighted(clean_512, 1.5, gaussian_blur, -0.5, 0)
        
        blurry_tensor = torch.tensor(blurry_img, dtype=torch.float32).unsqueeze(0) / 255.0
        clean_tensor = torch.tensor(clean_sharpened, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return blurry_tensor, clean_tensor

def get_deblur_dataloader(raw_dir, processed_dir, batch_size=4, shuffle=True):
    dataset = RetinalDeblurDataset(raw_dir, processed_dir)
    # 0 workers to avoid background thread synchronization deadlocks on Mac
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
