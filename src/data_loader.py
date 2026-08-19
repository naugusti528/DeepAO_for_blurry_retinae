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

    def __len__(self):
        return len(self.blurry_paths)

    def __getitem__(self, idx):
        blurry_path = self.blurry_paths[idx]
        blurry_img = cv2.imread(blurry_path, cv2.IMREAD_UNCHANGED) # Already single channel green
        filename = os.path.basename(blurry_path).replace('blurry_', '')
        
        raw_search = glob.glob(os.path.join(self.raw_dir, f'**/{filename}'), recursive=True)
        if not raw_search:
            raise FileNotFoundError(f"Could not find matching clean image for: {filename}")
            
        color_clean = cv2.imread(raw_search[0])
        green_clean = color_clean[:, :, 1]
        clean_512 = cv2.resize(green_clean, (512, 512), interpolation=cv2.INTER_AREA)
        gaussian_blur = cv2.GaussianBlur(clean_512, (5, 5), 1.0)
        clean_sharpened = cv2.addWeighted(clean_512, 1.5, gaussian_blur, -0.5, 0)
        
        # Normalize to 0.0 - 1.0 and convert arrays to PyTorch Tensors [C, H, W]
        blurry_tensor = torch.tensor(blurry_img, dtype=torch.float32).unsqueeze(0) / 255.0
        clean_tensor = torch.tensor(clean_sharpened, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return blurry_tensor, clean_tensor

def get_deblur_dataloader(raw_dir, processed_dir, batch_size=1, shuffle=True):
    dataset = RetinalDeblurDataset(raw_dir, processed_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2 if os.name != 'nt' else 0)
