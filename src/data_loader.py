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
        print(f"Successfully loaded {len(self.blurry_paths)} processed blurry source references.")
        
    def __len__(self):
        return len(self.blurry_paths)

    def __getitem__(self, idx):
        blurry_path = self.blurry_paths[idx]
        filename = os.path.basename(blurry_path).replace('blurry_', '')
        if '-600' in filename:
            filename = filename.replace('-600', '')
        
        raw_match_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(raw_match_path):
            raise FileNotFoundError(f"File does not exist at: {raw_match_path}")
        
        blurry_img = cv2.imread(blurry_path, cv2.IMREAD_UNCHANGED)
        color_clean = cv2.imread(raw_match_path)
        if color_clean is None:
            raise FileNotFoundError(f"Failed to read clean target path image: {raw_match_path}")
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
