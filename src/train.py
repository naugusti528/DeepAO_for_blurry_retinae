import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import models
from src.model import UNet
from src.data_loader import get_deblur_dataloader

class AnatomicalPriorityLoss(nn.Module):
    def __init__(self, feature_weight=0.2, edge_weight=0.4, priority_weight=0.4):
        super(AnatomicalPriorityLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none') # Keep per-pixel tensor for dynamic masking
        self.edge_weight = edge_weight
        self.priority_weight = priority_weight

    def _get_laplacian_edges(self, x):
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, kernel, padding=1)

    def forward(self, predicted, target):
        pred_edges = self._get_laplacian_edges(predicted)
        target_edges = self._get_laplacian_edges(target)
        edge_loss = torch.mean(self.mse(pred_edges, target_edges))
        
        # Anatomical Priority Weighting (Emphasizing Vessels & Optic Disc)
        target_gradient = torch.abs(target_edges)
        priority_mask = 1.0 + (4.0 * target_gradient) + (2.0 * target)
        weighted_pixel_error = self.mse(predicted, target) * priority_mask
        priority_loss = torch.mean(weighted_pixel_error)
        
        return (self.edge_weight * edge_loss) + (self.priority_weight * priority_loss)

def train_model():
    RAW_TRAIN = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TRAIN = 'data/processed/train'
    
    print("Loading 512x512 green-channel data loader...")
    train_loader = get_deblur_dataloader(RAW_TRAIN, PROCESSED_TRAIN, batch_size=8, shuffle=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Initializing standard U-Net on device: {device}")
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = AnatomicalPriorityLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    os.makedirs('models', exist_ok=True)
    epochs = 21
    
    print("\nStarting Anatomical-Prioritized Training from Scratch (512x512)...")
    for epoch in range(1, epochs):
        model.train()
        running_loss = 0.0
        
        train_loader.dataset.samples = train_loader.dataset.samples
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
            del inputs, targets, outputs, loss
            torch.mps.empty_cache()
            if (batch_idx + 1) % 25 == 0:
                print(f"Epoch [{epoch}/10] | Batch [{batch_idx + 1}/250] | Current Step Loss: {running_loss / (batch_idx + 1):.4f}")
            if batch_idx >= 249: 
                break
                
        avg_epoch_loss = running_loss / 250
        print(f"--> Finished Epoch [{epoch}/10] | Average Priority Loss: {avg_epoch_loss:.4f}\n")
        torch.save(model.state_dict(), f'models/unet_anatomical_epoch_{epoch}.pth')
        
    print("Training complete! Weights saved as models/unet_anatomical_epoch_10.pth")

if __name__ == '__main__':
    train_model()
