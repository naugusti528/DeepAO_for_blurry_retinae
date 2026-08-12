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
        self.feature_weight = feature_weight
        self.edge_weight = edge_weight
        self.priority_weight = priority_weight
        
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
    def _to_rgb(self, x):
        return torch.cat([x, x, x], dim=1)

    def _get_laplacian_edges(self, x):
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, kernel, padding=1)

    def forward(self, predicted, target):
        device = predicted.device
        self.features = self.features.to(device)
        
        # 1. Multi-Scale Perceptual Feature Loss (VGG16)
        pred_rgb = (self._to_rgb(predicted) - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        target_rgb = (self._to_rgb(target) - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        
        p_f1 = self.features[:5](pred_rgb)
        t_f1 = self.features[:5](target_rgb)
        p_f2 = self.features[:10](pred_rgb)
        t_f2 = self.features[:10](target_rgb)
        perceptual_loss = torch.mean(self.mse(p_f1, t_f1)) + torch.mean(self.mse(p_f2, t_f2))
        
        # 2. Laplacian Boundary Edge Loss
        pred_edges = self._get_laplacian_edges(predicted)
        target_edges = self._get_laplacian_edges(target)
        edge_loss = torch.mean(self.mse(pred_edges, target_edges))
        
        # Anatomical Priority Weighting (Emphasizing Vessels & Optic Disc)
        target_gradient = torch.abs(target_edges)
        priority_mask = 1.0 + (3.0 * target_gradient) + (1.5 * target)
        weighted_pixel_error = self.mse(predicted, target) * priority_mask
        priority_loss = torch.mean(weighted_pixel_error)
        
        return (self.feature_weight * perceptual_loss) + (self.edge_weight * edge_loss) + (self.priority_weight * priority_loss)

def train_model():
    RAW_TRAIN = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TRAIN = 'data/processed/train'
    
    print("Loading 512x512 green-channel data loader...")
    train_loader = get_deblur_dataloader(RAW_TRAIN, PROCESSED_TRAIN, batch_size=1, shuffle=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Initializing standard U-Net on device: {device}")
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = AnatomicalPriorityLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    os.makedirs('models', exist_ok=True)
    epochs = 11
    
    print("\nStarting Anatomical-Prioritized Training from Scratch (512x512)...")
    for epoch in range(1, epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            torch.mps.empty_cache()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch}/10] Batch [{batch_idx + 1}] Loss: {loss.item():.4f}")
                
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"--> Finished Epoch [{epoch}/10] | Average Priority Loss: {avg_epoch_loss:.4f}")
        torch.save(model.state_dict(), f'models/unet_anatomical_epoch_{epoch}.pth')
        
    print("Training complete! Weights saved as models/unet_anatomical_epoch_10.pth")

if __name__ == '__main__':
    train_model()
