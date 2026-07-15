import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from src.model import UNet
from src.data_loader import get_deblur_dataloader

class MedicalPerceptualLoss(nn.Module):
    def __init__(self, feature_weight=0.85):
        super(MedicalPerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.feature_weight = feature_weight
        
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
    def _to_rgb(self, x):
        return torch.cat([x, x, x], dim=1)

    def forward(self, predicted, target):
        spatial_loss = self.mse(predicted, target)
        
        device = predicted.device
        self.features = self.features.to(device)
        
        pred_rgb = self._to_rgb(predicted)
        target_rgb = self._to_rgb(target)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        pred_rgb = (pred_rgb - mean) / std
        target_rgb = (target_rgb - mean) / std
        
        pred_features = self.features(pred_rgb)
        target_features = self.features(target_rgb)
        
        perceptual_loss = self.mse(pred_features, target_features)
        
        return (1.0 - self.feature_weight) * spatial_loss + self.feature_weight * perceptual_loss

def train_model():
    print("Initializing U-Net training pipeline with Perceptual Loss...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using processing hardware: {device}")

    RAW_TRAIN = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TRAIN = 'data/processed/train'
    
    train_loader = get_deblur_dataloader(RAW_TRAIN, PROCESSED_TRAIN, batch_size=2, shuffle=True)
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    checkpoint_path = 'models/unet_deblur_epoch_10.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading existing weights from {checkpoint_path} to begin fine-tuning...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No existing checkpoint found. Starting training from scratch.")
    
    criterion = MedicalPerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    epochs = 15  
    os.makedirs('models', exist_ok=True)

    print("\nStarting optimization loops...")
    for epoch in range(10, epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (blurry_batch, clean_batch) in enumerate(train_loader):
            inputs = blurry_batch.to(device)
            targets = clean_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Perceptual Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{epochs}] Completed | Average System Loss: {epoch_loss:.4f}\n")
        
        torch.save(model.state_dict(), f'models/unet_deblur_epoch_{epoch+1}.pth')

    print("Training pipeline successfully finished! Perceptual weights saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()
