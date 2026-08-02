import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from src.model import UNet
from src.data_loader import get_deblur_dataloader

class MedicalPerceptualLoss(nn.Module):
    def __init__(self, feature_weight=0.2, edge_weight=0.8):
        super(MedicalPerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.feature_weight = feature_weight
        self.edge_weight = edge_weight
        
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
    def _to_rgb(self, x):
        return torch.cat([x, x, x], dim=1)

    def _get_laplacian_edges(self,x):
        #this will penalize blurry blood vessels and sharpen smaller features
        kernel = torch.tensor([[0,1,0], [1,-4,1], [0,1,0]], dtype=torch.float32).view(1,1,3,3).to(x.device)
        return F.conv2d(x, kernel, padding=1)

    def forward(self, predicted, target):
        device = predicted.device
        self.features = self.features.to(device)
        
        pred_rgb = (self._to_rgb(predicted) - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        target_rgb = (self._to_rgb(target) - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

        p_f1 = self.features[:5](pred_rgb)
        t_f1 = self.features[:5](target_rgb)
        p_f2 = self.features[:10](pred_rgb)
        t_f2 = self.features[:10](target_rgb)
        
        perceptual_loss = self.mse(p_f1,t_f1) + self.mse(p_f2,t_f2)
        
        pred_edges = self._get_laplacian_edges(predicted)
        target_edges = self._get_laplacian_edges(target)
        edge_loss = self.mse(pred_edges, target_edges)
        
        return (0.3 * perceptual_loss) + (0.7 * edge_loss)
        
def train_model():
    print("Initializing U-Net training pipeline with Perceptual Loss...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using processing hardware: {device}")

    RAW_TRAIN = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TRAIN = 'data/processed/train'
    
    train_loader = get_deblur_dataloader(RAW_TRAIN, PROCESSED_TRAIN, batch_size=1, shuffle=True)
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    #checkpoint_path = 'models/unet_redo_edge_epoch_10.pth'
    #if os.path.exists(checkpoint_path):
    #    print(f"Loading existing weights from {checkpoint_path}...")
    #    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #else:
    #    print(f"{checkpoint_path} not found")
    
    criterion = MedicalPerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 20
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            torch.mps.empty_cache()            

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Perceptual Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{epochs}] Completed | Average System Loss: {epoch_loss:.4f}\n")
        
        torch.save(model.state_dict(), f'models/unet_redo_edge_epoch_{epoch}.pth')

    print("Training pipeline successfully finished! Perceptual weights saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()
