import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# Import the architectural components we just finished building
from src.model import UNet
from src.data_loader import get_deblur_dataloader

class HybridLoss(nn.Module):
    def __init__(self, ssim_weight=0.5):
        """
        Combines Mean Squared Error (MSE) with structural similarity metrics
        to better evaluate fine retinal features like blood vessels.
        """
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight

    def forward(self, predicted, target):
        mse_loss = self.mse(predicted, target)
        
        # Differentiable 1D structural metric approximation for stable training
        # We want to maximize similarity, which means minimizing (1 - similarity)
        loss_ssim = 1.0 - torch.mean(F.cosine_similarity(predicted, target, dim=2))
        
        return (1.0 - self.ssim_weight) * mse_loss + self.ssim_weight * loss_ssim

def train_model():
    print("Initializing U-Net training pipeline...")
    
    # 1. Device configuration (use Mac MPS acceleration if available, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using processing hardware: {device}")

    # 2. Map dataset directory splits
    RAW_TRAIN = 'data/raw/extracted_images/augmented_resized_V2/train'
    PROCESSED_TRAIN = 'data/processed/train'
    
    # 3. Initialize your verified PyTorch DataLoader
    train_loader = get_deblur_dataloader(RAW_TRAIN, PROCESSED_TRAIN, batch_size=2, shuffle=True)
    
    # 4. Instantiate the corrected U-Net model and map to device memory
    model = UNet(in_channels=1, out_channels=1).to(device)

    #adding these lines here after 5-epoch training on 115K images.
    #setting a checkpoint to prevent relearning of already learned features.
    checkpoint_path = 'models/unet_deblur_epoch_5.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading existing weights from {checkpoint_path} to continue fine-tuning...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 5. Define optimization parameters
    
    criterion = HybridLoss(ssim_weight=0.85)
    #prior to initial training on 115K images, the ssim weight was 0.5.
    #to improve deblurring on more minute features like blood vessels and clarity of optic disc

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    epochs = 10  # originally 5, now changing to 10. learning rate above also reduced by 10x
    os.makedirs('models', exist_ok=True)

    print("\nStarting optimization loops...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (blurry_batch, clean_batch) in enumerate(train_loader):
            # Move data tensors to current hardware acceleration unit
            inputs = blurry_batch.to(device)
            targets = clean_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward optimization pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Hybrid Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{epochs}] Completed | Average System Loss: {epoch_loss:.4f}\n")
        
        # Save structural checkpoint weights file locally
        torch.save(model.state_dict(), f'models/unet_deblur_epoch_{epoch+1}.pth')

    print("Training pipeline successfully finished! Weights saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()

