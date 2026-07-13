import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        import torch.nn.functional as F
        
        # Encoder forward pass with feature preservation for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder forward pass with dynamic padding to match sizes exactly
        x = self.up1(b)
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        return self.sigmoid(self.outc(x))

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 1. Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # 2. Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # 3. Decoder (Upsampling path + skip connections)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)  # 512 (from up1) + 512 (from skip connection)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)   # 256 (from up2) + 256 (from skip connection)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)   # 128 (from up3) + 128 (from skip connection)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)     # 64 (from up4) + 64 (from skip connection)

        # Final output convolution layer mapping back to 1 channel with a Sigmoid (0.0 to 1.0 range)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder forward pass with feature preservation for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder forward pass with concatenated skip layers
        x = self.up1(b)
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        return self.sigmoid(self.outc(x))

if __name__ == '__main__':
    # Test script execution to verify network shape metrics
    print("Testing U-Net architecture structural compilation...")
    model = UNet(in_channels=1, out_channels=1)
    
    # Create a synthetic tensor mimicking your exact batch data shape [2, 1, 600, 600]
    test_input = torch.randn(2, 1, 600, 600)
    
    try:
        test_output = model(test_input)
        print("Model compilation verification success!")
        print(f"Input Shape:  {test_input.shape}")
        print(f"Output Shape: {test_output.shape}")  # Should match perfectly
    except Exception as e:
        print(f"Model Verification Failed: {e}")

