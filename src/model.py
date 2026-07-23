import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbeddedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A physics-informed convolution layer that processes data in both 
        the spatial and frequency domains simultaneously to resolve heavy blur.
        """
        super().__init__()
        # Spatial processing path
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Frequency processing path (1x1 complex convolution proxy)
        self.freq_conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.freq_conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 1. Spatial Domain Pass
        spatial_out = self.relu(self.bn(self.spatial_conv(x)))
        
        # 2. Compute 2D FFT over height and width dimensions
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Process real and imaginary components independently to remain numerically stable
        freq_real = self.freq_conv_real(x_fft.real)
        freq_imag = self.freq_conv_imag(x_fft.imag)
        
        # Reconstruct the complex tensor and invert back to spatial domain
        freq_complex = torch.complex(freq_real, freq_imag)
        frequency_out = torch.fft.irfft2(freq_complex, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        frequency_out = torch.tanh(frequency_out) # Keeps frequency features perfectly bounded
        
        # 3. Fuse Domains cleanly via addition
        return spatial_out + frequency_out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            FourierEmbeddedConv(in_channels, out_channels),
            FourierEmbeddedConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (Upsampling path + skip connections)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        b = self.bottleneck(x4)
        
        x = self.up1(b)
        diffY, diffX = x4.size(-2) - x.size(-2), x4.size(-1) - x.size(-1)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        diffY, diffX = x3.size(-2) - x.size(-2), x3.size(-1) - x.size(-1)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        diffY, diffX = x2.size(-2) - x.size(-2), x2.size(-1) - x.size(-1)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        diffY, diffX = x1.size(-2) - x.size(-2), x1.size(-1) - x.size(-1)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        return self.sigmoid(self.outc(x))

if __name__ == '__main__':
    print("Testing Fourier-Embedded U-Net architecture compilation...")
    model = UNet(in_channels=1, out_channels=1)
    test_input = torch.randn(2, 1, 600, 600)
    try:
        test_output = model(test_input)
        print("Model verification success!")
        print(f"Input: {test_input.shape} -> Output: {test_output.shape}")
    except Exception as e:
        print(f"Compilation Failed: {e}")
