import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Decoder
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.dec4 = nn.Conv2d(64, 3, kernel_size=1)
        # Max Pool
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Decoder
        d1 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)
        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.dec2(d2)
        d3 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(d3)
        # Output
        output = self.dec4(d3)
        return output
