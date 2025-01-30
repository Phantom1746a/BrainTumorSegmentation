import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        # Encoder
        self.d1 = Downsample(in_channels, 64)
        self.d2 = Downsample(64, 128)
        self.d3 = Downsample(128, 256)
        self.d4 = Downsample(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.u1 = Upsample(1024, 512)
        self.u2 = Upsample(512, 256)
        self.u3 = Upsample(256, 128)
        self.u4 = Upsample(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1, p1 = self.d1(x)
        c2, p2 = self.d2(p1)
        c3, p3 = self.d3(p2)
        c4, p4 = self.d4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u1 = self.u1(b, c4)
        u2 = self.u2(u1, c3)
        u3 = self.u3(u2, c2)
        u4 = self.u4(u3, c1)
        
        # Output
        out = self.out(u4)
        return self.sigmoid(out)