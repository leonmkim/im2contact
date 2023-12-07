import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/milesial/Pytorch-UNet

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, feature_dim=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_dim = feature_dim

        self.inc = DoubleConv(n_channels, self.feature_dim)
        self.down1 = Down(self.feature_dim, self.feature_dim*2)
        self.down2 = Down(self.feature_dim*2, self.feature_dim*4)
        self.down3 = Down(self.feature_dim*4, self.feature_dim*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.feature_dim*8, (self.feature_dim*16) // factor)
        self.up1 = Up(self.feature_dim*16, (self.feature_dim*8) // factor, bilinear)
        self.up2 = Up(self.feature_dim*8, (self.feature_dim*4) // factor, bilinear)
        self.up3 = Up(self.feature_dim*4, (self.feature_dim*2) // factor, bilinear)
        self.up4 = Up(self.feature_dim*2, self.feature_dim, bilinear)
        self.outc = OutConv(self.feature_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel=3, pad=1): #
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        # changed from single conv to double conv after jose's suggestion about batchnorm
        mid_channels = (in_channels + out_channels)//2
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.silu(x)
        x = self.conv2(x)
        return x
        