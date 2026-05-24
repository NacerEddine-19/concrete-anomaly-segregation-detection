"""
Model architectures — aligned 1-to-1 with Concrete_Anomaly_Detection_v2.ipynb.

  UNet          : pixel-level binary segmentation (loaded from best_unet.pth)
  DoubleConv    : shared building block used by UNet
  SimpleCNNSeg  : lightweight alternative (not used in the app pipeline but kept
                  for compatibility if the user switches seg_model_type to 'cnn')
  build_classifier : ResNet-18 with the custom 4-class dropout head from the notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ══════════════════════════════════════════════════════════════════════════════
#  U-NET  (segmentation backbone — best_unet.pth)
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks (shared by U-Net)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Classic U-Net for binary pixel-level segmentation.
    Symmetric encoder–decoder with skip connections and bilinear upsampling.
    Input  : (B, 3, H, W) — normalised RGB
    Output : (B, 1, H, W) — pixel probabilities [0, 1]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2, 2)

        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x     = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final_conv(x))


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT CNN SEGMENTATION  (alternative backend, not used by default)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCNNSeg(nn.Module):
    """Lightweight encoder–decoder segmentation model (no skip connections)."""
    def __init__(self):
        super().__init__()
        self.enc1  = nn.Sequential(nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3  = nn.Sequential(nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),   nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),   nn.ReLU(inplace=True))
        self.out  = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d1 = self.dec1(self.up1(b))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))
        return torch.sigmoid(self.out(d3))


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER  (ResNet-18 + custom 4-class dropout head)
# ══════════════════════════════════════════════════════════════════════════════

def build_classifier(device: torch.device, num_classes: int = 4) -> nn.Module:
    """
    ResNet-18 with the exact same 4-class dropout head used during training
    (from get_classifier_architecture in the notebook).
    """
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return m.to(device)
