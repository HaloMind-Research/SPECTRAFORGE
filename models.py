import torch
import torch.nn as nn
from torchvision import models

class SPECTRAFORGE(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        def build_backbone():
            b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(b.children())[:-1])
        
        self.spatial = build_backbone()
        self.freq = build_backbone()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 1)
        )

    def forward(self, sp, fq):
        features = torch.cat([self.spatial(sp), self.freq(fq)], dim=1)
        return self.head(features).squeeze(1)

class EffNetBaseline(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(1280, 1))
        
    def forward(self, x):
        return self.model(x).squeeze(1)
