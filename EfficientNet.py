"""
EfficientNet.py — EfficientNet-B4 classifier.

PDF Section 8: Model Architecture
Single-frame CNN-based classifier for 3-class deepfake detection.
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNet(nn.Module):
    """
    EfficientNet-B4 backbone + 3-class classification head.

    Input:  (B, 3, 224, 224)
    Output: (B, 3)  — logits for real / filter / deepfake
    """

    def __init__(self, num_classes=3, dropout=0.4, freeze_backbone=False, pretrained=True):
        super().__init__()

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = backbone.classifier[1].in_features  # 1792
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        """Extract feature embeddings (before classification head)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
