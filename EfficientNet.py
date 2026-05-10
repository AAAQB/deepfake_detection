import torch
import torch.nn as nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4, freeze_backbone=False):
        super().__init__()

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b4(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),#try nn.Softmax(dim=1) later
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) #use direct tensors
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = EfficientNetClassifier(num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
 