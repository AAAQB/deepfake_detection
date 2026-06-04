import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class TemporalModel(nn.Module):
    def __init__(
        self,
        num_classes=3,
        hidden_size=512,
        num_layers=5,
        dropout=0.5,
        bidirectional=True,
        freeze_backbone=False
    ):
        super().__init__()

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b4(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten()
        )

        cnn_out_dim = backbone.classifier[1].in_features

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)       # (B*T, cnn_out_dim)
        features = features.view(B, T, -1)         # (B, T, cnn_out_dim)

        lstm_out, _ = self.lstm(features)          # (B, T, lstm_out_dim)
        last_out = lstm_out.mean(dim=1)            # (B, lstm_out_dim)

        out = self.classifier(last_out)            # (B, num_classes)
        return out

    def extract_sequence_embeddings(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        return features.view(B, T, -1)


if __name__ == "__main__":
    model = TemporalModel(num_classes=3, hidden_size=512, num_layers=2, bidirectional=True)
    dummy = torch.randn(2, 8, 3, 224, 224)   # batch=2, seq_len=8
    out = model(dummy)
