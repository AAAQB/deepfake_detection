

import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    """
    EfficientNet-B4 + multi-layer BiLSTM.

    Input:  (B, T, 3, 224, 224)  — T face crops per video
    Output: (B, 3)               — logits
    """

    def __init__(
        self,
        num_classes=3,
        hidden_size=512,
        num_layers=2,
        dropout=0.4,
        bidirectional=True,
        freeze_backbone=False,
        pretrained=True,
        seq_len=None,  # kept for backward compat, unused
    ):
        super().__init__()

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )

        cnn_out_dim = backbone.classifier[1].in_features  # 1792

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)          # (B*T, 1792)
        features = features.view(B, T, -1)            # (B, T, 1792)

        lstm_out, _ = self.lstm(features)             # (B, T, lstm_out_dim)

        # Mean pooling over the temporal dimension — more robust
        # than using only the last output for deepfake detection
        pooled = lstm_out.mean(dim=1)                 # (B, lstm_out_dim)

        out = self.classifier(pooled)                 # (B, num_classes)
        return out

    def extract_sequence_embeddings(self, x):
        """Extract per-frame feature embeddings before LSTM."""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        return features.view(B, T, -1)
