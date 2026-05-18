from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class LegacyChessEvalCNN(nn.Module):
    def __init__(self, extra_features_dim: int = 8, dropout: float = 0.2, num_buckets: int = 0):
        super().__init__()
        self.extra_features_dim = extra_features_dim
        self.num_buckets = num_buckets
        self.backbone = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 8 * 8 + extra_features_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.bucket_head = nn.Linear(64, num_buckets) if num_buckets > 0 else None

    def forward(self, board: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone(board)
        x = torch.flatten(x, start_dim=1)
        if self.extra_features_dim:
            if extras is None:
                extras = torch.zeros((board.shape[0], self.extra_features_dim), device=board.device, dtype=board.dtype)
            x = torch.cat([x, extras], dim=1)
        hidden = self.head[:-1](x)
        score = self.head[-1](hidden).squeeze(-1)
        if self.bucket_head is None:
            return score
        return score, self.bucket_head(hidden)


class ChessEvalCNN(nn.Module):
    def __init__(
        self,
        extra_features_dim: int = 8,
        width: int = 192,
        depth: int = 8,
        dropout: float = 0.15,
        head_hidden: int = 512,
        num_buckets: int = 0,
    ):
        super().__init__()
        self.extra_features_dim = extra_features_dim
        self.num_buckets = num_buckets
        if width <= 0:
            raise ValueError("width must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        stem = [
            nn.Conv2d(12, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        ]
        blocks = [ResidualBlock(width, dropout=min(dropout, 0.1)) for _ in range(depth)]
        self.backbone = nn.Sequential(*stem, *blocks)
        self.head_body = nn.Sequential(
            nn.Linear(width * 8 * 8 + extra_features_dim, head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        self.regression_head = nn.Linear(head_hidden // 2, 1)
        self.bucket_head = nn.Linear(head_hidden // 2, num_buckets) if num_buckets > 0 else None

    def forward(self, board: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone(board)
        x = torch.flatten(x, start_dim=1)
        if self.extra_features_dim:
            if extras is None:
                extras = torch.zeros((board.shape[0], self.extra_features_dim), device=board.device, dtype=board.dtype)
            x = torch.cat([x, extras], dim=1)
        hidden = self.head_body(x)
        score = self.regression_head(hidden).squeeze(-1)
        if self.bucket_head is None:
            return score
        return score, self.bucket_head(hidden)


class TorchvisionRegressionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        extra_features_dim: int = 16,
        dropout: float = 0.15,
        head_hidden: int = 512,
        num_buckets: int = 0,
    ):
        super().__init__()
        self.extra_features_dim = extra_features_dim
        self.model_name = model_name
        self.num_buckets = num_buckets
        if model_name == "resnet18":
            backbone = models.resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            backbone.fc = nn.Identity()
        elif model_name == "resnet34":
            backbone = models.resnet34(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            backbone.fc = nn.Identity()
        elif model_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=None)
            first = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                12,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision model: {model_name}")
        self.backbone = backbone
        self.head_body = nn.Sequential(
            nn.Linear(feature_dim + extra_features_dim, head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, max(head_hidden // 2, 32)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        hidden_dim = max(head_hidden // 2, 32)
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.bucket_head = nn.Linear(hidden_dim, num_buckets) if num_buckets > 0 else None

    def forward(self, board: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(board)
        if self.extra_features_dim:
            if extras is None:
                extras = torch.zeros((board.shape[0], self.extra_features_dim), device=board.device, dtype=board.dtype)
            features = torch.cat([features, extras], dim=1)
        hidden = self.head_body(features)
        score = self.regression_head(hidden).squeeze(-1)
        if self.bucket_head is None:
            return score
        return score, self.bucket_head(hidden)


def create_model(
    model_name: str = "resnet18",
    extra_features_dim: int = 16,
    width: int = 192,
    depth: int = 8,
    dropout: float = 0.15,
    head_hidden: int = 512,
    num_buckets: int = 0,
) -> nn.Module:
    if model_name in {"resnet18", "resnet34", "efficientnet_b0"}:
        return TorchvisionRegressionModel(
            model_name=model_name,
            extra_features_dim=extra_features_dim,
            dropout=dropout,
            head_hidden=head_hidden,
            num_buckets=num_buckets,
        )
    if model_name in {"legacy_cnn", "cnn_v1"}:
        return LegacyChessEvalCNN(extra_features_dim=extra_features_dim, dropout=dropout, num_buckets=num_buckets)
    if model_name != "cnn":
        raise ValueError(f"Unsupported model_name '{model_name}'. Available: cnn, legacy_cnn")
    return ChessEvalCNN(
        extra_features_dim=extra_features_dim,
        width=width,
        depth=depth,
        dropout=dropout,
        head_hidden=head_hidden,
        num_buckets=num_buckets,
    )
