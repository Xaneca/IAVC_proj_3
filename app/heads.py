"""Cabeças sobre as features do backbone (layer3: 256 × 14 × 14)."""
import torch
import torch.nn as nn

EXPRESSIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
GENDERS = ["male", "female"]   # categorias do dataset (UTKFace), não identidade de género


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(3),   # mantém a disposição espacial (cima/baixo da cara)
        nn.Flatten(),
        nn.Linear(in_dim * 9, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim),
    )


class AgeGenderHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = _mlp(in_channels, 1 + len(GENDERS))

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out[:, 0], out[:, 1:]          # idade (regressão), logits do género


class ExpressionHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = _mlp(in_channels, len(EXPRESSIONS))

    def forward(self, x: torch.Tensor):
        return self.net(x)                    # logits da expressão


def age_to_range(age: float) -> str:
    low = int(age // 10 * 10)
    return f"{low}-{low + 9}"
