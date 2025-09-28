"""
Model components for SCoDA (Self-contrastive + Distribution Alignment).
This file defines a lightweight ImageClassifier head over a ResNet-50 backbone.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ImageClassifier(nn.Module):
    """
    A simple classifier with:
      - ResNet-50 backbone (ImageNet pretrained)
      - Bottleneck MLP (Linear + ReLU + Dropout)
      - Linear classifier on top
    """

    def __init__(self, num_classes: int, bottleneck_dim: int = 256, finetune_backbone: bool = True):
        super().__init__()
        backbone = resnet18(pretrained="IMAGENET1K_V1")
        backbone.out_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone.out_features, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.finetune_backbone = finetune_backbone

    def get_parameters(self, base_lr: float = 1.0):
        """
        Parameter groups so the optimizer can use lower LR on the backbone
        and standard LR on new layers.
        """
        return [
            {"params": self.backbone.parameters(), "lr": (0.1 if self.finetune_backbone else 1.0) * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.classifier.parameters(), "lr": 1.0 * base_lr},
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feats_bottleneck: (B, bottleneck_dim)
            logits: (B, num_classes)
        """
        feats = self.backbone(x)
        feats_bottleneck = self.bottleneck(feats)
        logits = self.classifier(feats_bottleneck)
        return feats_bottleneck, logits


def info_nce_pairwise(out_1: torch.Tensor, out_2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Simplified InfoNCE for positive pairs (two views of the same sample) within a batch.
    This mirrors the approach in your digest's `calculate_contrastive_loss`, with clearer naming.
    """
    # L2 normalize
    z1 = F.normalize(out_1, dim=-1)
    z2 = F.normalize(out_2, dim=-1)

    # Concatenate views
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Similarities
    sim = torch.exp(z @ z.t() / temperature)  # (2B, 2B)

    # Remove self-similarities
    b = out_1.size(0)
    mask = (~torch.eye(2 * b, dtype=torch.bool, device=sim.device))
    sim = sim.masked_select(mask).view(2 * b, -1)

    # Positive similarities (diagonal pairs between z1 and z2)
    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)  # (2B,)

    loss = (-torch.log(pos / (sim.sum(dim=-1) + 1e-8))).mean()
    return loss
