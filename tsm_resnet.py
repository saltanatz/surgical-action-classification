# Accuracy 69%

from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM).

    Splits channels into three parts:
        - one part shifted forward in time
        - one part shifted backward
        - the rest unchanged

    This injects temporal information without adding extra parameters.
    """

    def __init__(self, num_segments: int, fold_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        batch_size, time, channels, height, width = x.shape

        fold = channels // self.fold_div
        if fold == 0:
            return x

        # three channel splits
        x1 = x[:, :, :fold, :, :]            # part to shift forward
        x2 = x[:, :, fold:2 * fold, :, :]    # part to shift backward
        x3 = x[:, :, 2 * fold:, :, :]        # part unchanged

        # shifts x1 forward in time (towards later frames)
        pad_right = torch.zeros(
            batch_size, 1, fold, height, width,
            device=x.device, dtype=x.dtype
        )
        x1_shifted = torch.cat([x1[:, 1:, :, :, :], pad_right], dim=1)

        # shifts x2 backward in time (towards earlier frames)
        pad_left = torch.zeros(
            batch_size, 1, fold, height, width,
            device=x.device, dtype=x.dtype
        )
        x2_shifted = torch.cat([pad_left, x2[:, :-1, :, :, :]], dim=1)

        # concatenate along channels
        out = torch.cat([x1_shifted, x2_shifted, x3], dim=2)
        return out

def insert_tsm_stages(
    backbone: nn.Module,
    num_segments: int,
    stages: List[str] = ("layer1", "layer2", "layer3", "layer4")
) -> nn.Module:
    """
    Wraps ResNet layers with TSM so temporal shifts happen before each spatial block.
    """

    class StageWrapper(nn.Module):
        def __init__(self, stage: nn.Module, n_seg: int) -> None:
            super().__init__()
            self.tsm = TemporalShift(n_seg)
            self.stage = stage
            self.num_segments = n_seg

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C, H, W)
            batch_size, time, channels, height, width = x.shape

            # apply temporal shift (still B, T, C, H, W)
            x = self.tsm(x)

            # flatten time into batch for 2D convs
            x = x.view(batch_size * time, channels, height, width)

            # run standard ResNet block
            x = self.stage(x)

            # restore (B, T, C, H, W)
            _, c2, h2, w2 = x.shape
            x = x.view(batch_size, time, c2, h2, w2)
            return x

    for name in stages:
        stage = getattr(backbone, name)
        wrapped = StageWrapper(stage, num_segments)
        setattr(backbone, name, wrapped)

    return backbone

class ResNetTSM(nn.Module):
    """
    ResNet18 backbone with TSM inserted into each ResNet stage.

    Args:
        num_classes: number of output classes.
        num_segments: number of frames per clip (temporal length).
        freeze_until: freeze earlier blocks up to this name ("layer2", "layer3", etc.) or "none".
        pretrained: whether to use ImageNet pretrained weights for ResNet18.
    """

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 16,
        freeze_until: str = "layer2",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments

        # Load ResNet18 backbone
        if pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=None)

        # Insert temporal shifts into the 4 main ResNet stages
        self.base = insert_tsm_stages(base, num_segments)

        # Optionally freeze early layers
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until != "none":
            for name, module in self.base.named_children():
                if name in freeze_order:
                    for p in module.parameters():
                        p.requires_grad = False
                if name == freeze_until:
                    # stop freezing after this block
                    pass

        # Replace FC with identity; add separate classification head
        feat_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.cls = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: (B, T, C, H, W)
        """
        batch_size, time, channels, height, width = x.shape

        # Collapse time into batch for the stem layers (conv1..maxpool)
        x = x.view(batch_size * time, channels, height, width)

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        # Bring time dimension back for TSM stages
        _, c1, h1, w1 = x.shape
        x = x.view(batch_size, time, c1, h1, w1)

        # ResNet stages with TSM wrapped
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        # Global average pooling: spatial then temporal
        batch_size, time, c2, h2, w2 = x.shape
        x = x.mean(dim=[3, 4])   # spatial GAP -> (B, T, C)
        x = x.mean(dim=1)        # temporal average -> (B, C)

        # Classifier head
        logits = self.cls(x)
        return logits

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = torch.sum(-true_dist * log_probs, dim=-1).mean()
        return loss


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # top1 accuracy
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    return acc


def group_params(model: ResNetTSM, base_lr: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []

    # stem: conv1 + bn1
    stem_params = list(model.base.conv1.parameters()) + list(model.base.bn1.parameters())
    stem_params = [p for p in stem_params if p.requires_grad]
    if stem_params:
        groups.append({"params": stem_params, "lr": base_lr * 0.1})

    # ResNet blocks
    for name, lr_mult in [
        ("layer1", 0.1),
        ("layer2", 0.2),
        ("layer3", 0.5),
        ("layer4", 1.0),
    ]:
        params = [p for p in getattr(model.base, name).parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * lr_mult})

    # Classification head: usually we want it to learn fastest
    head_params = [p for p in model.cls.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * 1.5})

    return groups

