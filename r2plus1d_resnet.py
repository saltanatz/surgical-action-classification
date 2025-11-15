# r2plus1d_resnet.py

from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class ResNetR2Plus1D(nn.Module):
    """
    R(2+1)D-18 backbone for video classification.

    Inputs:
        x: Tensor of shape (B, T, C, H, W),
           where T = num_segments (e.g., 16).

    Args:
        num_classes: number of output classes.
        num_segments: number of frames per clip.
        freeze_until: freeze earlier blocks up to this name ("layer2", "layer3", etc.) or "none".
        pretrained: whether to use Kinetics-400 pretrained weights.
    """

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 16,
        freeze_until: str = "layer2",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments

        if pretrained:
            base = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        else:
            base = r2plus1d_18(weights=None)

        # Replace classifier with identity so we can attach our own head
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.base = base
        self.cls = nn.Linear(in_features, num_classes)

        # Optionally freeze early layers
        self._freeze_backbone(freeze_until)

    def _freeze_backbone(self, freeze_until: str) -> None:
        if freeze_until is None or freeze_until.lower() == "none":
            return

        # Order is similar to ResNet
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until not in freeze_order:
            return

        idx = freeze_order.index(freeze_until)
        names_to_freeze = freeze_order[: idx + 1]

        for name in names_to_freeze:
            # Some torchvision video models (VideoResNet) expose a `stem`
            # module instead of top-level `conv1`/`bn1` attributes.
            if name in ("conv1", "bn1") and hasattr(self.base, "stem"):
                for p in self.base.stem.parameters():
                    p.requires_grad = False
            elif hasattr(self.base, name):
                for p in getattr(self.base, name).parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) -> (B, C, T, H, W) for torchvision video models.
        """
        assert x.dim() == 5, f"Expected 5D tensor (B, T, C, H, W), got {x.shape}"
        b, t, c, h, w = x.shape

        # Optionally sanity-check temporal length
        # if t != self.num_segments:
        #     raise ValueError(f"Expected {self.num_segments} frames, got {t}")

        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        feats = self.base(x)  # (B, in_features) because base.fc = Identity
        logits = self.cls(feats)
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


def group_params_r2plus1d(
    model: ResNetR2Plus1D,
    base_lr: float,
) -> List[Dict[str, Any]]:
    """
    Parameter groups with layer-wise learning-rate multipliers,
    mirroring the scheme in TSM ResNet.

    Returns:
        List of param groups suitable for torch.optim optimizer.
    """
    groups: List[Dict[str, Any]] = []

    # stem: conv1 + bn1 or the `stem` module used by VideoResNet
    stem_params = []
    if hasattr(model.base, "conv1") and hasattr(model.base, "bn1"):
        stem_params = list(model.base.conv1.parameters()) + list(model.base.bn1.parameters())
    elif hasattr(model.base, "stem"):
        # collect all parameters from the stem sequential module
        stem_params = list(model.base.stem.parameters())

    stem_params = [p for p in stem_params if p.requires_grad]
    if stem_params:
        groups.append({"params": stem_params, "lr": base_lr * 0.1})

    # ResNet-like stages
    for name, lr_mult in [
        ("layer1", 0.1),
        ("layer2", 0.2),
        ("layer3", 0.5),
        ("layer4", 1.0),
    ]:
        if hasattr(model.base, name):
            params = [p for p in getattr(model.base, name).parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": base_lr * lr_mult})

    # Classification head: learn fastest
    head_params = [p for p in model.cls.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * 1.5})

    return groups
