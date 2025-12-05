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
    """

    def __init__(self, num_segments: int, fold_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        fold = c // self.fold_div
        if fold == 0:
            return x

        x1 = x[:, :, :fold, :, :]          # part to shift forward
        x2 = x[:, :, fold:2 * fold, :, :]  # part to shift backward
        x3 = x[:, :, 2 * fold:, :, :]      # unchanged

        # forward shift: content moves to later frames
        x1_shifted = torch.zeros_like(x1)
        x1_shifted[:, 1:, ...] = x1[:, :-1, ...]

        # backward shift: content moves to earlier frames
        x2_shifted = torch.zeros_like(x2)
        x2_shifted[:, :-1, ...] = x2[:, 1:, ...]

        out = torch.cat([x1_shifted, x2_shifted, x3], dim=2)
        return out


def insert_tsm_stages(
    backbone: nn.Module,
    num_segments: int,
    stages: List[str] = ("layer1", "layer2", "layer3", "layer4"),
) -> nn.Module:
    """
    Wrap ResNet stages with TSM so temporal shifts happen before each spatial stage.
    """

    class StageWrapper(nn.Module):
        def __init__(self, stage: nn.Module, n_seg: int) -> None:
            super().__init__()
            self.tsm = TemporalShift(n_seg)
            self.stage = stage
            self.num_segments = n_seg

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C, H, W)
            b, t, c, h, w = x.shape
            x = self.tsm(x)
            x = x.view(b * t, c, h, w)
            x = self.stage(x)
            _, c2, h2, w2 = x.shape
            x = x.view(b, t, c2, h2, w2)
            return x

    for name in stages:
        stage = getattr(backbone, name)
        setattr(backbone, name, StageWrapper(stage, num_segments))

    return backbone


class ResNetTSM(nn.Module):
    """
    ResNet18 backbone with TSM inserted into each main stage.

    Args:
        num_classes: number of output classes.
        num_segments: number of frames per clip (T).
        freeze_until: freeze earlier blocks up to this name
                      (e.g. "layer1", "layer2", "layer3") or "none".
        pretrained: if True, use ImageNet-pretrained ResNet18 (transfer learning).
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

        # ImageNet pretraining for transfer learning
        if pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=None)

        # Insert TSM wrappers into stages
        self.base = insert_tsm_stages(base, num_segments)

        # Optionally freeze early layers (for transfer learning)
        # freeze everything up to and including `freeze_until`
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until != "none":
            for name, module in self.base.named_children():
                if name in freeze_order:
                    for p in module.parameters():
                        p.requires_grad = False
                if name == freeze_until:
                    break

        # Replace original FC with identity; use our own classification head
        feat_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.cls = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape

        # Stem on (B*T, C, H, W)
        x = x.view(b * t, c, h, w)
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        # Back to (B, T, C, H, W) for TSM stages
        _, c1, h1, w1 = x.shape
        x = x.view(b, t, c1, h1, w1)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        # Global average pooling: spatial then temporal
        b, t, c2, h2, w2 = x.shape
        x = x.mean(dim=[3, 4])  # spatial GAP -> (B, T, C)
        x = x.mean(dim=1)       # temporal average -> (B, C)

        logits = self.cls(x)
        return logits


    def load_tsm_from_checkpoint(
        self,
        ckpt_path: str,
        num_classes: int,
        num_segments: int = 16,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        freeze_until: str = "layer2",
        pretrained: bool = True, 
    ):
        device = torch.device(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        return self


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
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def group_params(model: ResNetTSM, base_lr: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []

    # stem
    stem_params = list(model.base.conv1.parameters()) + list(model.base.bn1.parameters())
    stem_params = [p for p in stem_params if p.requires_grad]
    if stem_params:
        groups.append({"params": stem_params, "lr": base_lr * 0.1})

    # ResNet blocks with different LR multipliers
    for name, lr_mult in [
        ("layer1", 0.1),
        ("layer2", 0.2),
        ("layer3", 0.5),
        ("layer4", 1.0),
    ]:
        params = [p for p in getattr(model.base, name).parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * lr_mult})

    # classification head
    head_params = [p for p in model.cls.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * 1.5})

    return groups

