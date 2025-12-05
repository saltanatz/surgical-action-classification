import torch
import torch.nn as nn
from torchvision import models


class FrameEncoder(nn.Module):
    """
    Args:
        pretrained (bool): if true, load ImageNet-pretrained weights.
        freeze_until (str): part of the backbone to freeze.
    """

    def __init__(self, pretrained: bool = True, freeze_until: str = "layer2"):
        super().__init__()

        if pretrained:
            try:
                base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                base = models.resnet18(pretrained=True)
        else:
            base = models.resnet18(weights=None)

        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.base = base
        self.out_dim = feat_dim
 
        self._freeze_backbone(freeze_until)

    def _freeze_backbone(self, freeze_until: str) -> None:
        valid = {"none": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
        if freeze_until not in valid:
            raise ValueError(f"freeze_until must be one of {list(valid.keys())}, got {freeze_until}")
        k = valid[freeze_until]
 
        if k >= 1:
            modules = [
                self.base.conv1,
                self.base.bn1,
                self.base.relu,
                self.base.maxpool,
                self.base.layer1,
            ]
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = False

        if k >= 2:
            for p in self.base.layer2.parameters():
                p.requires_grad = False

        if k >= 3:
            for p in self.base.layer3.parameters():
                p.requires_grad = False

        if k >= 4:
            for p in self.base.layer4.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, 3, H, W)

        Returns:
            Tensor of shape (N, out_dim)
        """
        return self.base(x)


class CNNLSTM(nn.Module):
    """

    Input shape: (B, T, C, H, W)

    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        pool: str = "mean",
        pretrained_encoder: bool = True,
        freeze_until: str = "layer2",
    ):
        super().__init__()
 
        self.encoder = FrameEncoder(pretrained=pretrained_encoder, freeze_until=freeze_until)
 
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,      
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.pool = pool
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
 
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: Tensor of shape (B, T, C, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, T, C, H, W = clips.shape
 
        x = clips.view(B * T, C, H, W)              # (B*T, C, H, W)
        feat = self.encoder(x)                      
        feat = feat.view(B, T, -1)                  # (B, T, D)
 
        out, _ = self.lstm(feat)                    # (B, T, H_lstm)
 
        if self.pool == "last":
            pooled = out[:, -1, :]                  
        elif self.pool == "max":
            pooled, _ = out.max(dim=1)               
        else:   
            pooled = out.mean(dim=1)                # (B, H_lstm)

        logits = self.cls(pooled)                   # (B, num_classes)
        return logits


def build_cnn_lstm(
    num_classes: int,
    lstm_hidden: int = 512,
    lstm_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.2,
    pool: str = "mean",
    pretrained_encoder: bool = True,
    freeze_until: str = "layer2",
) -> CNNLSTM:
    return CNNLSTM(
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pool=pool,
        pretrained_encoder=pretrained_encoder,
        freeze_until=freeze_until,
    )


__all__ = ["FrameEncoder", "CNNLSTM", "build_cnn_lstm"]
