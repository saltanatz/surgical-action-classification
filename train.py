# train.py
import os
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ClipsDataset, get_train_transform, get_val_transform, get_num_classes
from tsm_resnet import ResNetTSM, LabelSmoothingCE, group_params as tsm_group_params, accuracy
from r2plus1d_resnet import ResNetR2Plus1D, group_params_r2plus1d
from cnnlstm_resnet import CNNLSTM, group_params_cnnlstm   # <-- new file


# ------ Reproducibility --------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------- Config --------

@dataclass
class TrainConfig:
    # paths
    root: str = "./dataset"
    train_csv: str = "./dataset/train.csv"
    val_csv: str = "./dataset/val.csv"
    test_csv: str = "./dataset/test.csv"

    # training
    num_epochs: int = 20
    batch_size: int = 8
    num_workers: int = 8

    base_lr: float = 1e-3
    weight_decay: float = 1e-4

    time_size: int = 16
    resize_shape: Tuple[int, int] = (224, 224)
    use_pretrained: bool = True
    freeze_until: str = "layer2"

    # which model to train: "tsm_resnet", "r2plus1d", "cnnlstm"
    model_name: str = "tsm_resnet"

    save_dir: str = "checkpoints"

    # sampler settings
    use_class_balanced_sampler: bool = True
    tau: float = 0.5  

    # loss
    label_smoothing: float = 0.05

    # CNN-LSTM
    cnnlstm_pool: str = "mean"        
    cnnlstm_hidden: int = 512
    cnnlstm_layers: int = 1
    cnnlstm_bidirectional: bool = False
    cnnlstm_dropout: float = 0.2


# ------- DataLoaders ---------

def build_dataloaders(cfg: TrainConfig) -> Tuple[Dict[str, DataLoader], int]:
    # datasets
    train_ds = ClipsDataset(
        data_path=cfg.root,
        csv_file=cfg.train_csv,
        transform=get_train_transform(),
        resize_shape=cfg.resize_shape,
        time_size=cfg.time_size,
    )

    val_ds = ClipsDataset(
        data_path=cfg.root,
        csv_file=cfg.val_csv,
        transform=get_val_transform(),
        resize_shape=cfg.resize_shape,
        time_size=cfg.time_size,
    )

    test_ds = ClipsDataset(
        data_path=cfg.root,
        csv_file=cfg.test_csv,
        transform=get_val_transform(),
        resize_shape=cfg.resize_shape,
        time_size=cfg.time_size,
    )

    # ---- class-balanced sampler for TRAIN ONLY ----
    if cfg.use_class_balanced_sampler:
        train_df = pd.read_csv(cfg.train_csv)

        labs = train_df["label"].astype(str)
        counts = labs.value_counts()

        tau = cfg.tau
        w_per_class = {lab: (counts[lab] ** (-tau)) for lab in counts.index}

        # normalize weights to have mean ~ 1
        m = sum(w_per_class.values()) / len(w_per_class)
        w_per_class = {k: v / m for k, v in w_per_class.items()}

        sample_weights = labs.map(lambda x: w_per_class[x]).astype(float).values

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,  
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    # val/test: no rebalancing, just regular sequential loaders
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    num_classes = get_num_classes()
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, num_classes


# -------------------------
# Evaluation helpers
# -------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total)


def format_seconds(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


# ------- Model builders ---------

def build_model_and_params(cfg: TrainConfig, num_classes: int, device: torch.device):
    """
    Returns model and param_groups for optimizer.
    """
    if cfg.model_name == "tsm_resnet":
        model = ResNetTSM(
            num_classes=num_classes,
            num_segments=cfg.time_size,
            freeze_until=cfg.freeze_until,
            pretrained=cfg.use_pretrained,
        )
        param_groups = tsm_group_params(model, cfg.base_lr)

    elif cfg.model_name == "r2plus1d":
        model = ResNetR2Plus1D(
            num_classes=num_classes,
            num_segments=cfg.time_size,
            freeze_until=cfg.freeze_until,
            pretrained=cfg.use_pretrained,
        )
        param_groups = group_params_r2plus1d(model, cfg.base_lr)

    elif cfg.model_name == "cnnlstm":
        model = CNNLSTM(
            num_classes=num_classes,
            lstm_hidden=cfg.cnnlstm_hidden,
            lstm_layers=cfg.cnnlstm_layers,
            bidirectional=cfg.cnnlstm_bidirectional,
            dropout=cfg.cnnlstm_dropout,
            pool=cfg.cnnlstm_pool,
            pretrained=cfg.use_pretrained,
            freeze_until=cfg.freeze_until,
        )
        param_groups = group_params_cnnlstm(model, cfg.base_lr)

    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    model = model.to(device)
    return model, param_groups


# ------ Training loop ---------


def train_model(
    cfg: TrainConfig,
    loaders: Dict[str, DataLoader],
    num_classes: int,
    device: torch.device,
) -> None:
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    model, param_groups = build_model_and_params(cfg, num_classes, device)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    criterion = LabelSmoothingCE(cfg.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, f"best_{cfg.model_name}.pth")

    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)
            steps += 1

        scheduler.step()

        train_loss = total_loss / max(1, steps)
        train_acc = total_acc / max(1, steps)
        val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(
            f"[{cfg.model_name}] Epoch {epoch + 1}/{cfg.num_epochs} "
            f"| loss={train_loss:.4f} acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} | time={format_seconds(epoch_time)}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_segments": cfg.time_size,
                    "resize_shape": cfg.resize_shape,
                },
                save_path,
            )
            print(
                f"Saved best {cfg.model_name} → {save_path} "
                f"(val_acc={best_val:.3f})"
            )

    # final test
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_acc = evaluate(model, test_loader, device)
    print(f"[{cfg.model_name}] Final Test Accuracy = {test_acc:.4f}")

# ----- Main -------

def main():
    set_seed(42)
    cfg = TrainConfig()   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training model: {cfg.model_name}")

    loaders, num_classes = build_dataloaders(cfg)

    model_start = time.time()
    train_model(cfg, loaders, num_classes, device)
    model_end = time.time()

    elapsed = model_end - model_start
    print(
        f"[{cfg.model_name}] Training time: {format_seconds(elapsed)} "
        f"({elapsed:.1f} seconds)"
    )

if __name__ == "__main__":
    main()
