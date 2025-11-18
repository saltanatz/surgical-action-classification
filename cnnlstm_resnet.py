import os, time, random, json, csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from pathlib import Path
from dataset import ClipsDataset, get_train_transform, get_val_transform

# paths
ROOT      = "/path/to/dataset_root"
TRAIN_CSV = "/path/to/train.csv"
VAL_CSV   = "/path/to/val.csv"
TEST_CSV  = "/path/to/test.csv"
OUT_DIR   = "runs_cnnlstm"    

# configs
TIME_SIZE    = 16
RESIZE_SHAPE = (224, 224)
BATCH_SIZE   = 8
NUM_WORKERS  = 8
EPOCHS       = 30
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
POOL_MODE    = "mean"
 
PRETRAINED    = True
FREEZE_UNTIL  = "layer2"    
ENC_LR        = 1e-4
HEAD_LR       = 3e-4
UNFREEZE_E3   = True
UNFREEZE_E6   = True
ENC_LR_E3     = 7e-5
ENC_LR_E6     = 5e-5
 

SAMPLING_TAU  = 0.5  # sampling strength: 0=no rebalance, 1=strong 

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.size(0)
    return correct / max(1,total)

class FrameEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_until="layer2"):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(pretrained=pretrained)
        feat = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.out_dim = feat
        k = {"none":0,"layer1":1,"layer2":2,"layer3":3,"layer4":4}[freeze_until]
        if k >= 1:
            for m in [self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool, self.base.layer1]:
                for p in m.parameters(): p.requires_grad = False
        if k >= 2:
            for p in self.base.layer2.parameters(): p.requires_grad = False
        if k >= 3:
            for p in self.base.layer3.parameters(): p.requires_grad = False
        if k >= 4:
            for p in self.base.layer4.parameters(): p.requires_grad = False
    def forward(self,x): return self.base(x)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=1, bidirectional=False, dropout=0.2, pool="mean",
                 pretrained=True, freeze_until="layer2"):
        super().__init__()
        self.encoder = FrameEncoder(pretrained=pretrained, freeze_until=freeze_until)
        self.pool = pool
        self.lstm = nn.LSTM(self.encoder.out_dim, lstm_hidden, lstm_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if lstm_layers>1 else 0.0)
        feat = lstm_hidden * (2 if bidirectional else 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))
    def forward(self, clips):
        B,T,C,H,W = clips.shape
        f = self.encoder(clips.view(B*T,C,H,W)).view(B,T,-1)
        out,_ = self.lstm(f)
        if self.pool == "last": pooled = out[:,-1,:]
        elif self.pool == "max": pooled,_ = torch.max(out,1)
        else: pooled = out.mean(1)
        return self.cls(pooled)

def build_optimizer(model, enc_lr, head_lr):
    enc_params  = [p for p in model.encoder.base.parameters() if p.requires_grad]
    head_params = list(model.lstm.parameters()) + list(model.cls.parameters())
    return torch.optim.AdamW(
        [{"params": enc_params,  "lr": enc_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=WEIGHT_DECAY
    )

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pth"
    last_ckpt = out_dir / "last.pth"
    log_txt   = out_dir / "train.log"
    log_csv   = out_dir / "metrics.csv"
    cfg_json  = out_dir / "config.json"

    def log(msg):
        print(msg, flush=True)
        with open(log_txt, "a") as f: f.write(msg + "\n")
 
    cfg = {
        "root": ROOT, "train_csv": TRAIN_CSV, "val_csv": VAL_CSV, "test_csv": TEST_CSV,
        "time_size": TIME_SIZE, "resize_shape": RESIZE_SHAPE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY, "label_smooth": LABEL_SMOOTH,
        "pool": POOL_MODE, "pretrained": PRETRAINED, "freeze_until": FREEZE_UNTIL,
        "enc_lr": ENC_LR, "head_lr": HEAD_LR, "unfreeze_e3": UNFREEZE_E3, "unfreeze_e6": UNFREEZE_E6,
        "enc_lr_e3": ENC_LR_E3, "enc_lr_e6": ENC_LR_E6, "sampling_tau": SAMPLING_TAU
    }
    with open(cfg_json, "w") as f: json.dump(cfg, f, indent=2)
 
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","val_acc","enc_lr","head_lr","secs"])

    log(f"Device: {device}")
 
    train_ds = ClipsDataset(ROOT, TRAIN_CSV, get_train_transform(), RESIZE_SHAPE, TIME_SIZE)
    val_ds   = ClipsDataset(ROOT, VAL_CSV,   get_val_transform(),   RESIZE_SHAPE, TIME_SIZE)
    test_ds  = ClipsDataset(ROOT, TEST_CSV,  get_val_transform(),   RESIZE_SHAPE, TIME_SIZE)
 
    tr_df = pd.read_csv(TRAIN_CSV)
    labs = tr_df["label"].astype(str)
    counts = labs.value_counts()
    w_per_class = {lab: (counts[lab] ** (-SAMPLING_TAU)) for lab in counts.index}
    m = sum(w_per_class.values()) / len(w_per_class)
    w_per_class = {k: v/m for k,v in w_per_class.items()}
    sample_weights = labs.map(lambda x: w_per_class[x]).astype(float).values
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0)
    )

    num_classes = len(sorted(pd.read_csv(TRAIN_CSV)["label"].astype(str).unique()))
    model = CNNLSTM(num_classes=num_classes, pool=POOL_MODE, pretrained=PRETRAINED, freeze_until=FREEZE_UNTIL).to(device)

    optimizer = build_optimizer(model, ENC_LR, HEAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = 0.0
    for ep in range(EPOCHS):
        if UNFREEZE_E3 and ep == 3:
            for p in model.encoder.base.layer4.parameters(): p.requires_grad = True
            optimizer = build_optimizer(model, ENC_LR_E3, HEAD_LR)
            log("[unfreeze] layer4")
        if UNFREEZE_E6 and ep == 6:
            for p in model.encoder.base.layer3.parameters(): p.requires_grad = True
            optimizer = build_optimizer(model, ENC_LR_E6, HEAD_LR)
            log("[unfreeze] layer3")

        model.train(); tot_loss=tot_acc=steps=0; t0=time.time()
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(clips); loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(clips); loss = criterion(logits, labels)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            with torch.no_grad():
                tot_acc += (logits.argmax(1)==labels).float().mean().item()
                tot_loss += loss.item(); steps += 1

        scheduler.step()
        train_loss = tot_loss / max(1,steps)
        train_acc  = tot_acc  / max(1,steps)
        val_acc    = evaluate(model, val_loader, device)
        secs = time.time() - t0
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        enc_lr, head_lr = (lrs+[None,None])[:2]

        log(f"Epoch {ep+1:03d}/{EPOCHS} | loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  lr(enc)={enc_lr:.2e} lr(head)={head_lr:.2e}  ({secs:.1f}s)")
        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f); w.writerow([ep+1, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}", enc_lr, head_lr, f"{secs:.2f}"])

        torch.save({
            "epoch": ep+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "cfg": cfg
        }, last_ckpt)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": ep+1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "cfg": cfg
            }, best_ckpt)
            log(f"[ckpt] saved best → {best_ckpt} (val_acc={best_val:.3f})")

    state = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(state["model"]); model.to(device)
    test_acc = evaluate(model, test_loader, device)
    log(f"\nFinal Test Accuracy (best ckpt) = {test_acc:.4f}")

if __name__ == "__main__":
    main()
