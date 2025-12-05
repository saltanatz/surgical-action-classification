import os
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIGURATION ====================
CONFIG = {
    'base_path': Path('/home/zhadiger/Desktop/data_preprocessing/dataset/clips'),
    'num_classes': 7,
    'batch_size': 4,
    'num_epochs': 15,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'num_workers': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    'checkpoint_dir': Path('./checkpoints'),
    'checkpoint_load': Path('/home/zhadiger/Desktop/data_preprocessing/best_model_twostream.pth'),
    'use_flow': True,
    'tau_sampling': 0.5,
    'model_depth': 18,
    'target_frames': 16,
}

CONFIG['checkpoint_dir'].mkdir(parents=True, exist_ok=True)

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# ==================== Utilities ====================

def create_class_balanced_sampler(csv_path, label_col='label', tau=0.5):
    df = pd.read_csv(csv_path)
    # keep original dtype (could be int or string)
    labs = df[label_col]
    counts = labs.value_counts()

    # tempered class weights
    w_per_class = {lab: (counts[lab] ** (-tau)) for lab in counts.index}
    m = sum(w_per_class.values()) / len(w_per_class)
    w_per_class = {k: float(v / m) for k, v in w_per_class.items()}

    sample_weights = labs.map(lambda x: w_per_class[x]).astype(float).values

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Class weights (tau={tau}):")
    for lab, w in sorted(w_per_class.items(), key=lambda x: str(x[0])):
        print(f"  {lab}: {w:.4f}")

    return sampler

# ==================== TSM IMPLEMENTATION ====================

class TemporalShift(nn.Module):
    def __init__(self, num_segments: int, fold_div: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        fold = c // self.fold_div
        if fold == 0:
            return x

        # split
        x1 = x[:, :, :fold, :, :].clone()
        x2 = x[:, :, fold:2*fold, :, :].clone()
        x3 = x[:, :, 2*fold:, :, :].clone()

        # shift
        x1_shifted = torch.zeros_like(x1)
        x1_shifted[:, 1:, ...] = x1[:, :-1, ...]

        x2_shifted = torch.zeros_like(x2)
        x2_shifted[:, :-1, ...] = x2[:, 1:, ...]

        x_shifted = torch.cat([x1_shifted, x2_shifted, x3], dim=2)
        return x_shifted


class StageWrapper(nn.Module):
    def __init__(self, stage: nn.Module, num_segments: int, fold_div: int = 8):
        super().__init__()
        self.stage = stage
        self.tsm = TemporalShift(num_segments, fold_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        x = self.tsm(x)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.stage(x)
        if x.dim() == 4:
            _, c, h, w = x.shape
            x = x.view(b, t, c, h, w)
        return x


class ResNetTSM_OF(nn.Module):
    def __init__(self, num_classes: int, num_segments: int = 16,
                 depth: int = 18, pretrained: bool = True, use_flow: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.use_flow = use_flow

        if depth == 18:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Only ResNet18 supported")

        # Keep a copy of the original resnet for initializing flow
        resnet_copy = copy.deepcopy(resnet)

        # RGB stream
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = StageWrapper(resnet.layer1, num_segments)
        self.layer2 = StageWrapper(resnet.layer2, num_segments)
        self.layer3 = StageWrapper(resnet.layer3, num_segments)
        self.layer4 = StageWrapper(resnet.layer4, num_segments)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512
        hidden_dim = feat_dim // 2  # 256

        self.rgb_fc = nn.Linear(feat_dim, hidden_dim)

        # Flow stream (deep-copied weights to avoid sharing)
        if use_flow:
            # Create independent backbone by deep-copying modules
            self.flow_conv1 = copy.deepcopy(resnet_copy.conv1)
            self.flow_bn1 = copy.deepcopy(resnet_copy.bn1)
            # If input flow has 3 channels we can use conv1 directly; otherwise adapt
            # Flow-specific ResNet layers (deep copy)
            self.flow_layer1 = StageWrapper(copy.deepcopy(resnet_copy.layer1), num_segments)
            self.flow_layer2 = StageWrapper(copy.deepcopy(resnet_copy.layer2), num_segments)
            self.flow_layer3 = StageWrapper(copy.deepcopy(resnet_copy.layer3), num_segments)
            self.flow_layer4 = StageWrapper(copy.deepcopy(resnet_copy.layer4), num_segments)

            self.flow_fc = nn.Linear(feat_dim, hidden_dim)
            # Fusion maps concatenated features (rgb+flow) back to hidden_dim
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # classifier expects hidden_dim features
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_rgb(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # reshape to (B, T, C, H', W') for StageWrapper
        _, c, h2, w2 = x.shape
        x = x.view(b, t, c, h2, w2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, t, c, h3, w3 = x.shape
        x = x.view(b * t, c, h3, w3)
        x = self.avgpool(x)
        x = x.view(b, t, -1)
        x = x.mean(dim=1)  # (B, feat_dim)

        x = self.rgb_fc(x)  # (B, hidden_dim)
        return x

    def forward_flow(self, x):
        # x: (B, T, C, H, W)  -- here C=3 for (u, v, mag)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.flow_conv1(x)
        x = self.flow_bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        _, c, h2, w2 = x.shape
        x = x.view(b, t, c, h2, w2)

        x = self.flow_layer1(x)
        x = self.flow_layer2(x)
        x = self.flow_layer3(x)
        x = self.flow_layer4(x)

        b, t, c, h3, w3 = x.shape
        x = x.view(b * t, c, h3, w3)
        x = self.avgpool(x)
        x = x.view(b, t, -1)
        x = x.mean(dim=1)

        x = self.flow_fc(x)
        return x

    def forward(self, rgb, flow=None):
        rgb_feat = self.forward_rgb(rgb)

        if self.use_flow and flow is not None:
            flow_feat = self.forward_flow(flow)
            feat = torch.cat([rgb_feat, flow_feat], dim=1)
            feat = self.fusion(feat)
        else:
            feat = rgb_feat

        logits = self.classifier(feat)
        return logits

# ==================== DATASET ====================

class SurgicalVideoDataset(Dataset):
    def __init__(self, csv_path, base_path, target_frames=16):
        self.base_path = Path(base_path)
        self.target_frames = int(target_frames)
        self.data = pd.read_csv(csv_path)

        self.path_col = 'file_path' if 'file_path' in self.data.columns else ('video_path' if 'video_path' in self.data.columns else self.data.columns[0])
        self.label_col = 'label' if 'label' in self.data.columns else ('action_label' if 'action_label' in self.data.columns else self.data.columns[1])

        unique_labels = sorted(self.data[self.label_col].unique())
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        print(f"Dataset: {len(self.data)} samples, {len(self.label2idx)} classes")
        print(f"Classes: {self.idx2label}")

    def load_video(self, video_path):
        full_path = self.base_path / video_path
        if not full_path.exists():
            raise FileNotFoundError(f"Video not found: {full_path}")

        cap = cv2.VideoCapture(str(full_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read frames from {full_path}")

        # If shorter, pad by repeating last frame
        if len(frames) < self.target_frames:
            frames = frames + [frames[-1]] * (self.target_frames - len(frames))
        # If longer, sample uniformly
        if len(frames) > self.target_frames:
            indices = np.linspace(0, len(frames) - 1, self.target_frames).astype(int)
            frames = [frames[i] for i in indices]

        return frames

    def preprocess_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        frame = torch.tensor(frame).permute(2, 0, 1)  # C,H,W
        return frame

    def compute_optical_flow(self, frames):
        # frames: list of np arrays (BGR) length = target_frames
        flows = []
        for i in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flows.append(flow)
        # duplicate last flow to make len = target_frames
        if len(flows) == 0:
            flows = [np.zeros((224, 224, 2), dtype=np.float32) for _ in range(self.target_frames)]
        else:
            flows.append(flows[-1])

        flows = np.array(flows, dtype=np.float32)  # (T, H, W, 2)

        # compute magnitude
        u = flows[..., 0]
        v = flows[..., 1]
        mag = np.sqrt(u ** 2 + v ** 2 + 1e-8)

        # Normalize across sequence and spatial dims to have stable stats
        concat = np.stack([u, v, mag], axis=-1)  # (T,H,W,3)
        mean = concat.mean()
        std = concat.std() + 1e-8
        norm = (concat - mean) / std

        # convert to torch tensors (T, C, H, W)
        flow_frames = [self.preprocess_frame((norm[i] * 255.0).astype(np.float32)) for i in range(norm.shape[0])]
        flow_stack = torch.stack(flow_frames)  # (T, C, H, W)
        return flow_stack

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row[self.path_col]
        label = row[self.label_col]
        label_idx = self.label2idx[label]

        frames = self.load_video(video_path)  # list of RGB frames length = target_frames
        rgb_stack = torch.stack([self.preprocess_frame(f) for f in frames])  # (T, C, H, W)
        flow_stack = self.compute_optical_flow(frames)  # (T, C, H, W)

        return rgb_stack, flow_stack, int(label_idx)

# ==================== TRAIN / VAL / TEST ====================

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    start_time = time.time()

    pbar = tqdm(train_loader, desc='Train')
    for rgb, flow, labels in pbar:
        # rgb, flow: (B, T, C, H, W)
        rgb = rgb.to(device, dtype=torch.float)
        flow = flow.to(device, dtype=torch.float) if flow is not None else None
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(rgb, flow)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_time = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), acc, epoch_time


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    total_inference_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for rgb, flow, labels in tqdm(val_loader, desc='Val'):
            rgb = rgb.to(device, dtype=torch.float)
            flow = flow.to(device, dtype=torch.float) if flow is not None else None
            labels = labels.to(device)

            start = time.time()
            outputs = model(rgb, flow)
            inf_time = time.time() - start

            total_inference_time += inf_time
            total_samples += rgb.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_inf = total_inference_time / max(1, total_samples)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), acc, all_preds, all_labels, avg_inf

# ==================== CHECKPOINT HELPERS ====================


def adapt_checkpoint_for_flow(checkpoint_state, model_state):
    # Keep only keys that match the model_state and return filtered dict
    new_state = {}
    for k, v in checkpoint_state.items():
        if k in model_state:
            new_state[k] = v
    return new_state

# ==================== MAIN ====================

def main(csv_files):
    required_keys = {'train.csv', 'val.csv', 'test.csv'}
    if not isinstance(csv_files, dict) or not required_keys.issubset(csv_files.keys()):
        raise ValueError(f"csv_files must be a dict with keys {required_keys}")

    # sampler
    train_sampler = create_class_balanced_sampler(csv_files['train.csv'], label_col='label', tau=CONFIG['tau_sampling'])

    # datasets
    train_dataset = SurgicalVideoDataset(csv_files['train.csv'], CONFIG['base_path'], target_frames=CONFIG['target_frames'])
    val_dataset = SurgicalVideoDataset(csv_files['val.csv'], CONFIG['base_path'], target_frames=CONFIG['target_frames'])
    test_dataset = SurgicalVideoDataset(csv_files['test.csv'], CONFIG['base_path'], target_frames=CONFIG['target_frames'])

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=train_sampler,
                              shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)

    # model
    model = ResNetTSM_OF(num_classes=CONFIG['num_classes'], num_segments=CONFIG['target_frames'],
                         depth=CONFIG['model_depth'], pretrained=True, use_flow=CONFIG['use_flow'])
    model = model.to(CONFIG['device'])

    # If loading a checkpoint, attempt to load matching keys (RGB-only or two-stream)
    if CONFIG['checkpoint_load'] and CONFIG['checkpoint_load'].exists():
        try:
            ckpt = torch.load(CONFIG['checkpoint_load'], map_location=CONFIG['device'])
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
            filtered = adapt_checkpoint_for_flow(state, model.state_dict())
            model.load_state_dict(filtered, strict=False)
            print(f"Loaded checkpoint from {CONFIG['checkpoint_load']} (partial load allowed)")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")
    else:
        print("No checkpoint provided or path does not exist. Training from scratch.")

    # Optimizer / scheduler
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    metrics_history = {}

    t0 = time.time()
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loss, train_acc, train_time = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], scaler)
        val_loss, val_acc, _, _, val_inf = validate(model, val_loader, criterion, CONFIG['device'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {train_time:.2f}s")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Inf(ms): {val_inf*1000:.2f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, CONFIG['checkpoint_dir'] / 'best_model_twostream.pth')
            print("\u2713 Saved best model")

    total_time = time.time() - t0
    print(f"Total training time: {total_time/60:.2f} minutes")

    # Evaluate on test
    best_path = CONFIG['checkpoint_dir'] / 'best_model_twostream.pth'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=CONFIG['device'])['state_dict'])
    test_loss, test_acc, test_preds, test_labels, test_inf = validate(model, test_loader, criterion, CONFIG['device'])

    print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f} | Inf(ms/sample): {test_inf*1000:.2f}")

    # metrics
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
    prec_pc, rec_pc, f1_pc, support = precision_recall_fscore_support(test_labels, test_preds, average=None)

    class_names = [test_dataset.idx2label[i] for i in range(len(test_dataset.idx2label))]

    metrics = {
        'test_acc': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': {
            'class_names': class_names,
            'precision': [float(p) for p in prec_pc],
            'recall': [float(r) for r in rec_pc],
            'f1': [float(f) for f in f1_pc],
            'support': [int(s) for s in support]
        },
        'total_time_minutes': total_time/60,
        'inference_time_ms_per_sample': test_inf*1000
    }

    with open(CONFIG['checkpoint_dir'] / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=4)

    print('\nSaved metrics to metrics.json')

# ==================== Example run (uncomment to use) ====================
# csv_files = {
#     'train.csv': Path('/home/zhadiger/Desktop/data_preprocessing/dataset/train.csv'),
#     'val.csv': Path('/home/zhadiger/Desktop/data_preprocessing/dataset/val.csv'),
#     'test.csv': Path('/home/zhadiger/Desktop/data_preprocessing/test.csv'),
# }
# main(csv_files)
