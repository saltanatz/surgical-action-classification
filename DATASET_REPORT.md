# Surgical Video Dataset - Technical Report

**Component:** `data_preprocessing/dataset.py`  
PyTorch dataset for laparoscopic surgical action recognition  
---

## Overview

A custom PyTorch Dataset class that loads surgical video clips, extracts frames, and prepares them for action classification using deep learning models.

---

## Key Specifications

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Number of Classes** | 7 | Surgical action categories |
| **Frames per Clip** | 16 | Temporal window size |
| **Frame Resolution** | 768×768 | Spatial dimensions |
| **Color Channels** | 3 (RGB) | Color information |
| **Output Tensor Shape** | `(16, 3, 768, 768)` | Time × Channels × Height × Width |

---

## Surgical Action Classes

The dataset recognizes 7 laparoscopic surgical actions:

1. **UseClipper** (0) - Using surgical clippers
2. **HookCut** (1) - Cutting with hook tool
3. **PanoView** (2) - Panoramic viewing
4. **Suction** (3) - Suction operation
5. **AbdominalEntry** (4) - Entry into abdomen
6. **Needle** (5) - Needle handling
7. **LocPanoView** (6) - Localized panoramic view

---

## Data Pipeline

```
Video File → Frame Extraction → Resize → Transform → Normalize → Tensor
   (.mp4)       (PyAV decode)    (768×768)  (augment)  (16 frames)  (4D)
```

### Processing Steps

1. **Load Video**: Opens video file using PyAV (FFmpeg wrapper)
2. **Extract Frames**: Decodes all frames to RGB24 format
3. **Resize**: Standardizes spatial dimensions to 768×768
4. **Transform**: Applies data augmentation (training) or basic conversion (validation)
5. **Temporal Adjustment**: 
   - **If < 16 frames**: Pad by repeating last frame
   - **If > 16 frames**: Truncate to first 16 frames
6. **Stack**: Combine into single 4D tensor

---

## Data Augmentation

### Training Augmentation
- **RandomHorizontalFlip** (50%) - Mirrors image
- **RandomRotation** (±20°) - Rotates frame
- **ColorJitter** (±20%) - Varies brightness/contrast/saturation
- **RandomAffine** (±10% shift, 0.9-1.1× scale) - Translates/scales image
- **ToTensor** - Converts to normalized tensor [0, 1]

### Validation (No Augmentation)
- **ToTensor** only - Consistent evaluation

---

## Technical Details

### Input Format
- **CSV Structure**: `[index, video_path, action_label]`
- **Video Formats**: MP4, AVI, or any FFmpeg-supported format
- **Directory Structure**:
  ```
  data_path/
  └── clips/
      ├── video_001.mp4
      ├── video_002.mp4
      └── ...
  ```

### Output Format
```python
# Single sample
frames: torch.FloatTensor  # Shape: (16, 3, 768, 768)
                           # Range: [0.0, 1.0]
label: int                 # Range: [0-6]
```

---

## Integration Example

```python
from dataset import ClipsDataset, get_train_transform, get_val_transform

# Training dataset
train_dataset = ClipsDataset(
    data_path='/path/to/data',
    csv_file='train.csv',
    transform=get_train_transform(),
    resize_shape=(768, 768),
    time_size=16
)

# Validation dataset
val_dataset = ClipsDataset(
    data_path='/path/to/data',
    csv_file='val.csv',
    transform=get_val_transform(),
    resize_shape=(768, 768),
    time_size=16
)

# DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for frames, labels in train_loader:
    # frames: (batch_size, 16, 3, 768, 768)
    # labels: (batch_size,)
    outputs = model(frames)
    loss = criterion(outputs, labels)
    # ... training code
```
