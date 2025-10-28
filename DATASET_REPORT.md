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

## Design Decisions

### ✅ Strengths

1. **Temporal Consistency**: Fixed 16-frame clips enable batch processing
2. **High Resolution**: 768×768 preserves surgical detail
3. **Robust Augmentation**: Improves generalization
4. **Efficient Loading**: PyAV provides fast video decoding
5. **Flexible**: Compatible with 3D CNNs, LSTMs, and Video Transformers

### ⚠️ Limitations

1. **Truncation**: Videos >16 frames lose temporal information
2. **Memory Intensive**: ~28M values per sample (16×3×768×768)
3. **Fixed Frame Rate**: No intelligent temporal sampling
4. **Padding Strategy**: Repeating last frame may introduce bias

---

## Memory & Performance

### Per Sample
- **Tensor Size**: 16 frames × 3 channels × 768 × 768 = 28,311,552 values
- **Memory (FP32)**: ~108 MB uncompressed
- **Memory (FP16)**: ~54 MB with half-precision

### Typical Batch
- **Batch Size 8**: ~864 MB (FP32) or ~432 MB (FP16)
- **Batch Size 16**: ~1.7 GB (FP32) or ~864 MB (FP16)

**Recommendation**: Use GPU with ≥8GB VRAM for batch size 8-16

---

## Use Cases

### Primary Applications
- Surgical action recognition
- Temporal modeling (3D CNNs, Video Transformers)
- Surgical workflow analysis
- Real-time procedure monitoring

### Compatible Architectures
- **3D CNNs**: C3D, I3D, R3D, SlowFast
- **Recurrent**: LSTM, GRU with CNN feature extractors
- **Transformers**: Video Vision Transformers (ViViT), TimeSformer
- **Hybrid**: SLAM-ViT (specifically designed for this dataset)

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

---

## Optimization Recommendations

### For Faster Training
1. **Reduce Resolution**: 512×512 or 384×384 (saves 2-4× memory)
2. **Fewer Frames**: 8 frames instead of 16 (saves 2× memory)
3. **Mixed Precision**: Use FP16 with automatic mixed precision (2× faster)
4. **More Workers**: Increase `num_workers` for parallel data loading

### For Better Accuracy
1. **Temporal Sampling**: Sample frames uniformly instead of truncation
2. **Multi-Scale**: Train on multiple resolutions
3. **Stronger Augmentation**: Add Mixup, CutMix, or RandAugment
4. **Ensemble**: Combine predictions from different temporal windows

---

## Dependencies

```python
torch>=1.9.0          # PyTorch framework
torchvision>=0.10.0   # Transforms and utilities
av>=8.0.3             # PyAV for video decoding
pandas>=1.3.0         # CSV handling
Pillow>=8.3.0         # Image processing
```

---

## Code Quality

- **Modularity**: ✅ Well-separated concerns
- **Documentation**: ✅ Clear docstrings
- **Error Handling**: ⚠️ Could add validation for missing files
- **Efficiency**: ✅ Leverages PyAV for fast decoding
- **Maintainability**: ✅ Easy to extend and modify

---

## Conclusion

This dataset implementation provides a **robust foundation** for surgical video action recognition. It balances:
- **Spatial detail** (768×768 resolution)
- **Temporal context** (16 frames)
- **Data augmentation** (strong regularization)
- **Computational efficiency** (fixed-size tensors)

**Status**: ✅ Production-ready for training video classification models

**Recommended Next Steps**:
1. Implement uniform temporal sampling for better coverage
2. Add validation for missing/corrupt video files
3. Create dataset statistics visualization
4. Benchmark loading speed with different configurations

---

**Generated**: October 27, 2025  
**File**: `data_preprocessing/dataset.py`  
**Lines of Code**: 179  
**Complexity**: Moderate
