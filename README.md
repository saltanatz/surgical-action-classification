# Spatio-Temporal Action Classification in Laparoscopic Surgery Videos  
_A Deep Learning Project on Surgical Tool & Action Recognition_

This repository contains the implementation of a deep learning pipeline for **classifying surgical actions from laparoscopic videos**.  
We develop three spatio-temporal deep learning architectures:

- **TSM-ResNet** (Temporal Shift Module on ResNet-18)
- **R(2+1)D** (3D CNN with factorized 2D + 1D convolutions)
- **CNN–LSTM** (ResNet-18 frame encoder + LSTM temporal model)

The goal is to achieve a strong **accuracy–speed trade-off** suitable for near real-time use in the operating room.

> Project for the **Deep Learning** course at Nazarbayev University.  
> Authors: **Saltanat Zarkhinova, Amira Toksanbay, Yerulan Kongrat, Ayaulym Raikhankyzy, Aizhar Kudenova**.

---

## 1. Problem Description

Laparoscopic surgeries contain a variety of tool interactions and surgical actions. Automatically identifying actions from video helps support:

- Surgical workflow understanding  
- Skill assessment  
- Automated surgical reporting  
- Real-time AI assistance  

We formulate the task as **multiclass video clip classification**:

- **Input:** 16-frame video clip  
- **Output:** one of **7 surgical actions**

---

## 2. Dataset (Not Included)

We train and evaluate models on a laparoscopic surgery video dataset consisting of:

- Short video clips extracted from full surgical procedures  
- 7 action classes:
  - AbdominalEntry  
  - UseClipper  
  - HookCut  
  - Suturing  
  - PanoView  
  - LocPanoView  
  - Suction  

Datasets are **NOT included** in this repository due to licensing restrictions.

CSV files (`train.csv`, `val.csv`, `test.csv`) map clip file paths to labels.

---

## 3. Data Pipeline (`dataset.py`)

### `ClipsDataset`
Loads clips using **PyAV**, applies preprocessing, and ensures uniform temporal length (16 frames):

- Resize frames  
- Data augmentation for training  
- Repeat/truncate frames to length `T = 16`  
- Output shape: **(T, C, H, W)**

### Transforms
- **Training:** flip, rotation, color jitter, affine, `ToTensor`
- **Validation/Test:** `ToTensor`

### Classes
`get_num_classes()` returns **7**.

---

## 4. Implemented Models

### 4.1. TSM-ResNet (`tsm_resnet.py`)
Adds temporal modeling by shifting feature channels between frames:

- Base: ResNet-18 (ImageNet-pretrained)
- TSM blocks inserted before each ResNet stage  
- Spatial + temporal pooling  
- Linear classifier  
- Layer-wise LR groups available  
- Label smoothing supported  

---

### 4.2. R(2+1)D (`r2plus1d_resnet.py`)
A 3D CNN with factorized convolutions:

- Base: `torchvision.models.video.r2plus1d_18`
- Optional Kinetics-400 weights  
- Custom classification head  
- Temporal dim arranged as `(B, C, T, H, W)`
- Layer freezing and LR grouping supported  

---

### 4.3. CNN–LSTM (`cnn_lstm.py`)
Two-stage architecture:

1. Frame-level CNN encoder (ResNet-18, fc removed)  
2. LSTM for temporal dynamics  
3. Temporal pooling + classifier  

Advantages:
- Light and fast  
- Strong performance on surgical video clips  

---

## 5. Training Pipeline (`train.py`)

### 5.1. `TrainConfig`
Configurable parameters include:

- Data paths  
- Model choice: `tsm_resnet`, `r2plus1d`, `cnnlstm`  
- LR, batch size, epochs  
- Label smoothing  
- Class-balanced sampling (via WeightedRandomSampler)  
- CNN–LSTM hyperparameters  

### 5.2. DataLoaders
`build_dataloaders`:

- Creates datasets and loaders  
- Applies class balancing via:
  \[
  w_i = (\text{count}_i)^{-\tau}
  \]
- Returns all loaders + class count  

### 5.3. Training Loop
Uses:

- **AdamW** optimizer  
- **CosineAnnealingLR** scheduler  
- Mixed precision (`autocast`, `GradScaler`)  
- Gradient clipping  
- Best checkpoint saving  
- Final test evaluation  

---

## 6. Running the Code

### 6.1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6.2. Training a Model

Select a model in `TrainConfig`:

```python
model_name = "tsm_resnet"   # or "r2plus1d" or "cnnlstm"
```

---

## 7. Results

Below is the comparison of all video models trained on the SLAM dataset.  
Values are extracted from our final evaluation table presented in the report.

### **Test-set Performance & Inference Latency**

| Model       | Params (M) | Accuracy | Macro F1 | Mean Batch Time (s)   | Latency (ms/clip) |
|-------------|------------|----------|----------|------------------------|--------------------|
| **R(2+1)D** | 31.00      | 0.6244   | 0.5450   | 0.3654 ± 0.0149        | 45.67              |
| **TSM**     | 11.00      | 0.6054   | 0.5567   | 0.0478 ± 0.0037        | 5.97               |
| **ViViT**   | 3.70       | 0.3946   | 0.3146   | 0.0248 ± 0.0013        | 3.10               |
| **CNN–LSTM**| 13.28      | **0.7300** | **0.6542** | 0.0494 ± 0.0055     | 6.18               |

### **Key Insights**
- **CNN–LSTM** achieves the **highest accuracy** and **highest Macro F1**, confirming strong generalization across all classes.  
- **TSM** is the **fastest practical model** with **~6 ms latency** while maintaining competitive accuracy.  
- **R(2+1)D** provides strong spatio-temporal modeling but is computationally heavier.  
- **ViViT**, despite being light and fast, yields the **lowest accuracy** on this dataset.


---

## 8. Limitations & Future Work

### **Current Limitations**
- Imbalanced dataset  
- Short temporal window (16 frames)  
- Limited interpretability  
- No explicit motion modeling beyond channel shifting (TSM)

### **Future Work**
- Train on longer clips (32–64 frames)  
- Explore transformer-based video models:  
  - **ViViT**, **MViT**, **TimeSformer**, **Video Swin**  
- Self-supervised pretraining on unlabeled surgical videos  
- Improved imbalance handling:  
  - **Focal Loss**, **LDAM**, **Class reweighting**  
- Incorporate motion cues:  
  - **Optical flow**, **temporal gradients**, **motion vectors**  
- Video interpretability methods:  
  - **Grad-CAM++**  
  - **Attention rollout**  
  - **Temporal relevance maps**

---
