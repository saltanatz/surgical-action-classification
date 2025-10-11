"""
Visualize exactly what the model receives during training.
Shows the effects of data augmentation and tensor shapes.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import ClipsDataset, get_train_transform, get_val_transform

# Reverse label dictionary for display
LABEL_NAMES = {
    0: "UseClipper",
    1: "HookCut",
    2: "PanoView",
    3: "Suction",
    4: "AbdominalEntry",
    5: "Needle",
    6: "LocPanoView"
}


def tensor_to_image(tensor):
    """
    Convert a tensor to displayable image.
    
    Args:
        tensor: Tensor of shape (C, H, W) with values in [0, 1]
        
    Returns:
        numpy array of shape (H, W, C) with values in [0, 255]
    """
    # Clone to avoid modifying original
    img = tensor.clone()
    
    # Permute from (C, H, W) to (H, W, C)
    img = img.permute(1, 2, 0)
    
    # Convert to numpy
    img = img.numpy()
    
    # Clip to valid range and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def visualize_augmentation_comparison(dataset_train, dataset_val, sample_idx=0, num_augmentations=4):
    """
    Compare the same sample with and without augmentation.
    Shows multiple augmented versions to demonstrate randomness.
    
    Args:
        dataset_train: Dataset with training augmentation
        dataset_val: Dataset without augmentation (validation)
        sample_idx: Index of sample to visualize
        num_augmentations: Number of augmented versions to show
    """
    print("=" * 80)
    print("TRAINING DATA VISUALIZATION - Augmentation Comparison")
    print("=" * 80)
    
    # Get original (no augmentation)
    frames_original, label = dataset_val[sample_idx]
    
    print(f"\nSample Information:")
    print(f"  Sample Index: {sample_idx}")
    print(f"  Label: {label} ({LABEL_NAMES[label]})")
    print(f"  Tensor Shape: {frames_original.shape}")
    print(f"  Tensor dtype: {frames_original.dtype}")
    print(f"  Value range: [{frames_original.min():.3f}, {frames_original.max():.3f}]")
    
    # Get multiple augmented versions
    augmented_samples = []
    for _ in range(num_augmentations):
        frames_aug, _ = dataset_train[sample_idx]
        augmented_samples.append(frames_aug)
    
    # Select a few frames to display (e.g., frames 0, 5, 10, 15)
    frame_indices = [0, 5, 10, 15] if frames_original.shape[0] >= 16 else [0, frames_original.shape[0]//2, frames_original.shape[0]-1]
    
    # Create figure
    rows = len(frame_indices)
    cols = num_augmentations + 1  # +1 for original
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    fig.suptitle(f'Training Data Augmentation - Sample {sample_idx} - {LABEL_NAMES[label]}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each frame
    for row_idx, frame_num in enumerate(frame_indices):
        # Original (no augmentation)
        img_original = tensor_to_image(frames_original[frame_num])
        axes[row_idx, 0].imshow(img_original)
        axes[row_idx, 0].set_title(f'Original\nFrame {frame_num+1}', fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Augmented versions
        for col_idx, frames_aug in enumerate(augmented_samples):
            img_aug = tensor_to_image(frames_aug[frame_num])
            axes[row_idx, col_idx+1].imshow(img_aug)
            axes[row_idx, col_idx+1].set_title(f'Augmented #{col_idx+1}\nFrame {frame_num+1}')
            axes[row_idx, col_idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: training_augmentation_comparison.png")
    plt.show()


def visualize_batch(dataloader, num_batches=2):
    """
    Visualize actual batches that will be fed to the model during training.
    
    Args:
        dataloader: DataLoader instance
        num_batches: Number of batches to visualize
    """
    print("\n" + "=" * 80)
    print("TRAINING DATA VISUALIZATION - Batch Structure")
    print("=" * 80)
    
    for batch_idx, (batch_frames, batch_labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\n📦 Batch {batch_idx + 1}:")
        print(f"  Batch Shape: {batch_frames.shape}")
        print(f"    - Batch Size: {batch_frames.shape[0]}")
        print(f"    - Time Steps (frames): {batch_frames.shape[1]}")
        print(f"    - Channels: {batch_frames.shape[2]}")
        print(f"    - Height: {batch_frames.shape[3]}")
        print(f"    - Width: {batch_frames.shape[4]}")
        print(f"  Labels Shape: {batch_labels.shape}")
        print(f"  Labels: {batch_labels.tolist()}")
        print(f"  Label Names: {[LABEL_NAMES[l.item()] for l in batch_labels]}")
        print(f"  Tensor dtype: {batch_frames.dtype}")
        print(f"  Value range: [{batch_frames.min():.3f}, {batch_frames.max():.3f}]")
        print(f"  Memory size: {batch_frames.element_size() * batch_frames.nelement() / 1024 / 1024:.2f} MB")
        
        # Visualize first sample from batch
        batch_size = batch_frames.shape[0]
        time_steps = batch_frames.shape[1]
        
        # Create figure showing all samples in the batch
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Batch {batch_idx + 1} - First 4 Frames from Each Sample', 
                     fontsize=14, fontweight='bold')
        
        for sample_idx in range(batch_size):
            frames = batch_frames[sample_idx]  # Shape: (T, C, H, W)
            label = batch_labels[sample_idx].item()
            
            # Show first 4 frames
            for frame_idx in range(min(4, time_steps)):
                img = tensor_to_image(frames[frame_idx])
                axes[sample_idx, frame_idx].imshow(img)
                axes[sample_idx, frame_idx].set_title(
                    f'Sample {sample_idx+1}: {LABEL_NAMES[label]}\nFrame {frame_idx+1}'
                )
                axes[sample_idx, frame_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'training_batch_{batch_idx+1}.png', dpi=150, bbox_inches='tight')
        print(f" Saved visualization to: training_batch_{batch_idx+1}.png")
        plt.show()


def visualize_model_input_shape(dataloader):
    """
    Show exactly what tensor shape and format the model receives.
    
    Args:
        dataloader: DataLoader instance
    """
    print("\n" + "=" * 80)
    print("MODEL INPUT SPECIFICATION")
    print("=" * 80)
    
    # Get one batch
    batch_frames, batch_labels = next(iter(dataloader))
    
    print(f"\n🎯 Exact Model Input:")
    print(f"  Input Tensor Shape: {tuple(batch_frames.shape)}")
    print(f"    Format: (Batch, Time, Channels, Height, Width)")
    print(f"    Interpretation:")
    print(f"      - {batch_frames.shape[0]} samples per batch")
    print(f"      - {batch_frames.shape[1]} frames per video clip")
    print(f"      - {batch_frames.shape[2]} color channels (RGB)")
    print(f"      - {batch_frames.shape[3]}x{batch_frames.shape[4]} pixels per frame")
    print(f"\n  Label Tensor Shape: {tuple(batch_labels.shape)}")
    print(f"  Data Type: {batch_frames.dtype}")
    print(f"  Device: {batch_frames.device}")
    print(f"  Value Range: [{batch_frames.min().item():.4f}, {batch_frames.max().item():.4f}]")
    print(f"  Mean: {batch_frames.mean().item():.4f}")
    print(f"  Std: {batch_frames.std().item():.4f}")
    
    # Show per-channel statistics
    print(f"\nPer-Channel Statistics (RGB):")
    for c in range(batch_frames.shape[2]):
        channel_data = batch_frames[:, :, c, :, :]
        print(f"  Channel {c} ({"RGB"[c]}):")
        print(f"    Mean: {channel_data.mean().item():.4f}")
        print(f"    Std:  {channel_data.std().item():.4f}")
        print(f"    Min:  {channel_data.min().item():.4f}")
        print(f"    Max:  {channel_data.max().item():.4f}")
    
    # Show what ViViT model expects
    print(f"\n🔧 ViViT Model Expectations:")
    print(f"  Expected Input: Tensor of shape (B, T, C, H, W)")
    print(f"  Your Data:      Tensor of shape {tuple(batch_frames.shape)} ")
    print(f"  Value Range:    [0, 1] (normalized) ")
    print(f"  Data Type:      torch.float32 ")
    
    return batch_frames, batch_labels


def show_augmentation_effects(dataset_train, sample_idx=0, num_versions=6):
    """
    Show the effects of individual augmentation techniques.
    
    Args:
        dataset_train: Training dataset with augmentation
        sample_idx: Sample index
        num_versions: Number of augmented versions to generate
    """
    print("\n" + "=" * 80)
    print("AUGMENTATION EFFECTS - Random Variations")
    print("=" * 80)
    print("\nApplied Augmentations:")
    print("  1. Random Horizontal Flip (p=0.5)")
    print("  2. Random Rotation (±20 degrees)")
    print("  3. Color Jitter (brightness, contrast, saturation ±20%)")
    print("  4. Random Affine (translate ±10%, scale 90-110%)")
    print("  5. ToTensor (normalize to [0, 1])")
    
    # Generate multiple augmented versions
    augmented_frames = []
    for i in range(num_versions):
        frames, label = dataset_train[sample_idx]
        augmented_frames.append(frames)
        print(f"\n  Version {i+1} generated - Shape: {frames.shape}")
    
    label_name = LABEL_NAMES[dataset_train[sample_idx][1]]
    
    # Show first frame from each version
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle(f'Data Augmentation Variations - Sample {sample_idx} ({label_name})\nFirst Frame Only', 
                 fontsize=14, fontweight='bold')
    
    for idx, frames in enumerate(augmented_frames):
        img = tensor_to_image(frames[0])  # First frame
        axes[idx].imshow(img)
        axes[idx].set_title(f'Augmented Version {idx+1}', fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_augmentation_effects.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: training_augmentation_effects.png")
    plt.show()


def main():
    """
    Main function to visualize training data.
    """
    print("=" * 80)
    print("TRAINING DATA VISUALIZATION")
    print("Visualizing exactly what the model receives during training")
    print("=" * 80)
    
    # Configuration (update these paths)
    data_path = "/Users/itsu/Desktop/deep_learning/dataset"
    train_csv = f"{data_path}/train.csv"
    
    batch_size = 2
    time_size = 16
    height = width = 768
    
    print(f"\n Configuration:")
    print(f"  Data Path: {data_path}")
    print(f"  CSV File: {train_csv}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Time Size (frames): {time_size}")
    print(f"  Image Size: {height}x{width}")
    
    # Create training dataset with augmentation
    print(f"\nLoading training dataset (WITH augmentation)...")
    train_dataset = ClipsDataset(
        data_path=data_path,
        csv_file=train_csv,
        transform=get_train_transform(),
        resize_shape=(width, height),
        time_size=time_size
    )
    
    # Create validation dataset without augmentation for comparison
    print(f"Loading validation dataset (NO augmentation)...")
    val_dataset = ClipsDataset(
        data_path=data_path,
        csv_file=train_csv,  # Using same data for comparison
        transform=get_val_transform(),
        resize_shape=(width, height),
        time_size=time_size
    )
    
    print(f"Datasets loaded: {len(train_dataset)} samples")
    
    # Create DataLoader (as used in training)
    print(f"\nCreating DataLoader (as used in actual training)...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for visualization to avoid issues
        pin_memory=False
    )
    
    # 1. Show augmentation comparison
    print("\n" + "=" * 80)
    print("STEP 1: Augmentation Comparison")
    print("=" * 80)
    visualize_augmentation_comparison(train_dataset, val_dataset, sample_idx=0, num_augmentations=4)
    
    # 2. Show augmentation effects
    print("\n" + "=" * 80)
    print("STEP 2: Augmentation Effects")
    print("=" * 80)
    show_augmentation_effects(train_dataset, sample_idx=0, num_versions=6)
    
    # 3. Show model input specification
    print("\n" + "=" * 80)
    print("STEP 3: Model Input Specification")
    print("=" * 80)
    batch_frames, batch_labels = visualize_model_input_shape(train_loader)
    
    # 4. Visualize actual batches
    print("\n" + "=" * 80)
    print("STEP 4: Actual Training Batches")
    print("=" * 80)
    visualize_batch(train_loader, num_batches=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - What the Model Receives During Training")
    print("=" * 80)
    print(f"\n✨ Input Format:")
    print(f"  Shape: (Batch={batch_size}, Time={time_size}, Channels=3, Height={height}, Width={width})")
    print(f"  Type: {batch_frames.dtype}")
    print(f"  Range: [0.0, 1.0] (normalized)")
    print(f"  Device: CPU (moved to CUDA during training)")
    
    print(f"\n✨ Data Augmentation (Training Only):")
    print(f"  ✓ Random horizontal flip")
    print(f"  ✓ Random rotation (±20°)")
    print(f"  ✓ Color jittering")
    print(f"  ✓ Random affine transformations")
    
    print(f"\n✨ Output Labels:")
    print(f"  Shape: ({batch_size},)")
    print(f"  Type: torch.int64")
    print(f"  Classes: {len(LABEL_NAMES)} surgical actions")
    
    print(f"\n📁 Generated Files:")
    print(f"  - training_augmentation_comparison.png")
    print(f"  - training_augmentation_effects.png")
    print(f"  - training_batch_1.png")
    print(f"  - training_batch_2.png")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
