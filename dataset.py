"""
Dataset class for loading and preprocessing laparoscopic surgical video clips.
"""

import torch
from torch.utils.data import Dataset
import av
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class ClipsDataset(Dataset):
    """
    Dataset class for laparoscopic surgical action classification.
    
    Args:
        data_path (str): Root path to the dataset directory
        csv_file (str): Path to the CSV file containing clip paths and labels
        transform (transforms.Compose, optional): Transformations to apply to frames
        resize_shape (tuple): Target size for resizing frames (width, height)
        time_size (int): Number of frames per clip (default: 16)
    """
    
    # Label mapping for surgical actions
    LABEL_DICT = {
        "UseClipper": 0,
        "HookCut": 1,
        "PanoView": 2,
        "Suction": 3,
        "AbdominalEntry": 4,
        "Needle": 5,
        "LocPanoView": 6
    }
    
    def __init__(self, data_path, csv_file, transform=None, resize_shape=(768, 768), time_size=16):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Root path to the dataset directory
            csv_file (str): Path to the CSV file containing clip paths and labels
            transform (transforms.Compose, optional): Transformations to apply to frames
            resize_shape (tuple): Target size for resizing frames (width, height)
            time_size (int): Number of frames per clip
        """
        self.data_path = data_path
        self.data_label = pd.read_csv(csv_file)
        self.resize_shape = resize_shape
        self.time_size = time_size
        self.transform = transform
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_label)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (frames, label) where frames is a tensor of shape (time_size, C, H, W)
                   and label is an integer class label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get video path and label from CSV
        relative_video_path = self.data_label.iloc[idx, 1]
        video_path = os.path.join(self.data_path, "clips", relative_video_path)
        label = self.LABEL_DICT[self.data_label.iloc[idx, 2]]
        
        # Load and preprocess video frames
        frames = self._load_video_frames(video_path)
        
        # Ensure we have exactly time_size frames
        frames = self._adjust_frame_count(frames)
        
        # Stack frames into a tensor
        frames = torch.stack(frames)
        
        return frames, label
    
    def _load_video_frames(self, video_path):
        """
        Load frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of transformed frame tensors
        """
        container = av.open(video_path)
        frames = []
        
        for frame in container.decode(video=0):
            # Convert frame to numpy array and then to PIL Image
            img = frame.to_ndarray(format='rgb24')
            img = Image.fromarray(img).resize(self.resize_shape)
            
            # Apply transformations if provided
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        container.close()
        return frames
    
    def _adjust_frame_count(self, frames):
        """
        Adjust the number of frames to match time_size.
        If fewer frames than time_size, repeat the last frame.
        If more frames than time_size, truncate to time_size.
        
        Args:
            frames (list): List of frame tensors
            
        Returns:
            list: List of frames with length equal to time_size
        """
        if len(frames) < self.time_size:
            # Pad with last frame if not enough frames
            frames += [frames[-1]] * (self.time_size - len(frames))
        elif len(frames) > self.time_size:
            # Truncate if too many frames
            frames = frames[:self.time_size]
        
        return frames


def get_train_transform():
    """
    Get the data augmentation transforms for training.
    
    Returns:
        transforms.Compose: Composition of training transforms
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])


def get_val_transform():
    """
    Get the transforms for validation/testing (no augmentation).
    
    Returns:
        transforms.Compose: Composition of validation transforms
    """
    return transforms.Compose([
        transforms.ToTensor()
    ])


def get_num_classes():
    """
    Get the number of classes in the dataset.
    
    Returns:
        int: Number of classes
    """
    return len(ClipsDataset.LABEL_DICT)
