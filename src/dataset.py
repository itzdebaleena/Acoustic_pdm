"""
Acoustic PDM — PyTorch Dataset & DataLoader Module
==================================================
Dataset wrappers and DataLoader factory for batched loading of normalized
acoustic context tensor blocks of shape (Batch, 1, 128, 5).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AcousticDataset(Dataset):
    """
    PyTorch Dataset wrapper for preprocessed acoustic blocks.
    
    Each item is a normalized 4D block: (1, 128, 5) corresponding to
    1 channel, 128 Mel frequency bins, and 5 consecutive time frames.
    """
    
    def __init__(
        self,
        data: Union[str, Path, np.ndarray],
        labels: Optional[Union[np.ndarray, list]] = None,
        transform=None
    ):
        """
        Args:
            data: NumPy array of shape (N, 1, 128, 5) or path to .npy file.
            labels: Optional labels (0 for normal, 1 for anomaly).
            transform: Optional PyTorch transforms to apply.
        """
        if isinstance(data, (str, Path)):
            self.data = np.load(str(data)).astype(np.float32)
        else:
            self.data = data.astype(np.float32)
            
        # Ensure shape is (N, 1, 128, 5)
        if self.data.ndim == 3:
            self.data = np.expand_dims(self.data, axis=1)
            
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        x = self.data[idx]
        tensor = torch.from_numpy(x)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        if self.labels is not None:
            return tensor, self.labels[idx]
            
        return tensor


def get_dataloaders(
    train_data: Union[str, Path, np.ndarray],
    val_data: Optional[Union[str, Path, np.ndarray]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Factory to construct PyTorch DataLoaders for training and validation.
    
    Args:
        train_data: Array or file path for training normal blocks.
        val_data: Array or file path for validation normal blocks (optional).
        batch_size: Mini-batch size (default 32).
        num_workers: Number of DataLoader worker processes (0 on Windows if multiprocessing issue).
        pin_memory: Pin memory for faster GPU tensor transfer.
        
    Returns:
        (train_loader, val_loader) tuple.
    """
    # Safe worker count for Windows
    if os.name == "nt" and num_workers > 0:
        num_workers = 0
        
    train_dataset = AcousticDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = None
    if val_data is not None:
        val_dataset = AcousticDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
    return train_loader, val_loader
