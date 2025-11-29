"""
Fast dataloader for pre-computed tensor trajectory data.

This dataloader works with .pt files created by convert_to_tensor.py.
All trajectories are pre-trimmed to the same length, so no runtime trimming is needed.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Any


class TensorTrajectoryDataset(Dataset):
    """
    Dataset for loading pre-computed tensor trajectory data.
    
    Uses GLOBAL normalization (statistics computed across entire dataset) instead of
    per-trajectory normalization. This preserves velocity magnitude information (physics).
    
    The .pt file contains:
    - X: (N, T, 2) noisy trajectories
    - Y: (N, T, 2) clean trajectories  
    - measurement_noise_std: (N, 2) noise std per trajectory
    - seq_length: int
    - n_trajectories: int
    - dim: int
    
    Normalization modes:
    - Train mode (stats=None): Compute global mean/std from data
    - Test/Val mode (stats provided): Use provided mean/std for consistency
    """
    
    def __init__(
        self, 
        tensor_path: str, 
        normalize: bool = True,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Args:
            tensor_path: Path to the .pt file
            normalize: Whether to normalize trajectories
            stats: Optional dictionary with 'mean' and 'std' tensors for normalization.
                   If None (Train mode): compute global stats from data.
                   If provided (Test/Val mode): use provided stats.
        """
        self.tensor_path = Path(tensor_path)
        self.normalize = normalize
        
        # Load the entire dataset into memory
        print(f"Loading tensor dataset from {self.tensor_path}...")
        data = torch.load(self.tensor_path)
        
        self.X = data['X']  # (N, T, 2)
        self.Y = data['Y']  # (N, T, 2)
        self.measurement_noise_std = data['measurement_noise_std']  # (N, 2)
        self.seq_length = data['seq_length']
        self.n_trajectories = data['n_trajectories']
        self.dim = data['dim']
        
        print(f"  Loaded {self.n_trajectories} trajectories")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Dimensions: {self.dim}")
        
        # Compute or use provided normalization stats
        if self.normalize:
            if stats is not None:
                # Test/Val mode: Use provided stats
                print("  Using provided normalization stats (Test/Val mode)")
                self.global_mean = stats['mean'].view(1, 1, self.dim)  # (1, 1, 2)
                self.global_std = stats['std'].view(1, 1, self.dim)    # (1, 1, 2)
            else:
                # Train mode: Compute global stats from data
                print("  Computing global normalization stats (Train mode)")
                # Global mean/std across all trajectories and all timesteps
                # X shape: (N, T, 2) -> mean/std shape: (2,)
                self.global_mean = self.X.mean(dim=(0, 1), keepdim=True)  # (1, 1, 2)
                self.global_std = self.X.std(dim=(0, 1), keepdim=True)    # (1, 1, 2)
            
            # Avoid division by zero
            self.global_std = torch.where(
                self.global_std < 1e-8, 
                torch.ones_like(self.global_std), 
                self.global_std
            )
            
            print(f"  Global mean: {self.global_mean.squeeze().tolist()}")
            print(f"  Global std: {self.global_std.squeeze().tolist()}")
            
            # Pre-normalize X using global stats (broadcasts across N and T)
            self.X_normalized = (self.X - self.global_mean) / self.global_std
        else:
            self.X_normalized = self.X
            self.global_mean = torch.zeros(1, 1, self.dim)
            self.global_std = torch.ones(1, 1, self.dim)
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        """
        Export the normalization statistics for use with validation/test datasets.
        
        Returns:
            Dictionary with 'mean' and 'std' tensors, shape (2,) each.
        """
        return {
            'mean': self.global_mean.squeeze(),  # (2,)
            'std': self.global_std.squeeze()     # (2,)
        }
    
    def __len__(self):
        return self.n_trajectories
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with:
            - X: Normalized noisy trajectory (T, 2)
            - Y: Clean trajectory (T, 2) - not normalized for evaluation
            - X_mean: Global mean used for normalization (1, 2)
            - X_std: Global std used for normalization (1, 2)
            - measurement_noise_std: (2,)
            - dt: Time step (fixed at 0.04)
        """
        return {
            'X': self.X_normalized[idx],                    # (T, 2) normalized
            'Y': self.Y[idx],                               # (T, 2) not normalized
            'X_mean': self.global_mean.squeeze(0),          # (1, 2) - same for all items
            'X_std': self.global_std.squeeze(0),            # (1, 2) - same for all items
            'measurement_noise_std': self.measurement_noise_std[idx],  # (2,)
            'dt': 0.04
        }


def tensor_collate_fn(batch):
    """
    Simple collate function for tensor dataset.
    
    Since all sequences have the same length, we just stack them.
    No trimming needed!
    """
    # Stack all tensors
    X = torch.stack([item['X'] for item in batch])  # (B, T, 2)
    Y = torch.stack([item['Y'] for item in batch])  # (B, T, 2)
    X_mean = torch.stack([item['X_mean'] for item in batch])  # (B, 1, 2)
    X_std = torch.stack([item['X_std'] for item in batch])    # (B, 1, 2)
    measurement_noise_std = torch.stack([item['measurement_noise_std'] for item in batch])  # (B, 2)
    
    # Create dt tensor (all same value)
    batch_size = len(batch)
    seq_len = X.shape[1]
    dt = torch.full((batch_size, seq_len), batch[0]['dt'])  # (B, T)
    
    return {
        'X': X,
        'Y': Y,
        'X_mean': X_mean,
        'X_std': X_std,
        'measurement_noise_std': measurement_noise_std,
        'dt': dt,
        'lengths': torch.full((batch_size,), seq_len, dtype=torch.long)
    }


def create_tensor_dataloader(
    tensor_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
    stats: Optional[Dict[str, torch.Tensor]] = None
) -> tuple:
    """
    Create a DataLoader for tensor trajectory data.
    
    Args:
        tensor_path: Path to the .pt file
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        normalize: Whether to normalize trajectories
        stats: Optional normalization stats (for val/test sets, use train stats)
    
    Returns:
        Tuple of (DataLoader, Dataset) - dataset included to access get_stats()
    """
    dataset = TensorTrajectoryDataset(tensor_path, normalize=normalize, stats=stats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        pin_memory=torch.cuda.is_available()  # Faster GPU transfer
    )
    return loader, dataset


if __name__ == '__main__':
    # Quick test
    script_dir = Path(__file__).parent
    tensor_path = script_dir / "data" / "tensor_dataset.pt"
    
    if tensor_path.exists():
        # Test Train mode (compute stats)
        print("=" * 50)
        print("Testing TRAIN mode (compute global stats)")
        print("=" * 50)
        train_dataset = TensorTrajectoryDataset(str(tensor_path))
        print(f"\nDataset loaded: {len(train_dataset)} trajectories")
        
        # Get stats for val/test
        stats = train_dataset.get_stats()
        print(f"\nExported stats:")
        print(f"  mean: {stats['mean']}")
        print(f"  std: {stats['std']}")
        
        # Test single item
        item = train_dataset[0]
        print(f"\nSingle item:")
        for key, val in item.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {val}")
        
        # Test Val mode (use provided stats)
        print("\n" + "=" * 50)
        print("Testing VAL mode (use provided stats)")
        print("=" * 50)
        val_dataset = TensorTrajectoryDataset(str(tensor_path), stats=stats)
        
        # Test dataloader
        print("\nTesting dataloader...")
        loader, dataset = create_tensor_dataloader(str(tensor_path), batch_size=4)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {val}")
    else:
        print(f"Tensor file not found: {tensor_path}")
        print("Run convert_to_tensor.py first to create it.")

