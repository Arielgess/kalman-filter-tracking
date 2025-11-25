"""
PyTorch Dataset for loading trajectory data from JSON files.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class TrajectoryDataset(Dataset):
    """
    Dataset for loading trajectory data from JSON files.
    
    Each JSON file contains:
    - X: list of [x, y] positions (noisy measurements)
    - Y: list of [x, y] positions (clean, optional)
    - meta: metadata dictionary
    - dt: time step
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_duration: int,
        split: str = 'train',  # 'train', 'val', or 'test'
        delta: float = 0.04
    ):
        """
        Args:
            data_dir: Root directory containing class/subclass/trajectory_*.json
            sample_duration: Duration in time steps to extract from each trajectory
            split: 'train', 'val', or 'test' (70/15/15 split)
            delta: Time step in seconds (default 0.04)
        """
        self.data_dir = Path(data_dir)
        self.sample_duration = sample_duration
        self.split = split
        self.delta = delta
        
        # Collect all trajectory files
        self.trajectory_files = []
        self._collect_trajectories()
    
    def _collect_trajectories(self):
        """Collect all trajectory JSON files and filter by split."""
        class_dirs = sorted([
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('class_')
        ])
        
        for class_dir in class_dirs:
            subclass_dirs = sorted([
                d for d in class_dir.iterdir()
                if d.is_dir() and d.name.startswith('subclass_')
            ])
            
            for subclass_dir in subclass_dirs:
                json_files = sorted(subclass_dir.glob('trajectory_*.json'))
                
                for traj_idx, json_file in enumerate(json_files):
                    # Load to check duration
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    X = np.array(data['X'])
                    if X.shape[0] >= self.sample_duration:
                        # Split: 70% train, 15% val, 15% test
                        split_idx = traj_idx % 10
                        if self.split == 'train' and split_idx < 7:
                            self.trajectory_files.append(json_file)
                        elif self.split == 'val' and split_idx == 7:
                            self.trajectory_files.append(json_file)
                        elif self.split == 'test' and split_idx >= 8:
                            self.trajectory_files.append(json_file)
    
    def __len__(self) -> int:
        return len(self.trajectory_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sample: (T, 2) tensor - trajectory features
            dt: (T,) tensor - time intervals
            scale: (2,) tensor - normalization scales
            duration: scalar tensor - duration
        """
        json_file = self.trajectory_files[idx]
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get trajectory data
        X = np.array(data['X'])  # (T_full, 2)
        
        # Extract sample of specified duration
        X_sample = X[:self.sample_duration]  # (T, 2)
        
        # Use positions directly (2D: x, y)
        T = X_sample.shape[0]
        
        # Compute scales (local std for normalization)
        scales = np.std(X_sample, axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        
        # Create tensors
        sample = torch.FloatTensor(X_sample)  # (T, 2)
        dt = torch.FloatTensor([self.delta] * T)  # (T,)
        scale = torch.FloatTensor(scales)  # (2,)
        duration = torch.tensor(self.sample_duration, dtype=torch.long)
        
        return sample, dt, scale, duration


def prepare_mixed_dataloader(
    data_dir: str,
    sample_durations: list,
    batch_size: int,
    split: str = 'train',
    shuffle: bool = True,
    delta: float = 0.04
):
    """
    Create a mixed dataloader that combines samples of different durations.
    
    Args:
        data_dir: Root directory containing trajectory JSON files
        sample_durations: List of durations to include
        batch_size: Batch size for DataLoader
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle batches
        delta: Time step in seconds
        
    Returns:
        List of batches, where each batch is (samples, dt, scales, durations)
    """
    from torch.utils.data import DataLoader
    import random
    
    all_batches = []
    
    for dur in sample_durations:
        dataset = TrajectoryDataset(
            data_dir=data_dir,
            sample_duration=dur,
            split=split,
            delta=delta
        )
        
        if len(dataset) > 0:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
            
            for batch in dataloader:
                # batch is (samples, dt, scales, durations) from Dataset
                # Convert to format expected by Training: (data, dt, scale, dur)
                # Training expects data with T+1 timesteps (for prediction target)
                samples, dt, scales, durations = batch
                
                # Training code does: samples = data[:, :-1, :], labels = data[:, 1:, :]
                # So we need data with T+1 timesteps. Duplicate last timestep.
                batch_size_dim = samples.shape[0]
                T = samples.shape[1]
                d_model = samples.shape[2]  # 2 for (x, y)
                data = torch.zeros(batch_size_dim, T + 1, d_model)
                data[:, :-1, :] = samples
                data[:, -1, :] = samples[:, -1, :]  # Repeat last timestep
                
                # dt also needs T+1
                dt_extended = torch.zeros(batch_size_dim, T + 1)
                dt_extended[:, :-1] = dt
                dt_extended[:, -1] = dt[:, -1]
                
                all_batches.append((data, dt_extended, scales, durations.float()))
    
    if shuffle:
        random.shuffle(all_batches)
    
    return all_batches

