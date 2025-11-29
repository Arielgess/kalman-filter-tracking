"""
Script to convert JSON trajectory data to a compact tensor format.

This script:
1. Finds the shortest trajectory in the dataset
2. Trims all trajectories to that length
3. Saves as a single .pt file (much smaller than JSON)

Usage:
    python convert_to_tensor.py
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def find_shortest_trajectory_length(data_dir: Path) -> int:
    """Find the length of the shortest trajectory in the dataset."""
    min_length = float('inf')
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        trajectories_file = class_dir / "trajectories.json"
        if not trajectories_file.exists():
            continue
            
        print(f"Scanning {class_dir.name}...")
        with open(trajectories_file, 'r') as f:
            trajectories = json.load(f)
        
        for traj in tqdm(trajectories, desc="Checking lengths"):
            X = traj.get('X', [])
            length = len(X)
            if length < min_length:
                min_length = length
    
    return int(min_length)


def convert_to_tensor(data_dir: Path, output_path: Path, trim_length: int = None):
    """
    Convert JSON trajectory data to tensor format.
    
    Args:
        data_dir: Path to the data directory containing class folders
        output_path: Path to save the .pt file
        trim_length: Length to trim trajectories to (if None, uses shortest)
    """
    all_X = []  # Noisy trajectories
    all_Y = []  # Clean trajectories
    all_measurement_noise = []  # Measurement noise std per trajectory
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        trajectories_file = class_dir / "trajectories.json"
        if not trajectories_file.exists():
            continue
        
        print(f"Loading {class_dir.name}...")
        with open(trajectories_file, 'r') as f:
            trajectories = json.load(f)
        
        for traj in tqdm(trajectories, desc="Processing trajectories"):
            X = np.array(traj.get('X', []))
            Y = np.array(traj.get('Y', [])) if 'Y' in traj else X.copy()
            
            # Trim to specified length
            X = X[:trim_length]
            Y = Y[:trim_length]
            
            # Get measurement noise std from metadata
            meta = traj.get('meta', {})
            meas_noise = meta['measurement_noise_std']
            all_X.append(X)
            all_Y.append(Y)
            all_measurement_noise.append(meas_noise)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(np.stack(all_X))  # (N, T, 2)
    Y_tensor = torch.FloatTensor(np.stack(all_Y))  # (N, T, 2)
    meas_noise_tensor = torch.FloatTensor(np.array(all_measurement_noise))  # (N, 2)
    
    # Create dataset dictionary
    dataset = {
        'X': X_tensor,
        'Y': Y_tensor,
        'measurement_noise_std': meas_noise_tensor,
        'seq_length': trim_length,
        'n_trajectories': len(all_X),
        'dim': X_tensor.shape[-1]
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    
    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved tensor dataset to {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Trajectories: {dataset['n_trajectories']}")
    print(f"  Sequence length: {dataset['seq_length']}")
    print(f"  X shape: {X_tensor.shape}")
    print(f"  Y shape: {Y_tensor.shape}")
    
    return dataset


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "data"
    output_path = script_dir / "data" / "tensor_dataset.pt"
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Step 1: Find shortest trajectory
    print("Step 1: Finding shortest trajectory length...")
    min_length = find_shortest_trajectory_length(data_dir)
    print(f"\nShortest trajectory length: {min_length}")
    
    # Step 2: Convert to tensor with trimming
    print(f"\nStep 2: Converting to tensor (trimming to {min_length} timesteps)...")
    dataset = convert_to_tensor(data_dir, output_path, trim_length=min_length)
    
    print("\n✓ Conversion complete!")

