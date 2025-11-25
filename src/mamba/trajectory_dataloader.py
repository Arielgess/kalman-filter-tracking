import json
import re
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """
    Simple Dataset for loading trajectory JSON files.
    All data is loaded once during initialization for better performance.
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the data/ directory containing class folders
        """
        self.data_dir = Path(data_dir)
        self.trajectory_data = []  # List of pre-loaded trajectory data dictionaries

        # Load all trajectories from trajectories.json files
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            # Extract actual class name (remove class_X_ prefix)
            if class_name.startswith("class_"):
                parts = class_name.split("_", 2)
                if len(parts) >= 3:
                    actual_class_name = parts[2]
                else:
                    actual_class_name = class_name
            else:
                actual_class_name = class_name

            # Check for trajectories.json file
            trajectories_file = class_dir / "trajectories.json"
            if trajectories_file.exists():
                # Load the entire file once
                with open(trajectories_file, 'r') as f:
                    trajectories = json.load(f)
                # Store each trajectory with its metadata
                for traj_data in trajectories:
                    # Add class_name to the data for easy access
                    traj_data_copy = traj_data.copy()
                    traj_data_copy['class_name'] = actual_class_name
                    self.trajectory_data.append(traj_data_copy)
            else:
                # Fallback: look for individual trajectory_*.json files
                for traj_file in sorted(class_dir.glob("trajectory_*.json")):
                    # Load individual files during init
                    with open(traj_file, 'r') as f:
                        data = json.load(f)
                    data['class_name'] = actual_class_name
                    self.trajectory_data.append(data)

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, idx):
        """
        Returns:
            X: Noisy trajectory (T, 2) as numpy array
            Y: Clean trajectory (T, 2) as numpy array, or None if not available
            class_name: String class name
            dt: Time step
            meta: Dictionary with metadata
        """
        # Data is already loaded, just retrieve it
        data = self.trajectory_data[idx]
        X = np.array(data["X"])
        Y = np.array(data["Y"]) if "Y" in data else None
        dt = data.get("dt", 0.04)
        meta = data.get("meta", {})
        class_name = data.get("class_name", "unknown")

        # Extract vel_change_std and measurement_noise_std from class_name
        # Example: 'cv_0.5_vel_change_0.8_obs_noise'
        # First number is vel_change_std, second number is measurement_noise_std
        numbers = re.findall(r'\d+\.?\d*', class_name)
        if len(numbers) >= 2:
            vel_change_std = float(numbers[0])
            measurement_noise_std = float(numbers[1])
        else:
            # Default values if pattern doesn't match
            vel_change_std = None
            measurement_noise_std = None

        # Normalize per feature (x and y separately) across time dimension
        # This gives shape (2,) for mean/std, one per coordinate
        X_mean = np.mean(X, axis=0)  # (2,)
        X_std = np.std(X, axis=0)  # (2,)
        # Avoid division by zero
        X_std = np.where(X_std < 1e-8, 1.0, X_std)

        X = (X - X_mean) / X_std

        #if Y is not None:
        #    Y_mean = np.mean(Y, axis=0)  # (2,)
        #    Y_std = np.std(Y, axis=0)  # (2,)
        #    Y_std = np.where(Y_std < 1e-8, 1.0, Y_std)
        #    Y = (Y - Y_mean) / Y_std

        return {
            "X": X,
            "Y": Y,
            "class_name": class_name,
            "dt": dt,
            "meta": meta,
            "X_mean": X_mean,
            "X_std": X_std,
            #"Y_mean": Y_mean,
            #"Y_std": Y_std,
            "vel_change_std": vel_change_std,
            "measurement_noise_std": measurement_noise_std
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for trajectory data.

    Args:
        data_dir: Path to the data/ directory
        batch_size: Batch size (default 1, since trajectories have variable lengths)
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance
    """
    dataset = TrajectoryDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None  # Return as-is for variable length sequences
    )




