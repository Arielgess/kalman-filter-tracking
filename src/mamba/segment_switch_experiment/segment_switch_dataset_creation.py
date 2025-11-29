from src.dataset.dataset_configurations import (
    ClassConfig, DatasetConfig, IMMSpec, SegmentSpec,
    CVSpec, CASpec, CTSpec, ParamRange
)
from src.dataset.random_gen import random_generator
from src.dataset.dataset_configurations import generate_dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any


def training_config(dim: int = 2) -> DatasetConfig:
    """
    The 'Soup': A massive dataset of randomized physics and randomized noise levels.
    Goal: Teach Mamba general tracking and reactivity.
    """
    
    # --- Blueprints (The Ingredients) ---
    # Note: We put dummy '0.0' for measurement_noise_std here because 
    # the IMMSpec will OVERRIDE it with the global value.
    
    blueprints = [
        # 1. Constant Velocity (Quiet & Noisy Process)
        SegmentSpec(CVSpec(
            vel_change_std=ParamRange(0.01, 0.6),       # Physics Range
            measurement_noise_std=0.0                   # Ignored
        ), T=0),
        
        # 2. Coordinated Turns (Left & Right)
        SegmentSpec(CTSpec(
            omega=ParamRange(0.1, 2.0),                 # Variable Turn Rates
            omega_noise_std=ParamRange(0.001, 0.2),
            measurement_noise_std=0.0                   # Ignored
        ), T=0),
        SegmentSpec(CTSpec(
            omega=ParamRange(-2.0, -0.1),
            omega_noise_std=ParamRange(0.001, 0.2),
            measurement_noise_std=0.0                   # Ignored
        ), T=0),
        
        # 3. Constant Acceleration (Accel & Decel)
        SegmentSpec(CASpec(
            acceleration=ParamRange(0.2, 4.0),          # Positive Accel
            accel_noise_std=ParamRange(0.01, 0.35),
            measurement_noise_std=0.0                   # Ignored
        ), T=0),
        SegmentSpec(CASpec(
            acceleration=ParamRange(-4.0, -0.2),        # Negative Accel (Decel)
            accel_noise_std=ParamRange(0.01, 0.35),
            measurement_noise_std=0.0                   # Ignored
        ), T=0)
    ]

    # --- The Soup Class ---
    soup_class = ClassConfig(
        name="Universal_Physics",
        model_spec=IMMSpec(
            segments=blueprints,
            randomize_blueprint=True,
            min_segment_length=30,      # Long enough to see physics
            max_segment_length=100,     # Long enough to stabilize
            # THE KEY: Variable noise between trajectories, but constant WITHIN a trajectory
            # This trains the model to adapt its 'gain' to the noise level
            measurement_noise_std=ParamRange(0.15, 0.45) 
        ),
        n_trajectories=55000
    )

    return DatasetConfig(
        seed=42,
        dim=dim,
        dt=0.04,
        T=500,  # Long trajectories to allow multiple switches
        classes=[soup_class],
        store_clean=True
    )

def save_dataset_to_files(dataset: Dict[str, Any], output_dir: str = "data", indent: int = 2):
    """
    Save dataset to files in the format specified in DATA_FORMAT.md.
    
    Args:
        dataset: Dictionary returned from generate_dataset() with structure:
            {
                "ClassName1": {"X": [...], "Y": [...], "meta": [...]},
                "ClassName2": {"X": [...], "Y": [...], "meta": [...]},
                ...
                "config": {...}
            }
        output_dir: Root directory where data/ folder will be created
        indent: JSON indentation level (2 for readable, None for compact). 
                For large datasets (1000+ trajectories), consider using indent=None 
                to reduce file size by ~30-40%
    
    File Size Notes:
        - 1000 trajectories × 300 timesteps ≈ 10-15 MB per class (with indent=2)
        - 1000 trajectories × 300 timesteps ≈ 6-9 MB per class (with indent=None)
        - JSON format is human-readable but not the most efficient
        - For very large datasets, consider using compressed formats (npz, hdf5) instead
    """
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class names (exclude "config")
    class_names = [name for name in dataset.keys() if name != "config"]
    
    # Sort class names to ensure consistent ordering
    class_names.sort()
    
    print(f"Saving dataset with {len(class_names)} classes to {data_dir}")
    
    for class_idx, class_name in enumerate(class_names):
        class_data = dataset[class_name]
        X_list = class_data["X"]
        Y_list = class_data.get("Y")
        meta_list = class_data["meta"]
        
        # Create class directory with index prefix
        class_dir_name = f"class_{class_idx}_{class_name}"
        class_dir = data_dir / class_dir_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract model_type and params from first trajectory's meta
        if len(meta_list) == 0:
            raise ValueError(f"No trajectories found for class {class_name}")
        
        first_meta = meta_list[0]
        model_type = first_meta.get("model_type", "IMM")
        params = first_meta.get("params", {})
        
        # Create and save metadata.json
        metadata = {
            "model_type": model_type,
            "params": params
        }
        metadata_path = class_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Collect all trajectories for this class
        trajectories = []
        for traj_idx, (X, meta) in enumerate(zip(X_list, meta_list)):
            # Extract dt from meta
            dt = meta.get("dt", 0.04)
            
            # Extract measurement_noise_std from params
            params = meta.get("params", {})
            measurement_noise_std = None
            
            # Extract measurement_noise_std based on model type
            model_type = meta.get("model_type", "IMM")
            if model_type == "IMM":
                # For IMM models, measurement_noise_std is in the first segment's params
                # (all segments have the same value)
                if "segments" in params and len(params["segments"]) > 0:
                    first_segment_params = params["segments"][0].get("params", {})
                    measurement_noise_std = first_segment_params.get("measurement_noise_std")
            else:
                # For non-IMM models, it's directly in params
                measurement_noise_std = params.get("measurement_noise_std")
            
            # Build initial_state dictionary, including acceleration if present
            initial_state_meta = meta.get("initial_state", {})
            initial_state = {
                "position": initial_state_meta.get("position", []),
                "velocity": initial_state_meta.get("velocity", [])
            }
            # Add acceleration only if it exists and is not None
            if "acceleration" in initial_state_meta and initial_state_meta["acceleration"] is not None:
                initial_state["acceleration"] = initial_state_meta["acceleration"]
            
            # Prepare trajectory data
            traj_data = {
                "class_name": class_name,
                "trajectory_index": traj_idx,
                "dt": dt,
                "X": X.tolist() if isinstance(X, np.ndarray) else X,
                "meta": {
                    "initial_state": initial_state,
                    "measurement_noise_std": measurement_noise_std
                }
            }
            
            # Add Y if available
            if Y_list is not None and traj_idx < len(Y_list):
                Y = Y_list[traj_idx]
                traj_data["Y"] = Y.tolist() if isinstance(Y, np.ndarray) else Y
            
            trajectories.append(traj_data)
        
        # Save all trajectories in a single file
        trajectories_path = class_dir / "trajectories.json"
        print(f"  Saving {len(trajectories)} trajectories for class {class_name}...", end=" ")
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=indent)
        
        # Calculate and display file size
        file_size_mb = trajectories_path.stat().st_size / (1024 * 1024)
        print(f"({file_size_mb:.2f} MB)")
    
    print(f"\n✓ Dataset saved to {data_dir}")


if __name__ == '__main__':
    dataset_config = training_config(dim=2)
    
    print("Generating dataset with configuration:")
    print(f"  Classes: {len(dataset_config.classes)}")
    for cls in dataset_config.classes:
        print(f"    - {cls.name}: {cls.n_trajectories} trajectories")
        if hasattr(cls.model_spec, 'segments'):
            print(f"      Segments: {len(cls.model_spec.segments)}")
            print(f"      Randomize blueprint: {cls.model_spec.randomize_blueprint}")
            print(f"      Segment length range: [{cls.model_spec.min_segment_length}, {cls.model_spec.max_segment_length}]")
            print(f"      Target T: {dataset_config.T}")
    
    dataset = generate_dataset(dataset_config)
    
    save_dataset_to_files(dataset)

