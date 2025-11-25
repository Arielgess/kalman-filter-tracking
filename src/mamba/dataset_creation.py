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


def create_class_config_high_maneuver(measurement_noise_std: np.ndarray, n_trajectories: int) -> ClassConfig:
    class_config = ClassConfig(
        name=f"HighlyManeuveringTargetSharpTurns_{measurement_noise_std[0]}_obs_noise",
        model_spec=IMMSpec(
            segments=[
                # Sharp CT turn
                SegmentSpec(
                    model_spec=CTSpec(
                        # omega=rng.uniform(0.5, 1.5),  # High turn rate: 28-86 degrees/sec
                        # omega_noise_std=rng.uniform(0.05, 0.3),  # Turn rate variability
                        omega=1.2,
                        omega_noise_std=0.2,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=100
                ),
                # Rapid acceleration
                SegmentSpec(
                    model_spec=CASpec(
                        # accel_noise_std=rng.uniform(0.3, 0.5, size=dim),  # High acceleration variability
                        # acceleration=rng.uniform(-3.0, 3.0, size=dim),  # ±3 m/s² (aggressive)
                        acceleration=np.array([2, 2]),
                        accel_noise_std=np.array([0.4, 0.4]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=50
                ),
                # Brief steady flight
                SegmentSpec(
                    model_spec=CVSpec(
                        # vel_change_std=rng.uniform(0.4, 0.6, size=dim),  # High process noise
                        vel_change_std=np.array([0.5, 0.5]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=50
                ),
                # Another sharp turn (opposite direction)
                SegmentSpec(
                    model_spec=CTSpec(
                        # omega=rng.uniform(-1.5, -0.5),  # Negative omega (opposite turn)
                        # omega_noise_std=rng.uniform(0.05, 0.3),
                        omega=-1.2,
                        omega_noise_std=0.2,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=100
                ),
            ],
            randomize_blueprint=True,  # Unpredictable maneuver sequences
            min_segment_length=15,
            max_segment_length=35,
            measurement_noise_std=np.array([0.5, 0.5])  # Required by ModelSpec base class
        ),
        n_trajectories=n_trajectories
    )
    return class_config


def create_class_config_moderate_maneuver(measurement_noise_std: np.ndarray, n_trajectories: int) -> ClassConfig:
    class_config = ClassConfig(
        name=f"ModerateManeuveringTargetWithTurns_{measurement_noise_std[0]}_obs_noise",
        model_spec=IMMSpec(
            segments=[
                # Long steady cruise
                SegmentSpec(
                    model_spec=CVSpec(
                        # vel_change_std=rng.uniform(0.1, 0.15, size=dim),  # Low process noise
                        vel_change_std=np.array([0.125, 0.125]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=100
                ),
                # Gentle turn
                SegmentSpec(
                    model_spec=CTSpec(
                        # omega=rng.uniform(0.1, 0.3),  # Low turn rate: 6-17 degrees/sec
                        omega=0.2,
                        # omega_noise_std=rng.uniform(0.01, 0.03),  # Stable turn control
                        omega_noise_std=0.02,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=50
                ),
                # Mild acceleration (speed adjustment)
                SegmentSpec(
                    model_spec=CASpec(
                        # accel_noise_std=rng.uniform(0.08, 0.12, size=dim),  # Low variability
                        accel_noise_std=np.array([0.1, 0.1]),
                        # acceleration=rng.uniform(-0.5, 0.5, size=dim),  # Gentle: ±0.5 m/s²
                        acceleration=np.array([0.0, 0.0]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=50
                ),
                # Return to steady cruise
                SegmentSpec(
                    model_spec=CVSpec(
                        # vel_change_std=rng.uniform(0.1, 0.15, size=dim),
                        vel_change_std=np.array([0.125, 0.125]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=100
                ),
            ],
            randomize_order=False,  # Predictable flight pattern
            randomize_blueprint=False,
            measurement_noise_std=measurement_noise_std
        ),
        n_trajectories=n_trajectories
    )
    return class_config


def create_class_config_bursts(measurement_noise_std: np.ndarray, n_trajectories: int) -> ClassConfig:
    class_config = ClassConfig(
        name=f"AgileTargetErraticTurnsAndBursts_{measurement_noise_std[0]}_obs_noise",
        model_spec=IMMSpec(
            segments=[
                # Banking turn (pursuit)
                SegmentSpec(
                    model_spec=CTSpec(
                        #omega=rng.uniform(0.6, 1.4),  # 34-69 degrees/sec (agile turn)
                        omega=1.0,
                        #omega_noise_std=rng.uniform(0.08, 0.15),  # Wing-flapping variability
                        omega_noise_std=0.12,
                        measurement_noise_std=measurement_noise_std  # Smaller target
                    ),
                    T=40
                ),
                # Diving acceleration (attack)
                SegmentSpec(
                    model_spec=CASpec(
                        #accel_noise_std=rng.uniform(0.4, 0.6, size=dim),  # High biological variability
                        accel_noise_std=np.array([0.5, 0.5]),
                        #acceleration=rng.uniform(-4.0, -1.0, size=dim),  # Diving: -1 to -4 m/s²
                        acceleration=np.array([-2.5, -2.5]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=20
                ),
                # Rapid climb (escape)
                SegmentSpec(
                    model_spec=CASpec(
                        #accel_noise_std=rng.uniform(0.4, 0.6, size=dim),
                        accel_noise_std=np.array([0.5, 0.5]),
                        #acceleration=rng.uniform(1.5, 4.0, size=dim),  # Climbing: +1.5 to +4 m/s²
                        acceleration=np.array([2.5, 2.5]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=100
                ),
                # Brief gliding
                SegmentSpec(
                    model_spec=CVSpec(
                        #vel_change_std=rng.uniform(0.5, 0.8, size=dim),  # High flapping noise
                        vel_change_std=np.array([0.65, 0.65]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Another banking turn
                SegmentSpec(
                    model_spec=CTSpec(
                        #omega=rng.uniform(-1.2, -0.6),  # Opposite direction turn
                        omega=-0.9,
                        #omega_noise_std=rng.uniform(0.08, 0.15),
                        omega_noise_std=0.12,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=80
                ),
            ],
            randomize_order=True,  # Erratic, unpredictable flight
            randomize_blueprint=False,
            measurement_noise_std=measurement_noise_std
        ),
        n_trajectories=n_trajectories
    )
    return class_config


def create_class_gentle_turns(measurement_noise_std: np.ndarray, n_trajectories: int) -> ClassConfig:
    class_config = ClassConfig(
        name=f"GlidingTargetGentleTurns_{measurement_noise_std[0]}_obs_noise",
        model_spec=IMMSpec(
            segments=[
                # Long gliding phase
                SegmentSpec(
                    model_spec=CVSpec(
                        #vel_change_std=rng.uniform(0.18, 0.22, size=dim),  # Moderate flapping noise
                        vel_change_std=np.array([0.20, 0.20]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=150
                ),
                # Gentle banking turn
                SegmentSpec(
                    model_spec=CTSpec(
                        #omega=rng.uniform(0.15, 0.35),  # 9-20 degrees/sec
                        omega=0.25,
                        omega_noise_std=0.03,  # Low turn variability
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=50
                ),
                # Mild altitude change
                SegmentSpec(
                    model_spec=CASpec(
                        #accel_noise_std=rng.uniform(0.10, 0.14, size=dim),  # Low acceleration noise
                        accel_noise_std=np.array([0.12, 0.12]),
                        #acceleration=rng.uniform(-0.8, 0.8, size=dim),  # Gentle: ±0.8 m/s²
                        acceleration=np.array([1.0, 1.0]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=40
                ),
                # Return to gliding
                SegmentSpec(
                    model_spec=CVSpec(
                        #vel_change_std=rng.uniform(0.2, 0.22, size=dim),
                        vel_change_std=np.array([0.2, 0.2]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
            ],
            randomize_order=False,  # Predictable patterns
            randomize_blueprint=False,
            measurement_noise_std=measurement_noise_std
        ),
        n_trajectories=n_trajectories
    )
    return class_config


def create_class_rapid_mode_switching(measurement_noise_std: np.ndarray, n_trajectories: int) -> ClassConfig:
    class_config = ClassConfig(
        name=f"ExtremeManeuveringTargetRapidModeSwitching_{measurement_noise_std[0]}_obs_noise",
        model_spec=IMMSpec(
            segments=[
                # Extreme turn #1
                SegmentSpec(
                    model_spec=CTSpec(
                        #omega=rng.uniform(1.0, 2.0),  # Very high: 57-115 degrees/sec
                        omega=2.0,
                        #omega_noise_std=rng.uniform(0.1, 0.2),  # High variability
                        omega_noise_std=0.15,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Extreme acceleration
                SegmentSpec(
                    model_spec=CASpec(
                        #accel_noise_std=rng.uniform(0.6, 0.9, size=dim),  # Very high noise
                        accel_noise_std=np.array([0.75, 0.75]),
                        #acceleration=rng.uniform(-5.0, 5.0, size=dim),  # Extreme: ±5 m/s²
                        acceleration=np.array([4.0, 4.0]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Chaotic constant velocity
                SegmentSpec(
                    model_spec=CVSpec(
                        #vel_change_std=rng.uniform(0.7, 1.0, size=dim),
                        vel_change_std=np.array([0.85, 0.85]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Extreme turn #2 (opposite direction)
                SegmentSpec(
                    model_spec=CTSpec(
                        #omega=rng.uniform(-2.0, -1.0),  # Opposite direction
                        omega=-1.5,
                        #omega_noise_std=rng.uniform(0.1, 0.2),
                        omega_noise_std=0.15,
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Rapid deceleration
                SegmentSpec(
                    model_spec=CASpec(
                        #accel_noise_std=rng.uniform(0.6, 0.9, size=dim),
                        accel_noise_std=np.array([0.75, 0.75]),
                        #acceleration=rng.uniform(-6.0, -2.0, size=dim),  # Strong braking
                        acceleration=np.array([-4.0, -4.0]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
                # Brief CV before next maneuver
                SegmentSpec(
                    model_spec=CVSpec(
                        #vel_change_std=rng.uniform(0.7, 1.0, size=dim),
                        vel_change_std=np.array([0.85, 0.85]),
                        measurement_noise_std=measurement_noise_std
                    ),
                    T=60
                ),
            ],
            randomize_blueprint=True,  # Maximum unpredictability
            min_segment_length=20,  # Very short segments
            max_segment_length=40,  # Frequent mode changes
            measurement_noise_std=measurement_noise_std
        ),
        n_trajectories=n_trajectories
    )
    return class_config


class_config_methods = [
    create_class_config_high_maneuver,
    create_class_config_moderate_maneuver,
    create_class_config_bursts,
    create_class_gentle_turns,
    create_class_rapid_mode_switching
    ]


def create_dataset_config(
        dim: int,
        seed: int,
        dt: float,
        T: int,
        measurement_noises: list[np.ndarray],
        n_trajectories_per_class: int):
    class_configs = []
    for measurement_noise_std in measurement_noises:
        for class_config_method in class_config_methods:
            class_configs.append(class_config_method(measurement_noise_std=measurement_noise_std,
                                                     n_trajectories=n_trajectories_per_class))
    dataset_config = DatasetConfig(
        seed=seed,
        dim=dim,
        dt=dt,
        T=T,  # Target trajectory length for blueprint mode
        init_pos_range=(-50.0, 50.0),
        init_speed_range=(-20, 20.0),  # Reasonable flight speeds
        classes=class_configs,
        store_clean=True,
    )

    return dataset_config


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
                    "initial_state": initial_state
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
    dataset_config = create_dataset_config(
        dim=2,
        seed=43,
        dt=0.04,
        T=300,
        measurement_noises=[np.array([0.1, 0.1]), np.array([0.5, 0.5]), np.array([0.8, 0.8]), np.array([1.5, 1.5])],
        n_trajectories_per_class=1000
    )

    dataset = generate_dataset(dataset_config)

    save_dataset_to_files(dataset)
