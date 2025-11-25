# Data Format Specification

## Directory Structure

```
data/
├── class_0_HighlyManeuveringTargetSharpTurns_0.2_obs_noise/
│   ├── metadata.json
│   └── trajectories.json
├── class_1_ModerateManeuveringTargetWithTurns_0.2_obs_noise/
│   ├── metadata.json
│   └── trajectories.json
├── class_2_AgileTargetErraticTurnsAndBursts_0.2_obs_noise/
│   ├── metadata.json
│   └── trajectories.json
├── class_3_GlidingTargetGentleTurns_0.2_obs_noise/
│   ├── metadata.json
│   └── trajectories.json
└── class_4_ExtremeManeuveringTargetRapidModeSwitching_0.2_obs_noise/
    ├── metadata.json
    └── trajectories.json
```

## Class Metadata File

Each class directory contains a `metadata.json` file with model configuration:

```json
{
  "model_type": "IMM",
  "params": {
    "segments": [...],
    "randomize_blueprint": true,
    "min_segment_length": 15,
    "max_segment_length": 35,
    "measurement_noise_std": [0.2, 0.2]
  }
}
```

## Trajectories JSON File Format

Each class directory contains a `trajectories.json` file with a list of all trajectories for that class:

```json
[
  {
    "class_name": "HighlyManeuveringTargetSharpTurns_0.2_obs_noise",
    "trajectory_index": 0,
    "dt": 0.04,
    "X": [[x1, y1], [x2, y2], ...],
    "Y": [[x1_clean, y1_clean], [x2_clean, y2_clean], ...],
    "meta": {
      "initial_state": {
        "position": [x0, y0],
        "velocity": [vx0, vy0],
        "acceleration": [ax0, ay0]
      },
      "measurement_noise_std": [0.2, 0.2]
    }
  },
  {
    "class_name": "HighlyManeuveringTargetSharpTurns_0.2_obs_noise",
    "trajectory_index": 1,
    "dt": 0.04,
    "X": [[x1, y1], [x2, y2], ...],
    "Y": [[x1_clean, y1_clean], [x2_clean, y2_clean], ...],
    "meta": {
      "initial_state": {
        "position": [x0, y0],
        "velocity": [vx0, vy0]
      },
      "measurement_noise_std": [0.2, 0.2]
    }
  },
  ...
]
```

## Notes

- **Class names**: Include the observation noise value in the format `{ClassName}_{noise_value}_obs_noise`
- **metadata.json**: Contains model type and parameters shared across all trajectories in the class
- **trajectories.json**: Contains a JSON array of all trajectories for the class
- **X**: Noisy 2D position measurements (T, 2) - required
- **Y**: Clean 2D positions (T, 2) - optional
- **meta**: Per-trajectory metadata (initial state, measurement noise)
- **initial_state.position**: Initial position [x0, y0] - required
- **initial_state.velocity**: Initial velocity [vx0, vy0] - required
- **initial_state.acceleration**: Initial acceleration [ax0, ay0] - optional (only included if not None)

## DataLoader Compatibility

This format is well-suited for PyTorch DataLoader usage:

- **Dataset Implementation**: Create a custom `Dataset` class that:
  - Scans all class directories to collect class file paths
  - Loads class metadata once (can be cached)
  - Loads `trajectories.json` files on-demand (can cache per class)
  - Accesses individual trajectories from the loaded array in `__getitem__`
  
- **Efficient Loading**: 
  - All trajectories for a class are in a single JSON file, enabling batch loading per class
  - Class-level metadata can be loaded once and reused for all trajectories in that class
  - Consider caching loaded `trajectories.json` files in memory if memory allows
  
- **Batching Considerations**:
  - Trajectories may have variable lengths (T varies)
  - Use `collate_fn` to handle padding/truncation for batching
  - Consider grouping by trajectory length or using dynamic batching
  
- **Example Structure**:
  ```python
  class TrajectoryDataset(Dataset):
      def __init__(self, data_root):
          # Collect all class directories
          # Load class metadata
          # Optionally pre-load trajectories.json files
          pass
      
      def __getitem__(self, idx):
          # Access trajectory from loaded trajectories.json array
          # Return X, Y (if available), class_id, etc.
          pass
  ```




