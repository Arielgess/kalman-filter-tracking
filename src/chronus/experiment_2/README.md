# Experiment 2: UAV and Bird Motion Model Classes

This experiment defines 5 distinct motion model classes for tracking Unmanned Aerial Vehicles (UAVs) and birds, each with tailored Interacting Multiple Model (IMM) configurations.

## Overview

### Motion Model Classes

1. **UAV_HighlyManeuvering**: Combat/racing drones
   - Sharp CT turns (ω = 0.5-1.5 rad/s)
   - Aggressive CA accelerations (±3 m/s²)
   - Blueprint randomization for unpredictable maneuvers
   - High process noise (σ_v = 0.4-0.6 m/s²)

2. **UAV_ModerateManeuvering**: Surveillance/delivery drones
   - Predominantly CV motion (steady cruise)
   - Gentle CT turns (ω = 0.1-0.3 rad/s)
   - Mild CA adjustments (±0.5 m/s²)
   - Low process noise (σ_v = 0.1 m/s²)

3. **Bird_HighlyManeuvering**: Predatory birds (falcon/swallow)
   - Banking CT turns (ω = 0.6-1.2 rad/s)
   - Diving/climbing CA segments (-4 to +4 m/s²)
   - Order randomization for erratic flight
   - High biological noise (σ_v = 0.5-0.8 m/s²)

4. **Bird_ModerateManeuvering**: Gliding birds (pigeon/seagull)
   - Long CV gliding segments
   - Gentle CT banking (ω = 0.15-0.35 rad/s)
   - Mild CA altitude changes (±0.8 m/s²)
   - Moderate noise (σ_v = 0.2 m/s²)

5. **Generic_ExtremeManeuvers**: Highly maneuvering objects
   - All models (CV, CA, CT) with aggressive parameters
   - Extreme turns (ω = 1.0-2.0 rad/s)
   - Rapid acceleration (±5-6 m/s²)
   - Blueprint randomization with very short segments (10-25 steps)
   - Maximum noise (σ_v = 0.7-1.0 m/s²)

## New Features Implemented

### Trajectory Randomization

Two new randomization modes added to `generate_composite_trajectory`:

1. **Simple Order Randomization** (`randomize_order=True`)
   - Shuffles predefined segment order
   - Useful for erratic patterns (e.g., bird hunting behavior)

2. **Blueprint-Based Randomization** (`randomize_blueprint=True`)
   - Treats segments as templates
   - Generates trajectory by randomly selecting segments with random lengths
   - Continues until target length T is reached
   - Fills remainder with CV if < min_length remaining
   - Maximizes unpredictability for highly maneuvering targets

### IMM Tracking Configurations

Each class has a corresponding IMM estimator with:
- **CV, CA, and CT Kalman Filters** with noise parameters derived from segment specifications
- **Mode transition probabilities** tuned to maneuverability:
  - Highly maneuvering: 70% stay in mode, 15% transition (frequent switching)
  - Moderate maneuvering: 85% stay in mode, 7-8% transition (stable flight)
- **Initial model probabilities**: 60% CV, 20% CA, 20% CT

## Files

- `uav_bird_motion_models.ipynb`: Main notebook with all class definitions and justifications
- `README.md`: This file

## Usage

```python
from src.dataset.dataset_configurations import generate_dataset, DatasetConfig

# Generate trajectories for all 5 classes
data = generate_dataset(dataset_config)

# Use corresponding IMM for tracking
imm_estimator = create_imm_for_class(uav_high_maneuver)
```

## References

- Li, X. R., & Jilkov, V. P. (2003). "Survey of maneuvering target tracking. Part I: Dynamic models."
- Bar-Shalom, Y., et al. (2004). "Tracking and Data Fusion: A Handbook of Algorithms."
- [Drone Racing Research (2019)](https://ieeexplore.ieee.org/document/8794064)
- [Falcon Flight Biomechanics (2011)](https://doi.org/10.1098/rsif.2011.0239)
- [Swallow Maneuverability (2014)](https://doi.org/10.1242/jeb.104901)
- [Ornithopter Flight Dynamics (2017)](https://doi.org/10.2514/1.C034320)
- [Dragonfly Maneuverability (2013)](https://doi.org/10.1073/pnas.1214359110)

