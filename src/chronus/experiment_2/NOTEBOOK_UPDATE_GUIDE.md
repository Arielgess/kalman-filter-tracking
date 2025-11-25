# Notebook Update Guide - Oracle IMM with Singleton RNG

## Summary of Changes

1. **Created singleton RNG** in `src/dataset/random_gen.py`
2. **Updated `dataset_configurations.py`** to use singleton RNG and return dict-by-class
3. **Simplified `create_kf()` methods** to expect plain float/array values (no ParamRange handling)

## Step 1: Add imports at the top of notebook (after existing imports)

```python
from src.dataset.random_gen import set_seed, sample_uniform
```

## Step 2: Set the global seed ONCE at the beginning

Add this cell early in your notebook (before defining any classes):

```python
# Set global random seed for reproducibility
set_seed(42)
print("✓ Global RNG seed set to 42")
```

## Step 3: Replace ALL ParamRange() with sample_uniform()

### Before (using ParamRange):
```python
model_spec=CTSpec(
    omega=ParamRange(0.5, 1.5),
    omega_noise_std=ParamRange(0.05, 0.3),
    measurement_noise_std=measurement_noise_std
)
```

### After (using sample_uniform):
```python
model_spec=CTSpec(
    omega=sample_uniform(0.5, 1.5),
    omega_noise_std=sample_uniform(0.05, 0.3),
    measurement_noise_std=measurement_noise_std
)
```

## Step 4: Replace IMM creation function

### NEW Oracle IMM Creation Function

```python
def create_oracle_imm_from_class_config(class_config: ClassConfig, dt: float = 0.04, dim: int = 2):
    """
    Create Oracle IMM from ClassConfig using the segments' create_kf methods.
    This is the TRUE oracle - it knows the exact underlying models and noise!
    
    The specs already contain sampled noise values (not ranges), so create_kf()
    uses these exact values.
    """
    spec = class_config.model_spec
    
    # Create KF for each unique model type in segments
    filters = []
    model_types_seen = set()
    
    for seg in spec.segments:
        model_type = seg.model_spec.model_type
        if model_type not in model_types_seen:
            # Use the spec's create_kf method - ORACLE KNOWLEDGE!
            kf = seg.model_spec.create_kf(dt=dt, dim=dim)
            filters.append(kf)
            model_types_seen.add(model_type)
    
    # Set transition matrix based on randomization
    n_filters = len(filters)
    if spec.randomize_blueprint or spec.randomize_order:
        stay_prob = 0.70  # Highly maneuvering
    else:
        stay_prob = 0.85  # Moderate maneuvering
    
    switch_prob = (1 - stay_prob) / (n_filters - 1) if n_filters > 1 else 0
    M = np.full((n_filters, n_filters), switch_prob)
    np.fill_diagonal(M, stay_prob)
    
    # Uniform initial probabilities
    mu = np.ones(n_filters) / n_filters
    
    return IMMEstimator(filters, mu, M)

print("✓ Oracle IMM creation function defined")
```

## Step 5: Update cell 27 (organization cell)

```python
# Organize class configs for evaluation
class_configs = [
    uav_high_maneuver,
    uav_moderate_maneuver,
    bird_high_maneuver,
    bird_moderate_maneuver,
    generic_extreme_maneuver
]

class_names = [config.name for config in class_configs]

dt = 0.04
dim = 2

print("✓ Organized class configurations")
print(f"  Classes: {class_names}")
```

## Step 6: Update cell 29 (evaluation loop)

Replace:
```python
imm = create_imm_for_class(class_config, dt=dt, dim=dim)
```

With:
```python
imm = create_oracle_imm_from_class_config(class_config, dt=dt, dim=dim)
```

## Step 7: Update dataset usage (since it's now dict-by-class)

### Before:
```python
for i, meta in enumerate(full_data['meta']):
    class_name = meta['class']
    X = full_data['X'][i]
```

### After:
```python
for class_name in full_data.keys():
    if class_name == 'config':
        continue
    class_data = full_data[class_name]
    for idx, X in enumerate(class_data['X']):
        # process trajectory
```

Or even simpler:
```python
class_data = full_data[class_name]
X_list = class_data['X']
Y_list = class_data['Y']
meta_list = class_data['meta']
```

## Key Benefits of This Approach

1. **True Oracle**: Each KF is created with the EXACT noise values used to generate the data
2. **Single RNG Source**: All randomness comes from one seeded singleton
3. **Reproducible**: Set seed once, everything follows deterministically
4. **Cleaner Code**: No ParamRange handling in create_kf(), just plain values
5. **Dict-by-Class Dataset**: Easier to work with class-specific data

## Quick Find-Replace Guide

In your notebook, do these replacements:

1. `ParamRange(` → `sample_uniform(`
2. `from src.dataset.dataset_configurations import (\n    ClassConfig, DatasetConfig, IMMSpec, SegmentSpec,\n    CVSpec, CASpec, CTSpec, ParamRange\n)` → `from src.dataset.dataset_configurations import (\n    ClassConfig, DatasetConfig, IMMSpec, SegmentSpec,\n    CVSpec, CASpec, CTSpec\n)\nfrom src.dataset.random_gen import set_seed, sample_uniform`
3. Add `set_seed(42)` cell early on
4. Replace `create_imm_for_class` function with `create_oracle_imm_from_class_config`
5. Update dataset access to use dict-by-class structure

## Example: Complete Class Definition (Updated)

```python
# Class 1: UAV - Highly Maneuvering
measurement_noise_std = np.array([0.3, 0.3])

uav_high_maneuver = ClassConfig(
    name="UAV_HighlyManeuvering",
    model_spec=IMMSpec(
        segments=[
            # Sharp CT turn
            SegmentSpec(
                model_spec=CTSpec(
                    omega=sample_uniform(0.5, 1.5),  # Sampled once here!
                    omega_noise_std=sample_uniform(0.05, 0.3),  # Sampled once here!
                    measurement_noise_std=measurement_noise_std
                ),
                T=25
            ),
            # CA acceleration
            SegmentSpec(
                model_spec=CASpec(
                    accel_noise_std=np.array([sample_uniform(0.3, 0.5), sample_uniform(0.3, 0.5)]),
                    acceleration=np.array([sample_uniform(1.5, 3.0), sample_uniform(1.5, 3.0)]),
                    measurement_noise_std=measurement_noise_std
                ),
                T=30
            ),
            # CV cruise
            SegmentSpec(
                model_spec=CVSpec(
                    vel_change_std=np.array([sample_uniform(0.4, 0.6), sample_uniform(0.4, 0.6)]),
                    measurement_noise_std=measurement_noise_std
                ),
                T=20
            ),
        ],
        randomize_blueprint=True,
        min_segment_length=15,
        max_segment_length=35
    ),
    n_trajectories=100
)
```

Now the IMM created with `create_oracle_imm_from_class_config(uav_high_maneuver)` will use the EXACT sampled values!













