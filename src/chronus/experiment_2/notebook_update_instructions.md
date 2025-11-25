# Notebook Update Instructions

## Changes Made to dataset_configurations.py

1. Added `create_kf(dt, dim)` method to `CVSpec`, `CASpec`, and `CTSpec`
2. These methods create properly configured Kalman Filters based on the spec's noise parameters

## Changes Needed in uav_bird_motion_models.ipynb

### Step 1: Add a helper cell at the top (after imports)

```python
# Helper function to sample from range
import random
np.random.seed(42)  # For reproducibility

def sample_uniform(low, high):
    """Sample uniformly from range"""
    return np.random.uniform(low, high)
```

### Step 2: Replace all ParamRange() with sampled values

**Example OLD code:**
```python
model_spec=CTSpec(
    omega=ParamRange(0.5, 1.5),  # High turn rate
    omega_noise_std=ParamRange(0.05, 0.3),
    measurement_noise_std=measurement_noise_std
)
```

**Example NEW code:**
```python
model_spec=CTSpec(
    omega=sample_uniform(0.5, 1.5),  # High turn rate
    omega_noise_std=sample_uniform(0.05, 0.3),
    measurement_noise_std=measurement_noise_std
)
```

### Step 3: Update the IMM creation to use create_kf()

**OLD approach (cell 16 - delete this):**
- Manual filter creation with averaging noise parameters

**NEW approach:**
```python
def create_oracle_imm_from_class_config(class_config: ClassConfig, dt: float = 0.04, dim: int = 2):
    """
    Create Oracle IMM from ClassConfig using the segments' create_kf methods.
    """
    spec = class_config.model_spec
    
    # Create KF for each unique model type in segments
    filters = []
    model_types_seen = set()
    
    for seg in spec.segments:
        model_type = seg.model_spec.model_type
        if model_type not in model_types_seen:
            kf = seg.model_spec.create_kf(dt=dt, dim=dim)
            filters.append(kf)
            model_types_seen.add(model_type)
    
    # Set transition matrix based on randomization
    n_filters = len(filters)
    if spec.randomize_blueprint or spec.randomize_order:
        # Highly maneuvering
        stay_prob = 0.70
    else:
        # Moderate maneuvering
        stay_prob = 0.85
    
    switch_prob = (1 - stay_prob) / (n_filters - 1)
    M = np.full((n_filters, n_filters), switch_prob)
    np.fill_diagonal(M, stay_prob)
    
    # Uniform initial probabilities
    mu = np.ones(n_filters) / n_filters
    
    return IMMEstimator(filters, mu, M)

print("✓ Oracle IMM creation function defined")
```

### Step 4: Update cell 27 (organization cell)

```python
# Organize class configs and IMMs for evaluation
class_configs = [
    uav_high_maneuver,
    uav_moderate_maneuver,
    bird_high_maneuver,
    bird_moderate_maneuver,
    generic_extreme_maneuver
]

class_names = [config.name for config in class_configs]

# Create dictionary mapping class names to IMM creation function
# (The function will be called later when needed)
dt = 0.04
dim = 2

print("✓ Organized class configurations")
print(f"  Classes: {class_names}")
```

### Step 5: Update cell 29 (evaluation loop)

```python
# Simply call:
imm = create_oracle_imm_from_class_config(class_config, dt=dt, dim=dim)
```

This way:
- Each segment spec has exact knowledge of its noise parameters
- The `create_kf()` methods use these exact parameters
- No averaging or guessing - true oracle knowledge!













