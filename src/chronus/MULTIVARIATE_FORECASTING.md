# Multivariate Forecasting with Chronos-2

## TL;DR

**YES! Chronos-2 supports multivariate forecasting.** You can predict X and Y (and Z) **jointly** instead of separately!

## What's the Difference?

### Univariate Approach (What I Initially Showed)
```python
# Predict X dimension separately
df_x = pd.DataFrame({'item_id': ..., 'timestamp': ..., 'target': x_data})
pred_x = pipeline.predict_df(df_x, target='target')

# Predict Y dimension separately  
df_y = pd.DataFrame({'item_id': ..., 'timestamp': ..., 'target': y_data})
pred_y = pipeline.predict_df(df_y, target='target')
```

❌ **Problems:**
- X and Y predicted independently
- Doesn't capture correlations between dimensions
- Less efficient (2 model calls)
- Can't learn that X and Y movements are related

### Multivariate Approach (BETTER!)
```python
# Predict X and Y TOGETHER
df = pd.DataFrame({
    'item_id': ['trajectory'] * T,
    'timestamp': timestamps,
    'x': x_data,  # X dimension
    'y': y_data   # Y dimension
})

pred_df = pipeline.predict_df(
    df,
    prediction_length=1,
    target=['x', 'y'],  # ← Multiple targets!
    id_column='item_id',
    timestamp_column='timestamp'
)

# Output has both predictions: pred_df['x[0.5]'] and pred_df['y[0.5]']
```

✅ **Benefits:**
- Captures X-Y correlations
- More efficient (1 model call)
- Better for trajectories where dimensions are related
- Can model complex multivariate patterns

## How It Works

Chronos-2's multivariate model learns:
1. **Individual patterns**: How X and Y each evolve
2. **Cross-dependencies**: How X affects Y and vice versa
3. **Joint dynamics**: Combined motion patterns

For example:
- If X is increasing, Y might also tend to increase (positive correlation)
- Circular or curved trajectories have strong X-Y coupling
- Vehicle dynamics where lateral and longitudinal movements are related

## Code Example: Full Workflow

```python
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline

# Load model
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Your 2D trajectory data
noisy_trajectory = ...  # Shape: (T, 2)
clean_trajectory = ...  # Shape: (T, 2)

# Prepare multivariate DataFrame
timestamps = pd.date_range(start='2025-01-01', periods=T, freq='0.04S')

df_observed = pd.DataFrame({
    'item_id': ['traj_1'] * T,
    'timestamp': timestamps,
    'x': noisy_trajectory[:, 0],  # X coordinate (noisy)
    'y': noisy_trajectory[:, 1]   # Y coordinate (noisy)
})

df_clean = pd.DataFrame({
    'item_id': ['traj_1'] * T,
    'timestamp': timestamps,
    'x': clean_trajectory[:, 0],  # X coordinate (clean)
    'y': clean_trajectory[:, 1]   # Y coordinate (clean)
})

# Iterative 1-step ahead forecasting
context_length = 50
predictions_x = []
predictions_y = []
clean_x = []
clean_y = []

for i in range(context_length, len(df_observed)):
    # Context includes BOTH X and Y history
    context_df = df_observed.iloc[:i].copy()
    
    # Predict BOTH X and Y simultaneously
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=1,
        quantile_levels=[0.5],
        id_column='item_id',
        timestamp_column='timestamp',
        target=['x', 'y']  # ← Joint prediction!
    )
    
    # Extract predictions (note the column names!)
    predictions_x.append(pred_df['x[0.5]'].values[0])
    predictions_y.append(pred_df['y[0.5]'].values[0])
    clean_x.append(df_clean.iloc[i]['x'])
    clean_y.append(df_clean.iloc[i]['y'])

# Calculate 2D position errors
predictions_x = np.array(predictions_x)
predictions_y = np.array(predictions_y)
clean_x = np.array(clean_x)
clean_y = np.array(clean_y)

position_errors = np.sqrt((clean_x - predictions_x)**2 + 
                         (clean_y - predictions_y)**2)

print(f"Mean 2D position error: {position_errors.mean():.4f}")
```

## Output Column Names

When you use multivariate forecasting, the prediction DataFrame columns follow this pattern:

```python
# For target=['x', 'y'] and quantile_levels=[0.1, 0.5, 0.9]:
pred_df.columns:
# ['item_id', 'timestamp', 
#  'x[0.1]', 'x[0.5]', 'x[0.9]',    # X predictions at different quantiles
#  'y[0.1]', 'y[0.5]', 'y[0.9]']    # Y predictions at different quantiles

# Access predictions:
x_pred = pred_df['x[0.5]'].values[0]  # X median prediction
y_pred = pred_df['y[0.5]'].values[0]  # Y median prediction
```

## When to Use Multivariate vs Univariate?

### Use Multivariate When:
✅ Dimensions are correlated (X and Y movements are related)  
✅ You have circular, curved, or coordinated motion  
✅ Trajectory dynamics couple dimensions (e.g., vehicle turning)  
✅ You want to capture cross-dimensional dependencies  
✅ Efficiency matters (fewer model calls)

### Use Univariate When:
❌ Dimensions are truly independent  
❌ Different noise characteristics per dimension  
❌ You want simpler, more interpretable models  
❌ Debugging/analysis of individual dimensions

## Performance Comparison

In the notebook, Cell 19 compares both approaches:

| Metric | Univariate | Multivariate |
|--------|------------|--------------|
| Predictions | X and Y separate | X and Y joint |
| Model Calls | 2 per timestep | 1 per timestep |
| Speed | Slower (2x calls) | Faster (1 call) |
| Correlations | ❌ Not captured | ✅ Captured |
| Best For | Independent dimensions | Coupled dimensions |

## Real-World Applications

### 1. Vehicle Tracking
```python
target=['x', 'y']  # Position coordinates
# Captures: Turning dynamics, coordinated motion
```

### 2. Aircraft Tracking
```python
target=['x', 'y', 'z']  # 3D position
# Captures: Altitude changes during turns
```

### 3. Robot Arm Control
```python
target=['joint1', 'joint2', 'joint3']
# Captures: Coordinated joint movements
```

### 4. Financial Time Series
```python
target=['stock_A', 'stock_B']
# Captures: Co-movement of related stocks
```

## Extending to 3D

For 3D trajectories:

```python
df = pd.DataFrame({
    'item_id': ['traj_3d'] * T,
    'timestamp': timestamps,
    'x': trajectory[:, 0],
    'y': trajectory[:, 1],
    'z': trajectory[:, 2]
})

pred_df = pipeline.predict_df(
    df,
    target=['x', 'y', 'z'],  # All three dimensions!
    ...
)

# Access predictions
x_pred = pred_df['x[0.5]'].values[0]
y_pred = pred_df['y[0.5]'].values[0]
z_pred = pred_df['z[0.5]'].values[0]
```

## Limitations

1. **All targets must have same length** (same number of timesteps)
2. **All targets share the same timestamps** (synchronized data)
3. **Mixed frequencies not supported** (all variables at same sampling rate)
4. **More targets = slower** (but still better than separate calls)

## Summary

**YES, use multivariate forecasting for trajectory data!** It's:
- ✅ More accurate (captures correlations)
- ✅ More efficient (fewer model calls)
- ✅ Better suited for tracking applications
- ✅ Already implemented in your notebook (Cell 17)

The univariate approach I initially showed works, but multivariate is the recommended approach for X-Y (or X-Y-Z) trajectory forecasting!

## See It In Action

Run these cells in your notebook:
- **Cell 15**: Univariate (separate X, Y)
- **Cell 17**: Multivariate (joint X-Y) ← **Recommended!**
- **Cell 19**: Visual comparison of both approaches

The notebook will show you which approach works better for your specific data! 📊


