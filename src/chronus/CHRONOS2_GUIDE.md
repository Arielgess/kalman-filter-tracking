# Chronos-2 Iterative Forecasting Guide

## Overview

This guide explains how to use Chronos-2 for **1-step ahead iterative forecasting** in tracking/filtering applications where you:
1. Feed Chronos **noisy observations** (sensor measurements)
2. Chronos predicts the next noisy observation
3. The true noisy observation is revealed and fed back
4. Evaluate predictions against **clean ground truth**
5. Repeat for the next timestep

This workflow is ideal for comparing with Kalman Filters, as both receive noisy measurements but can be evaluated against clean truth.

## Quick Start

### Installation

```bash
pip install chronos-forecasting
```

### Key Concept: Noisy Observations vs Clean Ground Truth

In tracking applications, you typically have:
- **Noisy observations**: What sensors actually measure (GPS positions, radar returns, etc.)
- **Clean ground truth**: The true positions (often not available in real-time, but used for evaluation)

**Chronos workflow:**
```python
# Input: Noisy observations (what Chronos sees)
context_df = df_observed.iloc[:i]  # Contains noisy measurements

# Prediction: Next noisy observation
prediction = pipeline.predict_df(context_df, prediction_length=1)

# Evaluation: Against clean ground truth
error = clean_truth[i] - prediction
```

This is different from Kalman Filters, which:
- Take noisy observations as input
- Directly output estimated clean state
- But can also be evaluated against clean ground truth

### Basic Usage

```python
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the model
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Prepare your data: noisy observations and clean ground truth
# In real tracking: noisy_data comes from sensors, clean_data from simulation/labels
noisy_observations = your_noisy_measurements  # What sensors measure
clean_ground_truth = your_clean_truth         # True positions (for evaluation)

timestamps = pd.date_range(start='2025-01-01', periods=len(noisy_observations), freq='1S')

# DataFrame with NOISY observations (what Chronos sees)
# IMPORTANT: Must include 'item_id' column for Chronos-2
df_observed = pd.DataFrame({
    'item_id': ['trajectory_1'] * len(noisy_observations),  # Required!
    'timestamp': timestamps,
    'target': noisy_observations
})

# DataFrame with CLEAN ground truth (for error calculation)
df_clean = pd.DataFrame({
    'item_id': ['trajectory_1'] * len(clean_ground_truth),
    'timestamp': timestamps,
    'target': clean_ground_truth
})

# Iterative 1-step ahead forecasting
context_length = 50
predictions = []
observed_vals = []
clean_vals = []

for i in range(context_length, len(df_observed)):
    # Use all past NOISY observations up to current timestep
    context_df = df_observed.iloc[:i].copy()
    
    # Predict 1 step ahead (predicting next noisy observation)
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=1,
        quantile_levels=[0.5],
        id_column='item_id',  # Required: column name containing time series IDs
        timestamp_column='timestamp',
        target='target',
    )
    
    predicted_value = pred_df['0.5'].values[0]
    predictions.append(predicted_value)
    observed_vals.append(df_observed.iloc[i]['target'])  # Actual noisy value
    clean_vals.append(df_clean.iloc[i]['target'])        # Clean ground truth
    
    # The true NOISY observation at timestep i is automatically included 
    # in the next iteration's context_df

# Evaluate against clean ground truth
errors_vs_clean = np.array(clean_vals) - np.array(predictions)
mae = np.abs(errors_vs_clean).mean()
print(f"Mean Absolute Error vs Clean Truth: {mae:.4f}")
```

## Key Concepts

### 1. Context Window

The context is the historical data used to make predictions. You have two options:

#### Expanding Window (Recommended for accuracy)
```python
context_df = df.iloc[:i]  # Uses all data from start to current position
```

#### Rolling Window (Recommended for efficiency)
```python
window_size = 100
context_df = df.iloc[max(0, i-window_size):i]  # Uses only last 100 observations
```

### 2. Prediction Parameters

```python
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=1,        # Always 1 for 1-step ahead
    quantile_levels=[0.1, 0.5, 0.9],  # 0.5 = median (main prediction)
    id_column=None,              # Use None for single time series
    timestamp_column='timestamp', # Name of your timestamp column
    target='target',             # Name of your target column
)
```

**Quantile levels:**
- `0.5`: Median prediction (most commonly used)
- `0.1`, `0.9`: Confidence intervals (10th and 90th percentiles)

### 3. Data Format

Chronos-2 requires pandas DataFrame with:
- **item_id column**: Unique identifier for each time series (REQUIRED, even for single series)
- **timestamp column**: DateTime index or column
- **target column**: The values to predict
- **Optional**: Additional covariate columns

```python
df = pd.DataFrame({
    'item_id': ['my_trajectory'] * T,  # Required: unique ID for this time series
    'timestamp': pd.date_range(start='2025-01-01', periods=T, freq='0.04S'),
    'target': trajectory_data[:, 0]  # Single dimension
})

# Then use in predict_df:
pred_df = pipeline.predict_df(
    df,
    prediction_length=1,
    id_column='item_id',  # Must match column name
    timestamp_column='timestamp',
    target='target'
)
```

## Multi-Dimensional Trajectories

For tracking problems with multiple dimensions (X, Y, Z), forecast each dimension independently:

```python
def forecast_dimension(noisy_data, clean_data, dim_idx, context_length=50):
    """
    Forecast single dimension with noisy observations, evaluate vs clean truth.
    """
    dim_noisy = noisy_data[:, dim_idx]
    dim_clean = clean_data[:, dim_idx]
    timestamps = pd.date_range(start='2025-01-01', periods=len(dim_noisy), freq='0.04S')
    
    df_obs = pd.DataFrame({
        'item_id': [f'trajectory_dim{dim_idx}'] * len(dim_noisy),  # Required!
        'timestamp': timestamps,
        'target': dim_noisy  # NOISY observations
    })
    
    predictions = []
    clean_vals = []
    
    for i in range(context_length, len(df_obs)):
        context_df = df_obs.iloc[:i]  # Context = noisy observations
        pred_df = pipeline.predict_df(
            context_df, prediction_length=1, quantile_levels=[0.5],
            id_column='item_id', timestamp_column='timestamp', target='target'
        )
        predictions.append(pred_df['0.5'].values[0])
        clean_vals.append(dim_clean[i])  # Store clean truth for evaluation
    
    return np.array(predictions), np.array(clean_vals)

# Forecast all dimensions
x_preds, x_clean = forecast_dimension(noisy_trajectory, clean_trajectory, 0)
y_preds, y_clean = forecast_dimension(noisy_trajectory, clean_trajectory, 1)

# Calculate 2D position error vs clean ground truth
position_errors = np.sqrt((x_clean - x_preds)**2 + (y_clean - y_preds)**2)
mean_position_error = position_errors.mean()
print(f"Mean 2D position error: {mean_position_error:.4f}")
```

## Performance Considerations

### Speed Optimization

1. **Use rolling window** instead of expanding window for long sequences
2. **Use GPU** if available: `device_map="cuda"`
3. **Reduce context length** if speed is critical
4. **Batch processing**: Not directly supported, must loop

### Memory Management

- Expanding window grows memory usage over time
- Rolling window has constant memory usage
- For very long sequences (>10,000 timesteps), use rolling window

## Comparison: Chronos-2 vs Kalman Filters

| Aspect | Chronos-2 | Kalman Filter |
|--------|-----------|---------------|
| **Approach** | Data-driven (learns patterns) | Model-based (requires motion model) |
| **Setup** | Load pretrained model | Define state space model |
| **Assumptions** | None (learns from data) | Linear/Gaussian assumptions |
| **Flexibility** | Handles complex patterns | Limited to model assumptions |
| **Speed** | Slower (neural network) | Faster (closed-form updates) |
| **Interpretability** | Black box | Transparent (physics-based) |

### When to use Chronos-2:
- You have lots of data and want to learn patterns
- Trajectories don't follow simple motion models
- You want quick experimentation without model design

### When to use Kalman Filters:
- You have a good motion model (CV, CA, CT)
- Real-time performance is critical
- You need interpretable state estimates (position, velocity)
- Limited training data available

## Common Issues

### Issue: "df does not contain all expected columns. Missing columns: [None]"

**Cause**: The `item_id` column is missing from your DataFrame.

**Solution**: Always include an `item_id` column, even for single time series:

```python
df = pd.DataFrame({
    'item_id': ['my_series'] * len(data),  # Add this!
    'timestamp': timestamps,
    'target': data
})

# Then use:
pred_df = pipeline.predict_df(
    df,
    id_column='item_id',  # Not None!
    timestamp_column='timestamp',
    target='target'
)
```

### Issue: "Frequency not supported"
```python
# Use valid pandas frequency strings
freq = '0.04S'  # 0.04 seconds
freq = '100ms'  # 100 milliseconds
freq = 'D'      # Daily
```

### Issue: Slow inference
```python
# Use smaller context window
context_length = 50  # Instead of 200+

# Use rolling window
window_size = 100
context_df = df.iloc[max(0, i-window_size):i]

# Use GPU if available
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
```

### Issue: Poor predictions
```python
# Try larger context window
context_length = 200

# Use more quantiles for uncertainty
quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Ensure your time series is properly scaled
# Chronos-2 works best with normalized data
```

## Resources

- **GitHub Repository**: https://github.com/amazon-science/chronos-forecasting
- **Quickstart Notebook**: https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
- **Paper**: Check the repository for the latest research paper

## Example: Complete Workflow

See the `run.ipynb` notebook in this directory for a complete working example that:
1. Loads trajectory data from your project
2. Performs iterative 1-step ahead forecasting
3. Evaluates prediction accuracy
4. Visualizes results
5. Compares expanding vs rolling window approaches
6. Handles multi-dimensional trajectories

