# Chronos-2 Workflow for Noisy Observations & Clean Evaluation

## Your Question

> "I have the observed data, and the real (clean) data. I want the chronus to receive the observed data, but when calculating the error I want it to compute it against the real data."

**Answer: YES! This is exactly what Chronos-2 can and should do!**

## The Workflow

```
                    ITERATIVE 1-STEP AHEAD FORECASTING
                    
┌─────────────────────────────────────────────────────────────────┐
│                        At Timestep i                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CONTEXT (Input to Chronos)                                  │
│     ┌──────────────────────────────────────────┐               │
│     │ All past NOISY observations              │               │
│     │ [obs₁, obs₂, obs₃, ..., obs_{i-1}]      │               │
│     │ (What sensors have measured so far)      │               │
│     └──────────────────────────────────────────┘               │
│                          │                                       │
│                          ▼                                       │
│                                                                  │
│  2. CHRONOS PREDICTION                                          │
│     ┌──────────────────────────────────────────┐               │
│     │ Chronos-2 Model                          │               │
│     │ Predicts: next NOISY observation         │               │
│     │ pred_i (what sensor will measure next)   │               │
│     └──────────────────────────────────────────┘               │
│                          │                                       │
│                          ▼                                       │
│                                                                  │
│  3. OBSERVE TRUE NOISY VALUE                                    │
│     ┌──────────────────────────────────────────┐               │
│     │ Actual noisy observation obs_i           │               │
│     │ (Real sensor measurement)                │               │
│     └──────────────────────────────────────────┘               │
│                          │                                       │
│                          ▼                                       │
│                                                                  │
│  4. EVALUATE vs CLEAN GROUND TRUTH                              │
│     ┌──────────────────────────────────────────┐               │
│     │ error = clean_truth_i - pred_i           │               │
│     │                                           │               │
│     │ This is your desired evaluation!         │               │
│     └──────────────────────────────────────────┘               │
│                          │                                       │
│                          ▼                                       │
│                                                                  │
│  5. FEED BACK for Next Iteration                                │
│     ┌──────────────────────────────────────────┐               │
│     │ Add obs_i to context                     │               │
│     │ Next iteration will use:                 │               │
│     │ [obs₁, obs₂, ..., obs_{i-1}, obs_i]     │               │
│     └──────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Points

### ✅ What Chronos Sees (Input)
- **Noisy observations** from sensors
- Historical sequence of measurements with noise
- Realistic sensor data

### ✅ What Chronos Predicts (Output)
- **Next noisy observation**
- Prediction includes expected noise pattern
- Not trying to "denoise" the signal

### ✅ What You Use for Evaluation
- **Clean ground truth**
- True positions without noise
- Fair comparison metric

### ✅ What Gets Fed Back
- **True noisy observation** (not the prediction)
- Ensures model sees real data progression
- Prevents error accumulation

## Code Implementation

```python
# You have both:
noisy_observations = data['X'][0]      # What Chronos sees
clean_ground_truth = data['Y'][0]      # What you evaluate against

# Prepare DataFrames
df_observed = pd.DataFrame({
    'timestamp': timestamps,
    'target': noisy_observations  # INPUT to Chronos
})

df_clean = pd.DataFrame({
    'timestamp': timestamps,
    'target': clean_ground_truth  # For EVALUATION
})

# Iterative forecasting
for i in range(context_length, len(df_observed)):
    # 1. Context: all past noisy observations
    context_df = df_observed.iloc[:i]
    
    # 2. Predict next noisy observation
    pred_df = pipeline.predict_df(context_df, prediction_length=1, ...)
    prediction = pred_df['0.5'].values[0]
    
    # 3. True noisy observation (automatically used in next iteration)
    true_noisy = df_observed.iloc[i]['target']
    
    # 4. Evaluate against CLEAN ground truth
    true_clean = df_clean.iloc[i]['target']
    error = true_clean - prediction
    
    # 5. Feed back: true_noisy is already in df_observed,
    #    so next iteration's context will include it
```

## Why This Makes Sense

### Real-World Analogy
1. **GPS sensor** gives you noisy position measurements
2. You want to **predict** where GPS will say you are next
3. But you want to **evaluate** how close that is to your true position
4. This is exactly your workflow!

### Comparison with Kalman Filter

| Aspect | Chronos-2 | Kalman Filter |
|--------|-----------|---------------|
| **Input** | Noisy observations | Noisy observations |
| **Process** | Neural network prediction | State estimation with motion model |
| **Output** | Predicted noisy observation | Estimated clean state |
| **Evaluation** | Compare to clean truth | Compare to clean truth |

Both approaches receive the same noisy input and can be fairly compared!

## Your Updated Notebook

The notebook `run.ipynb` now implements exactly this workflow:

1. ✅ Loads both noisy (observed) and clean (ground truth) data
2. ✅ Feeds Chronos the noisy observations
3. ✅ Chronos predicts next noisy observation
4. ✅ Evaluates predictions against clean ground truth
5. ✅ Shows both error metrics (vs clean and vs observed)
6. ✅ Handles multi-dimensional trajectories
7. ✅ Visualizes all three: clean, noisy, and predicted

## Result

You get:
- **MAE vs Clean Truth**: How accurate Chronos is at predicting true positions
- **RMSE vs Clean Truth**: Root mean squared error against true positions
- **Position Error**: Euclidean distance in 2D/3D space

This allows you to directly compare Chronos with your Kalman Filter implementations!


