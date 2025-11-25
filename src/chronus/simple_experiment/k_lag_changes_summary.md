# K-Lag Prediction Implementation Summary

## Changes Made to `imm_run.ipynb`

### Overview
Implemented k-lag (k-step ahead) prediction evaluation for both Chronos-2 and IMM filter, replacing the previous 1-step ahead prediction setup.

### Key Changes

#### 1. **Variable Renaming**
- `context_length_rolling` → `initial_context_length`
- This better reflects that the context window is **expanding** (not rolling)

#### 2. **K-Lag Configuration** (Cell 6)
```python
k = 5  # k-step ahead prediction
initial_context_length = 3  # Minimum context to start predictions
```

#### 3. **Chronos Prediction Loop** (Cell 6)
- **Before**: Predicted 1 step ahead for each timestep
- **After**: 
  - Predicts `k` steps ahead using `prediction_length=k`
  - Loop runs from `initial_context_length` to `T_rolling - k`
  - Number of predictions: `T_rolling - initial_context_length - k`

**Key Logic:**
- At timestep `t`, use context `[0:t]` to predict `t+k`
- Context expands from `initial_context_length` to `T - k`

#### 4. **Chronos Prediction Parsing** (Cell 8)
- **Before**: Extracted single prediction from each forecast
- **After**:
  - Extracts only the **k'th prediction** (last one) from each k-step forecast
  - Uses timestamp sorting to identify the k'th step
  - Parses x and y predictions separately from the dataframe

#### 5. **IMM K-Lag Evaluation** (Cell 9)
- **Before**: Manual IMM prediction loop with 1-step ahead
- **After**: 
  - Uses `IMMKlagEvaluator` class from `imm_k_lag.py`
  - Calls `run_k_lag(measurements, k, clean_signal=None, normalize_mse=False)`
  - Properly implements k-step ahead prediction using IMM mixing algorithm
  - Separate IMM instance tracks model probabilities for analysis

**Key Implementation:**
```python
from src.imm_models.imm_k_lag import IMMKlagEvaluator

imm_k_lag_eval = IMMKlagEvaluator(imm)
mse_imm, preds_imm = imm_k_lag_eval.run_k_lag(
    measurements=measurements,
    k=k,
    clean_signal=None,
    normalize_mse=False
)
```

#### 6. **MSE Calculation** (Cell 9)
- **Before**: Compared prediction at time `t` with measurement at time `t`
- **After**: 
  - **Chronos**: Prediction at index `i` compared with measurement at `initial_context_length + i + k`
  - **IMM**: MSE calculated internally by `run_k_lag` method
  - Both compare k-step ahead predictions with actual measurements at `t+k`

**Key Logic for Chronos:**
```python
# prediction[i] predicts time initial_context_length + i + k
noisy_measurements = imm_data["X"][traj_idx][initial_context_length + k:]
mse_chronos = ((noisy_measurements - chronos_preds)**2).mean()
```

#### 7. **Results Data Structure** (Cell 9)
Added k-lag metadata to results:
```python
'imm_predictions': kf_predictions_per_trajectory[traj_idx],  # Shape: (T-k, 2)
'chronos_predictions': chronos_predictions_per_trajectory[traj_idx],  # Shape: (T-initial_context-k, 2)
'k': k,  # Store k-lag value
'initial_context_length': initial_context_length  # Store initial context
```

### Prediction Timeline

For a trajectory of length T=197, k=5, initial_context=3:

**Chronos:**
- Timestep 3: Use context [0:3] → predict timestep 8
- Timestep 4: Use context [0:4] → predict timestep 9
- ...
- Timestep 192: Use context [0:192] → predict timestep 197
- Total predictions: 189 (from timestep 8 to 197)

**IMM:**
- Similar structure, using IMMKlagEvaluator
- Predictions: (T-k, 2) = (192, 2)

### Verification Checklist

✅ Variable renamed from `context_length_rolling` to `initial_context_length`
✅ Chronos predicts k steps ahead (prediction_length=k)
✅ Only k'th prediction extracted from Chronos output
✅ Prediction loop runs from `initial_context_length` to `T - k`
✅ IMM uses IMMKlagEvaluator.run_k_lag() method
✅ MSE calculations compare prediction[t] with measurement[t+k]
✅ Both models evaluated on same k-lag basis
✅ Results structure includes k and initial_context_length metadata

### Notes

1. **Fair Comparison**: Both Chronos and IMM now predict exactly k steps ahead, making the comparison fair for multi-step forecasting scenarios.

2. **Expanding vs Rolling Window**: The context window expands (uses all history), not rolls (fixed window). This is more typical for time series forecasting.

3. **Model Probabilities**: IMM model probabilities are still tracked using a separate IMM instance since the k-lag evaluator doesn't expose them during prediction.

4. **MSE Against Noisy Measurements**: Both models are evaluated against actual noisy measurements (what sensors would observe), not clean ground truth.














