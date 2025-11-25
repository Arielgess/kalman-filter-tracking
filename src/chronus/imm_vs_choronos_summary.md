# Pretrained Chronos-2 vs. Oracle IMM Filter

## Experimental Results

### Experiment 1: Generated Composite Trajectories

The first experiment evaluated both methods on a dataset of 300 synthetic trajectories (15 classes, 20 trajectories per class). Each trajectory consisted of four sequential motion segments: constant acceleration (CA) for 40 timesteps, followed by moderate coordinated turn (CT) for 50 timesteps, constant velocity (CV) for 60 timesteps, and aggressive coordinated turn (CT) for 50 timesteps. So each trajectory was at a total length of 200 timesteps. The dataset systematically varied process noise levels (low, moderate, high) for acceleration, velocity changes, and turn rates. Moreover, I used three observation noise std levels (0.1, 0.5, 0.8). The class configuration resulted in configurations such as all-low-noise, all-high-noise, moderate-noise, and mixed noise scenarios. Both methods were evaluated using k=1 and k=5 step-ahead prediction with an expanding-window rolling forecast approach. The MSE values reported are normalized by dividing by the k-step ahead variance of each trajectory. Results are summarized in Tables 1 and 2. It's also worth saying that the IMM beat Chronos-2 in every single trajectory.

**Table 1: Experiment 1 Results (k=1)**

| Class | IMM MSE | Chronos-2 MSE |
|-------|---------|---------------|
| IMM_AllHighNoise_obs_0.1 | 0.144130 | 0.949934 |
| IMM_AllHighNoise_obs_0.5 | 0.504142 | 0.929516 |
| IMM_AllHighNoise_obs_0.8 | 0.599748 | 0.822718 |
| IMM_AllLowNoise_obs_0.1 | 0.123416 | 0.919354 |
| IMM_AllLowNoise_obs_0.5 | 0.515649 | 0.884524 |
| IMM_AllLowNoise_obs_0.8 | 0.571533 | 0.778215 |
| IMM_AllModerateNoise_obs_0.1 | 0.136659 | 0.843524 |
| IMM_AllModerateNoise_obs_0.5 | 0.555260 | 0.847936 |
| IMM_AllModerateNoise_obs_0.8 | 0.584699 | 0.826206 |
| IMM_HighCV_LowCA_obs_0.1 | 0.104129 | 0.904618 |
| IMM_HighCV_LowCA_obs_0.5 | 0.542708 | 0.890197 |
| IMM_HighCV_LowCA_obs_0.8 | 0.603385 | 0.837879 |
| IMM_LowCV_HighCA_obs_0.1 | 0.157192 | 0.875730 |
| IMM_LowCV_HighCA_obs_0.5 | 0.511092 | 0.853906 |
| IMM_LowCV_HighCA_obs_0.8 | 0.563194 | 0.847162 |

**Table 2: Experiment 1 Results (k=5)**

| Class | IMM Normalized MSE | Chronos-2 Normalized MSE |
|-------|---------|---------------|
| IMM_AllHighNoise_obs_0.1 | 0.050074 | 0.162283 |
| IMM_AllHighNoise_obs_0.5 | 0.155330 | 0.316173 |
| IMM_AllHighNoise_obs_0.8 | 0.321778 | 0.516975 |
| IMM_AllLowNoise_obs_0.1 | 0.043591 | 0.152299 |
| IMM_AllLowNoise_obs_0.5 | 0.160614 | 0.355647 |
| IMM_AllLowNoise_obs_0.8 | 0.255113 | 0.494020 |
| IMM_AllModerateNoise_obs_0.1 | 0.034364 | 0.134503 |
| IMM_AllModerateNoise_obs_0.5 | 0.145946 | 0.341323 |
| IMM_AllModerateNoise_obs_0.8 | 0.282295 | 0.511031 |
| IMM_HighCV_LowCA_obs_0.1 | 0.049717 | 0.147684 |
| IMM_HighCV_LowCA_obs_0.5 | 0.166598 | 0.332192 |
| IMM_HighCV_LowCA_obs_0.8 | 0.289134 | 0.484121 |
| IMM_LowCV_HighCA_obs_0.1 | 0.043270 | 0.148238 |
| IMM_LowCV_HighCA_obs_0.5 | 0.179648 | 0.373186 |
| IMM_LowCV_HighCA_obs_0.8 | 0.309804 | 0.538152 |


### Experiment 2: Classes with different motions

The second experiment evaluated both methods on more realistic motion patterns, comprising five distinct motion classes: highly maneuvering targets with sharp turns, moderate maneuvering targets with gentle turns, agile targets with erratic turns and bursts, gliding targets with gentle turns, and extreme maneuvering targets with rapid mode switching. This experiment included 500 trajectories (100 per class) with possibly variable-length segments and motion patterns. The MSE values reported are normalized by dividing by the k-step ahead variance of each trajectory. Results are summarized in Tables 3 and 4. Similar to the first experiment, the IMM beat Chronos-2 in every signle trajectory.

**Table 3: Experiment 2 Results (k=1)**

| Class | IMM MSE | Chronos-2 MSE |
|-------|---------|---------------|
| AgileTargetErraticTurnsAndBursts | 0.398778 | 0.865323 |
| ExtremeManeuveringTargetRapidModeSwitching | 0.473739 | 0.858247 |
| GlidingTargetGentleTurns | 0.404903 | 0.646190 |
| HighlyManeuveringTargetSharpTurns | 0.488470 | 0.852965 |
| ModerateManeuveringTargetWithTurns | 0.398322 | 0.648714 |

**Table 4: Experiment 2 Results (k=5)**

| Class | IMM MSE | Chronos-2 MSE |
|-------|---------|---------------|
| AgileTargetErraticTurnsAndBursts | 0.091365 | 0.143934 |
| ExtremeManeuveringTargetRapidModeSwitching | 0.143193 | 0.212449 |
| GlidingTargetGentleTurns | 0.097651 | 0.156544 |
| HighlyManeuveringTargetSharpTurns | 0.165226 | 0.234219 |
| ModerateManeuveringTargetWithTurns | 0.087863 | 0.139255 |


