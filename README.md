# Kalman Filter Tracking Framework

A Python framework for trajectory tracking using various Kalman filter implementations and motion models. The framework provides implementations of standard Kalman filters, extended Kalman filters, and Interacting Multiple Model (IMM) estimators for tracking applications in 2D and 3D space.

## Features

### Kalman Filter Implementations
- **Constant Velocity (CV)**: Basic linear Kalman filter for constant velocity motion
- **Constant Acceleration (CA)**: Linear Kalman filter for constant acceleration motion  
- **Coordinated Turn (CT)**: Extended Kalman filter for maneuvering targets with constant turn rate
- **Singer Model**: Extended Kalman filter for targets with correlated acceleration

### Advanced Tracking Methods
- **Interacting Multiple Model (IMM)**: Combines multiple motion models with probabilistic switching
- **K-lag Evaluation**: Multi-step ahead prediction capabilities for performance assessment
- **Parameter Estimation**: Expectation-Maximization (EM) algorithm for noise parameter estimation

### Motion Models and Trajectory Generation
- **Constant Velocity (CV)**: Nearly constant velocity with small random accelerations
- **Constant Acceleration (CA)**: Constant acceleration motion with noise
- **Coordinated Turn (CT)**: Circular motion with constant turn rate
- **Singer Model**: Correlated acceleration model with exponential decay
- **Composite Trajectories**: Combination of multiple motion segments, for example a composite trajectory can be composed from CA, CV and then CT sub-trajectories.
- **Dubins Airplane**: 3D Dubins path generation for aircraft trajectories

### Evaluation and Visualization Tools
- **Trajectory Classification**: Classification of trajectories against different KFs. One of the KFs must be the true fit. Outputs a Confusion Matrix
- **Performance Metrics**: MSE calculation. Also possible to generate a graph of the MSE with sliding window analysis
- **Visualization**: 2D/3D trajectory plotting with filtering results
- **K-lag Analysis**: k-lag prediction. Supported both MSE calculation and the Visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The framework is organized into several modules:

- `src/filters/`: Kalman filter implementations
- `src/imm_models/`: KF Models supported by the IMM and related utilities
- `src/motion_models/`: Trajectory generation functions
- `src/evaluation_tools/`: Performance evaluation utilities
- `src/visual/`: Plotting and visualization tools

See the `examples/` directory for detailed usage examples:

1. **Basic Tracking**: Simple trajectory tracking with different filter types
2. **Classification Using Confusion Matrix**: Model comparison and classification
3. **IMM**: Interacting Multiple Model implementation
4. **Estimating Noise With EM**: Parameter estimation using EM algorithm
