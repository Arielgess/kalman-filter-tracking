# Kalman Filter Tracking Framework

A comprehensive Python framework for Kalman Filter-based target tracking and evaluation, designed to serve as a robust baseline for deep learning research and general tracking applications.

## 🚀 Features

### Core Tracking Capabilities
- **Multiple Motion Models**: Support for Constant Velocity (CV), Constant Acceleration (CA), Coordinated Turn (CT), and Singer acceleration models
- **Extended Kalman Filters**: Full EKF implementation for non-linear motion models
- **Interacting Multiple Model (IMM)**: Advanced multi-model tracking with automatic model switching
- **2D and 3D Tracking**: Comprehensive support for both 2D and 3D tracking scenarios
- **K-lag Prediction**: Multi-step ahead prediction capabilities for performance evaluation

### Trajectory Generation
- **Synthetic Data Generation**: Generate realistic trajectories for testing and evaluation
- **Multiple Motion Patterns**: CV, CA, CT, and Singer model trajectory generation
- **Composite Trajectories**: Combine multiple motion segments for complex scenarios
- **Dubins Airplane Paths**: Basic Dubins airplane implementation (converted from C++, may need additional integration work)
- **Configurable Noise**: Realistic measurement and process noise modeling

### Evaluation and Analysis
- **Performance Metrics**: 
  - MSE (Mean Squared Error) with optional normalization
  - NMSE (Normalized Mean Squared Error) for cross-dataset comparison
  - K-lag prediction evaluation for multi-step ahead performance
  - Innovation analysis for model consistency checking
- **Trajectory Classification**: Automatic model classification based on tracking performance
- **Confusion Matrix Analysis**: Cross-model performance evaluation using MSE-based classification
- **Statistical Validation**: Framework for innovation-based model validation (documented in educational notebooks)
- **Visualization Tools**: Rich plotting capabilities for trajectories, predictions, and performance analysis

### Advanced Features
- **Parameter Estimation**: EM-based parameter estimation for linear Kalman filters only (Q and R matrices)
- **State Space Models**: Comprehensive implementation of discrete state space models
- **Modular Architecture**: Clean, extensible design for easy customization and extension

## 📁 Project Structure

```
kalman-filter-tracking/
├── src/
│   ├── filters/                    # Kalman Filter implementations
│   │   ├── base_kalman_filter.py   # Base KF class
│   │   ├── cv_kalman_filter.py     # Constant Velocity KF
│   │   ├── ca_kalman_filter.py     # Constant Acceleration KF
│   │   ├── ct_kalman_filter.py     # Coordinated Turn KF
│   │   ├── singer_model_kalman_filter.py  # Singer model KF
│   │   └── base_extended_kalman_filter.py # Extended KF base
│   ├── imm_models/                 # Interacting Multiple Model
│   │   ├── imm_k_lag.py           # IMM with k-lag prediction
│   │   └── models_for_imm.py      # IMM model definitions
│   ├── motion_models/              # Trajectory generation
│   │   ├── trajectory_generation/  # Synthetic trajectory generation
│   │   └── dubin_aiplane/         # Dubins airplane paths
│   ├── evaluation_tools/           # Performance evaluation
│   │   └── trajectory_classifier.py # Model classification
│   └── visual/                     # Visualization tools
│       └── visual_tools.py        # Plotting and visualization
├── examples/                       # Usage examples
│   └── 1. Basic Tracking.ipynb    # Basic tracking examples
├── summary/                        # Educational notebooks
│   ├── 1. State Space Models.ipynb
│   ├── 2. Kalman Filter.ipynb
│   ├── 3. Extended Kalman Filter.ipynb
│   ├── 4. IMM.ipynb
│   └── 5. Parameters Estimation With EM.ipynb
└── requirements.txt               # Dependencies
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kalman-filter-tracking.git
cd kalman-filter-tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Tracking Example

```python
from src.motion_models.trajectory_generation.route_generation import *
from src.filters.cv_kalman_filter import CVKalmanFilter, InitializationData
import numpy as np

# Generate a test trajectory
initial_state = TrajectoryState(position=np.array([0, 0]), velocity=np.array([14, 5]))
trajectories = generate_cv_trajectory(
    T=80, dt=0.04, initial_state=initial_state, 
    measurement_noise_std=np.array([0.4, 0.4]),
    vel_change_std=np.array([0.5, 0.5])
)

noisy_traj, clean_traj = trajectories[0][0], trajectories[0][1]

# Initialize and run Kalman Filter
init_data = InitializationData(
    observation_noise_std=np.array([0.4, 0.4]),
    process_noise_std=np.array([0.5, 0.5])
)

kf = CVKalmanFilter(2, 4, 2, 0.04, init_data)
kf.initialize()

# Evaluate performance
mse, predictions = kf.evaluate_on_trajectory(
    noisy_traj, clean_traj, k=1, return_predictions=True
)

print(f"Tracking MSE: {mse}")
```

### Multi-Model Tracking with IMM

```python
from src.imm_models.models_for_imm import create_imm_estimator
from src.imm_models.imm_k_lag import IMMKlagEvaluator

# Create IMM estimator with multiple models
imm = create_imm_estimator(models=['CV', 'CT'], dim=2, dt=0.04)

# Run k-lag evaluation
evaluator = IMMKlagEvaluator(imm)
mse, predictions = evaluator.run_k_lag(measurements, k=5)
```

## 📊 Motion Models

### Constant Velocity (CV)
- **Use Case**: Linear motion with small random accelerations
- **State**: Position and velocity
- **Best For**: Straight-line tracking with minor deviations

### Constant Acceleration (CA)
- **Use Case**: Motion with constant acceleration
- **State**: Position, velocity, and acceleration
- **Best For**: Accelerating/decelerating targets

### Coordinated Turn (CT)
- **Use Case**: Circular motion with constant turn rate
- **State**: Position, velocity, and turn rate
- **Best For**: Aircraft, vehicles making coordinated turns

### Singer Model
- **Use Case**: Maneuvering targets with correlated acceleration
- **State**: Position, velocity, and acceleration with time correlation
- **Best For**: Highly maneuvering targets

## 🎯 Evaluation Metrics

### Primary Metrics
- **MSE (Mean Squared Error)**: Standard tracking accuracy metric with optional normalization
- **NMSE (Normalized MSE)**: Cross-dataset comparable error metric normalized by measurement variance
- **K-lag Prediction**: Multi-step ahead prediction performance evaluation

### Statistical Validation
- **Innovation Analysis**: Basic residual computation (documented in educational notebooks, not fully implemented)
- **NIS (Normalized Innovation Squared)**: Chi-squared test for model validation (documented in educational notebooks only)
- **Model Consistency**: Framework for detecting model-data mismatches (documented, not implemented)

### Classification and Comparison
- **Model Classification**: Automatic identification of motion patterns using MSE-based classification
- **Confusion Matrix**: Cross-model performance analysis and classification accuracy
- **Trajectory Classification**: Performance-based trajectory categorization across different motion models

## 📈 Visualization

The framework includes comprehensive visualization tools:

- **Trajectory Plots**: 2D/3D trajectory visualization with ground truth and predictions
- **Performance Analysis**: MSE plots and error analysis
- **Model Probabilities**: IMM model probability evolution over time
- **Prediction Comparison**: Side-by-side comparison of different prediction horizons

*[Placeholder for tracking visualization images]*

*[Placeholder for performance comparison plots]*

*[Placeholder for IMM model probability plots]*

## 📚 Educational Resources

The `summary/` directory contains comprehensive educational notebooks:

1. **State Space Models**: Introduction to discrete state space models
2. **Kalman Filter**: Detailed KF theory and implementation
3. **Extended Kalman Filter**: Non-linear filtering techniques
4. **IMM**: Interacting Multiple Model algorithm
5. **Parameter Estimation**: EM-based parameter tuning (linear filters only)

## ⚠️ Current Limitations

- **EM Parameter Estimation**: Only available for linear Kalman filters (CV, CA models), not for EKF or non-linear models
- **Dubins Airplane**: Basic implementation converted from C++, may require additional integration work for full functionality
- **Statistical Tests**: NIS and other statistical validation methods are documented in educational notebooks but not implemented in the codebase
- **Extended Kalman Filter**: EKF base class exists but may need additional implementation for specific motion models

## 🔬 Research Applications

This framework is particularly well-suited for:

- **Deep Learning Research**: Baseline for neural network tracking algorithms
- **Sensor Fusion**: Multi-sensor tracking applications
- **UAV/Autonomous Vehicle Tracking**: Basic Dubins airplane path support (experimental)
- **Target Tracking**: General object tracking in 2D/3D space
- **Performance Benchmarking**: Standardized evaluation of tracking algorithms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Dubins airplane implementation is based on the [ntnu-arl/DubinsAirplane](https://github.com/ntnu-arl/DubinsAirplane) repository
- Built on top of the excellent [filterpy](https://github.com/rlabbe/filterpy) library
- Inspired by classical tracking literature and modern deep learning research

## 📞 Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

*This framework provides a solid foundation for tracking research and applications, combining classical filtering theory with modern software engineering practices.*