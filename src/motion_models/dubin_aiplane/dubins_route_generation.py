"""
Dubins Airplane Route Generation Compatibility Layer

This module provides a compatibility layer to match the route_generation.py output schema
while using the converted Dubins airplane trajectory generator.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
from KalmanFilter.dubins_trajectory.DubinsAirplaneFunctions import DubinsAirplanePath, ExtractDubinsAirplanePath, MinTurnRadius_DubinsAirplane


@dataclass
class TrajectoryState:
    """State information for trajectory generation and continuation."""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: Optional[np.ndarray] = None  # [ax, ay, az] for CA/Singer models
    omega: Optional[float] = None  # Turn rate for CT model
    tau: Optional[np.ndarray] = None  # Time constant(s) for Singer model
    heading: Optional[float] = None  # Heading angle for Dubins
    airspeed: Optional[float] = None  # Airspeed for Dubins
    
    def __post_init__(self):
        """Ensure all arrays have consistent dimensionality and correct dtypes."""
        # Convert arrays to float to avoid type casting issues
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float)
            if len(self.acceleration) != len(self.position):
                raise ValueError("Acceleration dimension must match position dimension")
        
        if len(self.velocity) != len(self.position):
            raise ValueError("Velocity dimension must match position dimension")
            
        if self.tau is not None:
            self.tau = np.asarray(self.tau, dtype=float)
            if len(self.tau) != len(self.position):
                raise ValueError("Tau dimension must match position dimension")


def generate_dubins_trajectory(
    dim: int = 3,  # Dubins airplane is always 3D
    initial_state: Optional[TrajectoryState] = None,
    measurement_noise_std: Union[float, np.ndarray] = 0.0,
    number_of_trajectories: int = 1,
    seed: Optional[int] = None,
    # Dubins-specific parameters
    end_position: Optional[np.ndarray] = None,
    end_heading: Optional[float] = None,
    airspeed: float = 15.0,
    bank_max: float = np.pi/4,
    gamma_max: float = np.pi/6,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Generate Dubins airplane trajectories compatible with route_generation.py schema.
    
    Args:
        T: Number of time steps (ignored for Dubins - uses path length)
        dt: Time step duration (ignored for Dubins - uses internal step size)
        dim: Dimensionality (must be 3 for Dubins airplane)
        initial_state: Initial state (position, velocity, heading, airspeed)
        measurement_noise_std: Per-axis std of measurement noise
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        end_position: Final position [x, y, z]
        end_heading: Final heading angle in radians
        airspeed: Airspeed magnitude
        bank_max: Maximum bank angle in radians
        gamma_max: Maximum climb/descent angle in radians
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    if dim != 3:
        raise ValueError("Dubins airplane trajectories are always 3D")

    rng = np.random.default_rng(seed)
    
    # Convert scalars to arrays
    if np.isscalar(measurement_noise_std):
        measurement_noise_std = np.full(3, float(measurement_noise_std))
    else:
        measurement_noise_std = np.asarray(measurement_noise_std, dtype=float)
    
    trajectories = []
    
    for _ in range(number_of_trajectories):
        # Prepare start and end configurations
        if initial_state is None:
            start_position = np.array([0.0, 0.0, -100.0])
            start_heading = 0.0
        else:
            start_position = initial_state.position
            # Extract heading from velocity direction
            start_heading = np.arctan2(initial_state.velocity[1], initial_state.velocity[0])
        
        if end_position is None:
            end_position = np.array([100.0, 100.0, -125.0])
        if end_heading is None:
            end_heading = np.pi/2
        
        # Create configuration arrays [x, y, z, heading, airspeed]
        start_config = np.array([
            start_position[0],
            start_position[1], 
            start_position[2],
            start_heading,
            airspeed
        ])
        
        end_config = np.array([
            end_position[0],
            end_position[1],
            end_position[2], 
            end_heading,
            airspeed
        ])
        
        # Compute minimum turning radius
        R_min = MinTurnRadius_DubinsAirplane(airspeed, bank_max)
        
        # Check if start and end nodes are too close
        horizontal_distance = np.linalg.norm(end_position[0:2] - start_position[0:2])
        if horizontal_distance < 6 * R_min:
            raise ValueError(
                f"Start and end poses are too close (distance: {horizontal_distance:.2f} < 6*R_min: {6*R_min:.2f}). "
                "Path of type RLR, LRL may be optimal. Consider increasing distance or using different parameters."
            )
        
        # Compute Dubins airplane path
        dubins_solution = DubinsAirplanePath(start_config, end_config, R_min, gamma_max)
        
        # Extract trajectory points
        trajectory_points = ExtractDubinsAirplanePath(dubins_solution)
        
        # Convert to clean positions array (T x 3)
        clean_positions = trajectory_points.T  # Shape: (T, 3)
        
        # Add measurement noise
        if np.any(measurement_noise_std > 0):
            measurement_noise = rng.normal(0, measurement_noise_std, size=clean_positions.shape)
            noisy_positions = clean_positions + measurement_noise
        else:
            noisy_positions = clean_positions.copy()
        
        # Create final state from last trajectory point
        final_position = clean_positions[-1]
        final_velocity = np.array([
            airspeed * np.cos(end_heading),
            airspeed * np.sin(end_heading),
            0.0  # Assume level flight at end
        ])
        
        final_state = TrajectoryState(
            position=final_position,
            velocity=final_velocity,
            heading=end_heading,
            airspeed=airspeed
        )
        
        trajectories.append((noisy_positions, clean_positions, final_state))
    
    return trajectories


def generate_dubins_trajectory_simple(
    start_position: np.ndarray,
    start_heading: float,
    end_position: np.ndarray, 
    end_heading: float,
    airspeed: float = 15.0,
    bank_max: float = np.pi/4,
    gamma_max: float = np.pi/6,
    measurement_noise_std: Union[float, np.ndarray] = 0.0,
    number_of_trajectories: int = 1,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Simplified interface for generating Dubins airplane trajectories.
    
    Args:
        start_position: Initial position [x, y, z]
        start_heading: Initial heading angle in radians
        end_position: Final position [x, y, z]
        end_heading: Final heading angle in radians
        airspeed: Airspeed magnitude
        bank_max: Maximum bank angle in radians
        gamma_max: Maximum climb/descent angle in radians
        measurement_noise_std: Per-axis std of measurement noise
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    # Create initial state
    start_velocity = np.array([
        airspeed * np.cos(start_heading),
        airspeed * np.sin(start_heading),
        0.0
    ])
    
    initial_state = TrajectoryState(
        position=np.asarray(start_position, dtype=float),
        velocity=start_velocity,
        heading=float(start_heading),
        airspeed=float(airspeed)
    )
    
    return generate_dubins_trajectory(
        T=0,  # Ignored for Dubins
        dt=0.01,  # Ignored for Dubins
        dim=3,
        initial_state=initial_state,
        measurement_noise_std=measurement_noise_std,
        number_of_trajectories=number_of_trajectories,
        seed=seed,
        end_position=end_position,
        end_heading=end_heading,
        airspeed=airspeed,
        bank_max=bank_max,
        gamma_max=gamma_max
    )
