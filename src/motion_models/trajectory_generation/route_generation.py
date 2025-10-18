import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from dataclasses import dataclass
from typing import Optional, List, Tuple
from scipy.linalg import expm


@dataclass
class TrajectoryState:
    """State information for trajectory generation and continuation."""
    position: np.ndarray  # [x, y] or [x, y, z]
    velocity: np.ndarray  # [vx, vy] or [vx, vy, vz]
    acceleration: Optional[np.ndarray] = None  # [ax, ay] or [ax, ay, az] for CA/Singer models
    omega: Optional[float] = None  # Turn rate for CT model
    tau: Optional[np.ndarray] = None  # Time constant(s) for Singer model
    
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

def generate_cv_trajectory(
    T: int,
    dt: float,
    initial_state: TrajectoryState,
    vel_change_std: float | np.ndarray = 0.0,  # per-axis std of acceleration noise
    measurement_noise_std: float | np.ndarray = 0.0,  # per-axis std of measurement noise
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Generate nearly-constant-velocity trajectories with small random accelerations.
    
    Args:
        T: Number of time steps
        dt: Time step duration
        initial_state: Initial state (required)
        vel_change_std: Per-axis std of acceleration noise
        measurement_noise_std: Per-axis std of measurement noise
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    dim = len(initial_state.position)
    assert dim in (2, 3), "dim must be 2 or 3"
    rng = np.random.default_rng(seed)

    # Convert scalars to arrays
    if np.isscalar(vel_change_std):
        vel_change_std = np.full(dim, float(vel_change_std))
    else:
        vel_change_std = np.asarray(vel_change_std, dtype=float)
        
    if np.isscalar(measurement_noise_std):
        measurement_noise_std = np.full(dim, float(measurement_noise_std))
    else:
        measurement_noise_std = np.asarray(measurement_noise_std, dtype=float)

    trajectories = []
    for _ in range(number_of_trajectories):
        # Initialize state
        state = TrajectoryState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy()
        )

        clean = np.empty((T, dim), dtype=float)
        clean[0] = state.position

        # Generate trajectory
        for t in range(1, T):
            # Store previous velocity before updating
            prev_velocity = state.velocity.copy()
            
            # Add random velocity change (not persistent acceleration)
            velocity_change = rng.normal(0.0, vel_change_std, size=dim)
            state.velocity += velocity_change
            
            # Update position using previous velocity
            state.position += prev_velocity * dt
            clean[t] = state.position

        # Add measurement noise
        meas_noise = rng.normal(0.0, measurement_noise_std, size=(T, dim))
        noisy = clean + meas_noise

        trajectories.append((noisy, clean, state))

    return trajectories


def generate_ca_trajectory(
    T: int,
    dt: float,
    initial_state: TrajectoryState,
    measurement_noise_std: float | np.ndarray = 0.0,
    accel_noise_std: float | np.ndarray = 0.0,
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Generate constant-acceleration trajectories with optional white acceleration noise.
    
    Args:
        T: Number of time steps
        dt: Time step duration
        initial_state: Initial state (required)
        measurement_noise_std: Std of measurement noise
        accel_noise_std: Per-axis std of acceleration noise
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    dim = len(initial_state.position)
    assert dim in (2, 3), "dim must be 2 or 3"
    rng = np.random.default_rng(seed)

    # Convert scalars to arrays
    if np.isscalar(measurement_noise_std):
        measurement_noise_std = np.full(dim, float(measurement_noise_std))
    else:
        measurement_noise_std = np.asarray(measurement_noise_std, dtype=float)
        
    if np.isscalar(accel_noise_std):
        accel_noise_std = np.full(dim, float(accel_noise_std))
    else:
        accel_noise_std = np.asarray(accel_noise_std, dtype=float)

    trajectories = []
    for _ in range(number_of_trajectories):
        # Initialize state
        state = TrajectoryState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
            acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else np.zeros(dim, dtype=float)
        )

        clean = np.zeros((T, dim), dtype=float)


        # Generate trajectory
        for t in range(T):
            # Store current position
            clean[t] = state.position

            # Propagate dynamics
            state.position += state.velocity * dt + 0.5 * state.acceleration * dt**2
            state.velocity += state.acceleration * dt

            # Add acceleration noise
            state.acceleration += rng.normal(0.0, accel_noise_std, size=dim)

        # Add measurement noise
        meas_noise = rng.normal(0.0, measurement_noise_std, size=(T, dim))
        noisy = clean + meas_noise

        trajectories.append((noisy, clean, state))

    return trajectories


def generate_ct_trajectory_simple(
    T: int,
    dt: float,
    omega: float,
    dim: int = 2,
    initial_state: Optional[TrajectoryState] = None,
    omega_noise_std: float = 0.0,  # Turn rate noise std
    measurement_noise_std: float | np.ndarray = 0.0,  # Per-axis std of measurement noise
    z_acceleration: Optional[float] | np.ndarray = None,  # Z-axis acceleration
    number_of_trajectories: int = 1,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Generate CT trajectories using simple physical equations with turn rate noise only.
    
    This method uses a simplified state representation:
    - 2D: [x, y, vx, vy, ω] where x,y are positions, vx,vy are velocities, ω is turn rate
    - 3D: [x, y, z, vx, vy, vz, ax, ay, az, ω] where Z-axis follows Constant Acceleration (CA) model
    
    The trajectory follows simple kinematic equations:
    - X-Y plane: Coordinated Turn (CT) motion with constant speed and turn rate
      - x_new = x + vx * dt
      - y_new = y + vy * dt
      - vx_new = vx * cos(ω*dt) - vy * sin(ω*dt)
      - vy_new = vx * sin(ω*dt) + vy * cos(ω*dt)
    - Z-axis (3D only): Constant Acceleration (CA) motion
      - z_new = z + vz*dt + 0.5*az*dt^2
      - vz_new = vz + az*dt
      - az_new = az (constant acceleration)
    - ω_new = ω + noise (only noise on turn rate)
    
    Args:
        T: Number of time steps
        dt: Time step duration
        omega: Turn rate in radians per second
        dim: Dimensionality (2 or 3)
        initial_state: Initial state. If None, random initialization
        omega_noise_std: Standard deviation of turn rate noise
        measurement_noise_std: Per-axis std of measurement noise (scalar or array)
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    assert dim in (2, 3), "dim must be 2 or 3"
    rng = np.random.default_rng(seed)

    # Convert scalars to arrays
    if np.isscalar(measurement_noise_std):
        measurement_noise_std = np.full(dim, float(measurement_noise_std))
    else:
        measurement_noise_std = np.asarray(measurement_noise_std, dtype=float)

    trajectories = []

    for _ in range(number_of_trajectories):
        # Initialize state
        if initial_state is None:
            position = rng.uniform(-10, 10, size=dim)
            velocity = rng.uniform(-5, 5, size=dim)
            state = TrajectoryState(position=position, velocity=velocity, omega=omega)
        else:
            state = TrajectoryState(
                position=initial_state.position.copy(),
                velocity=initial_state.velocity.copy(),
                omega=omega
            )
        
        # Initialize acceleration for 3D case (CA model for Z-axis)
        if dim == 3:
            if state.acceleration is not None:
                az = state.acceleration[2]
            elif z_acceleration is not None:
                az = z_acceleration
            else:
                az = rng.normal(0, 0.1)
                state.acceleration = np.array([0.0, 0.0, az])


        clean_positions = np.zeros((T, dim), dtype=float)
        clean_positions[0] = state.position

        # Current state variables
        x, y = state.position[0], state.position[1]
        vx, vy = state.velocity[0], state.velocity[1]
        current_omega = omega
        

        # Generate trajectory step by step using simple physics
        for t in range(1, T):
            # Add noise to turn rate
            omega_noise = rng.normal(0, omega_noise_std)
            current_omega = omega + omega_noise
            
            # Simple kinematic equations for CT motion (X-Y plane)
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Update velocity components (rotation by ω*dt)
            cos_omega_dt = np.cos(current_omega * dt)
            sin_omega_dt = np.sin(current_omega * dt)
            
            vx_new = vx * cos_omega_dt - vy * sin_omega_dt
            vy_new = vx * sin_omega_dt + vy * cos_omega_dt
            
            # Update state
            state.position[0] = x
            state.position[1] = y
            state.velocity[0] = vx_new
            state.velocity[1] = vy_new
            state.omega = current_omega
            
            # Update local variables for next iteration
            vx, vy = vx_new, vy_new
            
            # Handle z-axis for 3D case (Constant Acceleration model)
            if dim == 3:
                # CA model: z_new = z + vz*dt + 0.5*az*dt^2
                #          vz_new = vz + az*dt
                #          az_new = az (constant acceleration)
                z = state.position[2]
                vz = state.velocity[2]
                
                # Update Z position and velocity
                z_new = z + vz * dt + 0.5 * az * dt**2
                vz_new = vz + az * dt
                
                state.position[2] = z_new
                state.velocity[2] = vz_new
                state.acceleration[2] = az  # Keep acceleration constant

            clean_positions[t] = state.position

        # Add measurement noise
        if np.any(measurement_noise_std > 0):
            measurement_noise = rng.normal(0, measurement_noise_std, size=(T, dim))
            noisy_positions = clean_positions + measurement_noise
        else:
            noisy_positions = clean_positions.copy()

        trajectories.append((noisy_positions, clean_positions, state))

    return trajectories


def generate_singer_trajectory(
    T: int,
    dt: float,
    tau: float | np.ndarray,  # Singer time constant(s)
    dim: int = 2,
    sigma_a: float | np.ndarray = 0.5,  # Singer acceleration std
    initial_state: Optional[TrajectoryState] = None,
    noise_std: float | np.ndarray = 0.0,  # measurement noise std
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    """
    Generate trajectories following the Singer acceleration model.
    
    Args:
        T: Number of time steps
        dt: Time step duration
        tau: Singer time constant (1/alpha)
        dim: Dimensionality (2 or 3)
        sigma_a: Steady-state std of acceleration
        initial_state: Initial state. If None, random initialization
        noise_std: Measurement noise std
        number_of_trajectories: Number of trajectories to generate
        seed: RNG seed
        
    Returns:
        List of (noisy_positions, clean_positions, final_state) tuples
    """
    assert dim in (2, 3), "dim must be 2 or 3"
    rng = np.random.default_rng(seed)

    # Convert scalars to arrays
    if np.isscalar(tau):
        tau = np.full(dim, float(tau))
    else:
        tau = np.asarray(tau, dtype=float)
        
    if np.isscalar(sigma_a):
        sigma_a = np.full(dim, float(sigma_a))
    else:
        sigma_a = np.asarray(sigma_a, dtype=float)
        
    if np.isscalar(noise_std):
        noise_std = np.full(dim, float(noise_std))
    else:
        noise_std = np.asarray(noise_std, dtype=float)

    # Precompute per-axis Singer parameters
    alpha = 1.0 / tau  # decay rate
    exp_alpha_dt = np.exp(-alpha * dt)
    exp_2alpha_dt = np.exp(-2.0 * alpha * dt)

    # Process noise std for acceleration
    accel_process_std = sigma_a * np.sqrt((1.0 - exp_2alpha_dt) / (2.0 * alpha))

    trajectories = []

    for _ in range(number_of_trajectories):
        # Initialize state
        if initial_state is None:
            position = rng.uniform(0, 4, size=dim)
            velocity = rng.uniform(-0.4, 0.4, size=dim)
            acceleration = rng.uniform(-0.1, 0.1, size=dim)
            state = TrajectoryState(position=position, velocity=velocity, acceleration=acceleration, tau=tau)
        else:
            state = TrajectoryState(
                position=initial_state.position.copy(),
                velocity=initial_state.velocity.copy(),
                acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else np.zeros(dim, dtype=float),
                tau=tau
            )

        clean = np.empty((T, dim), dtype=float)
        clean[0] = state.position

        # Simulate trajectory
        for t in range(1, T):
            # Singer acceleration update (first-order Gauss-Markov)
            process_noise = rng.normal(0.0, accel_process_std, size=dim)
            state.acceleration = exp_alpha_dt * state.acceleration + process_noise

            # Compute integrals for position and velocity update
            small_x_mask = (alpha * dt) < 1e-6
            delta_v = np.zeros(dim, dtype=float)
            delta_x = np.zeros(dim, dtype=float)

            for i in range(dim):
                if small_x_mask[i]:
                    # Series expansion for small alpha*dt
                    delta_v[i] = state.acceleration[i] * dt * (1.0 - alpha[i] * dt / 2.0)
                    delta_x[i] = state.velocity[i] * dt + state.acceleration[i] * dt ** 2 / 2.0 * (1.0 - alpha[i] * dt / 3.0)
                else:
                    # Exact integrals from Singer model
                    int_exp = (1.0 - exp_alpha_dt[i]) / alpha[i]
                    delta_v[i] = state.acceleration[i] * int_exp
                    delta_x[i] = state.velocity[i] * dt + state.acceleration[i] * (dt - int_exp)

            # Update state
            state.velocity += delta_v
            state.position += delta_x
            clean[t] = state.position

        # Add measurement noise
        meas_noise = rng.normal(0.0, noise_std, size=(T, dim))
        noisy = clean + meas_noise

        trajectories.append((noisy, clean, state))

    return trajectories


def generate_composite_trajectory(
    trajectory_segments: List[Tuple[str, int, dict]],  # (model_type, steps, params)
    dt: float,
    dim: int = 2,
    initial_state: Optional[TrajectoryState] = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, TrajectoryState]:
    """
    Generate a composite trajectory composed of multiple trajectory segments.
    
    Args:
        trajectory_segments: List of (model_type, steps, params) tuples where:
            - model_type: 'CV', 'CA', 'CT', or 'SINGER'
            - steps: Number of time steps for this segment
            - params: Dictionary of parameters for the specific model
                For CT: omega (required), omega_noise_std, measurement_noise_std, z_acceleration (3D only)
        dt: Time step duration
        dim: Dimensionality (2 or 3)
        initial_state: Initial state. If None, random initialization
        seed: RNG seed
        
    Returns:
        (noisy_positions, clean_positions, final_state) tuple
    """
    rng = np.random.default_rng(seed)
    
    # Initialize state
    if initial_state is None:
        position = rng.uniform(0, 4, size=dim)
        velocity = rng.uniform(-0.4, 0.4, size=dim)
        state = TrajectoryState(position=position, velocity=velocity)
    else:
        state = TrajectoryState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
            acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else None,
            omega=initial_state.omega,
            tau=initial_state.tau
        )
    
    all_noisy = []
    all_clean = []
    
    for model_type, steps, params in trajectory_segments:
        # Generate segment
        if model_type == 'CV':
            results = generate_cv_trajectory(
                T=steps, dt=dt, initial_state=state, 
                vel_change_std=params.get('vel_change_std', 0.0),
                measurement_noise_std=params.get('measurement_noise_std', 0.0),
                number_of_trajectories=1, seed=rng.integers(0, 2**32)
            )
        elif model_type == 'CA':
            state.acceleration = params.get("acceleration", state.acceleration)
            results = generate_ca_trajectory(
                T=steps, dt=dt, initial_state=state,
                measurement_noise_std=params.get('measurement_noise_std', 0.0),
                accel_noise_std=params.get('accel_noise_std', 0.0),
                number_of_trajectories=1, seed=rng.integers(0, 2**32)
            )
        elif model_type == 'CT':
            results = generate_ct_trajectory_simple(
                T=steps, dt=dt, omega=params['omega'], dim=dim, initial_state=state,
                omega_noise_std=params.get('omega_noise_std', 0.0),
                measurement_noise_std=params.get('measurement_noise_std', 0.0),
                z_acceleration=params.get('z_acceleration', None),
                number_of_trajectories=1, seed=rng.integers(0, 2**32)
            )
        elif model_type == 'SINGER':
            results = generate_singer_trajectory(
                T=steps, dt=dt, tau=params['tau'], dim=dim, initial_state=state,
                sigma_a=params.get('sigma_a', 0.5),
                noise_std=params.get('noise_std', 0.0),
                number_of_trajectories=1, seed=rng.integers(0, 2**32)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Extract results
        noisy, clean, final_state = results[0]
        
        # Append to overall trajectory (skip first point to avoid duplication)
        if len(all_noisy) == 0:
            all_noisy.append(noisy)
            all_clean.append(clean)
        else:
            all_noisy.append(noisy[1:])  # Skip first point
            all_clean.append(clean[1:])  # Skip first point
        
        # Update state for next segment
        print(f"Final state: {final_state}")
        state = final_state
    
    # Concatenate all segments
    final_noisy = np.vstack(all_noisy)
    final_clean = np.vstack(all_clean)
    
    return final_noisy, final_clean, state


# Example usage of composite trajectory generation
def example_composite_trajectory():
    """
    Example showing how to generate a composite trajectory:
    - 50 steps of CV motion
    - 30 steps of CT turn
    - 40 steps of CA acceleration
    - 20 steps of CT turn again
    """
    dt = 0.04
    
    # Define trajectory segments
    segments = [
        ('CV', 50, {'vel_change_std': 0.1, 'measurement_noise_std': 0.2}),
        ('CT', 30, {'omega': 2.0, 'noise_std': 0.2}),
        ('CA', 40, {'measurement_noise_std': 0.2, 'accel_noise_std': 0.1}),
        ('CT', 20, {'omega': -1.5, 'noise_std': 0.2}),
    ]
    
    # Generate composite trajectory
    noisy, clean, final_state = generate_composite_trajectory(
        trajectory_segments=segments,
        dt=dt,
        dim=2,
        seed=42
    )
    
    print(f"Generated composite trajectory with {len(noisy)} total steps")
    print(f"Final state: position={final_state.position}, velocity={final_state.velocity}")
    
    return noisy, clean, final_state
