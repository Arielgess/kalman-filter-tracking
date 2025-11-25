# dataset_configurations.py
"""
Generic dataset configuration and generation for motion model trajectories.
Supports CV, CA, CT, and IMM (composite) models with clean polymorphic design.

Key features:
- Model-agnostic specs using inheritance
- Easy extensibility for new motion models
- IMM support with composite trajectories
- Generic parameter storage and metadata
"""

from __future__ import annotations
import math, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np
from src.motion_models.trajectory_generation.route_generation import (
    TrajectoryState, 
    generate_cv_trajectory,
    generate_ca_trajectory,
    generate_ct_trajectory_simple,
    generate_composite_trajectory
)
from src.dataset.random_gen import random_generator


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class ParamRange:
    low: float
    high: float
    kind: Literal["uniform", "loguniform"] = "uniform"

    def sample(self) -> float:
        rng = random_generator.get_rng()
        if self.kind == "uniform":
            return float(rng.uniform(self.low, self.high))
        elif self.kind == "loguniform":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(np.exp(rng.uniform(lo, hi)))
        else:
            raise ValueError(f"Unknown range kind: {self.kind}")

def sample_vec(
    spec: ParamRange | Tuple[float, float] | float | np.ndarray,
    dim: int,
    per_axis: bool = True,
) -> np.ndarray:
    if isinstance(spec, np.ndarray):
        spec = np.asarray(spec, dtype=float)
        if spec.shape == (dim,):
            return spec
        elif spec.size == 1:
            return np.full(dim, float(spec.item()), dtype=float)
        else:
            raise ValueError(f"Array shape {spec.shape} incompatible with dim={dim}")
    elif isinstance(spec, ParamRange):
        if per_axis:
            return np.array([spec.sample() for _ in range(dim)], dtype=float)
        v = spec.sample()
        return np.full(dim, v, dtype=float)
    elif isinstance(spec, tuple):
        return sample_vec(ParamRange(spec[0], spec[1]), dim, per_axis)
    else:
        return np.full(dim, float(spec), dtype=float)

# -----------------------------
# Model Specifications (Polymorphic)
# -----------------------------
@dataclass
class ModelSpec(ABC):
    """Base class for all model specifications"""
    measurement_noise_std: np.ndarray | ParamRange  # Common to all models
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier"""
        pass
    
    @abstractmethod
    def get_generation_params(self, dim: int) -> dict:
        """Return sampled parameters as dict ready for the generator function"""
        pass

@dataclass
class CVSpec(ModelSpec):
    """Constant Velocity model specification"""
    vel_change_std: np.ndarray | ParamRange  # per-axis accel-noise std (process)
    
    @property
    def model_type(self) -> str:
        return 'CV'
    
    def get_generation_params(self, dim: int) -> dict:
        return {
            'vel_change_std': sample_vec(self.vel_change_std, dim, per_axis=True),
            'measurement_noise_std': sample_vec(self.measurement_noise_std, dim, per_axis=True)
        }
    
    def create_kf(self, dt: float, dim: int):
        """Create a Kalman Filter for this CV model specification
        
        Oracle IMM: Uses exact noise parameters from spec (assumes they are sampled values, not ranges)
        """
        from src.imm_models.models_for_imm import IMMConstantVelocityKF, InitializationData
        
        # Convert to arrays if needed
        if isinstance(self.vel_change_std, np.ndarray):
            process_noise = self.vel_change_std
        else:
            process_noise = np.full(dim, float(self.vel_change_std))
        
        if isinstance(self.measurement_noise_std, np.ndarray):
            obs_noise = self.measurement_noise_std
        else:
            obs_noise = np.full(dim, float(self.measurement_noise_std))
        
        init_data = InitializationData(
            observation_noise_std=obs_noise,
            process_noise_std=process_noise
        )
        
        kf = IMMConstantVelocityKF(dim, 7, dim, dt=dt, initialization_data=init_data)
        kf.initialize()
        return kf

@dataclass
class CASpec(ModelSpec):
    """Constant Acceleration model specification"""
    accel_noise_std: np.ndarray | ParamRange
    acceleration: np.ndarray | ParamRange  # Initial acceleration (required, can be sampled or fixed)
    
    @property
    def model_type(self) -> str:
        return 'CA'
    
    def get_generation_params(self, dim: int) -> dict:
        return {
            'accel_noise_std': sample_vec(self.accel_noise_std, dim, per_axis=True),
            'acceleration': sample_vec(self.acceleration, dim, per_axis=True),
            'measurement_noise_std': sample_vec(self.measurement_noise_std, dim, per_axis=True)
        }
    
    def create_kf(self, dt: float, dim: int):
        """Create a Kalman Filter for this CA model specification
        
        Oracle IMM: Uses exact noise parameters from spec (assumes they are sampled values, not ranges)
        """
        from src.imm_models.models_for_imm import IMMConstantAccelerationKF, InitializationData
        
        # Convert to arrays if needed
        if isinstance(self.accel_noise_std, np.ndarray):
            process_noise = self.accel_noise_std
        else:
            process_noise = np.full(dim, float(self.accel_noise_std))
        
        if isinstance(self.measurement_noise_std, np.ndarray):
            obs_noise = self.measurement_noise_std
        else:
            obs_noise = np.full(dim, float(self.measurement_noise_std))
        
        init_data = InitializationData(
            observation_noise_std=obs_noise,
            process_noise_std=process_noise,
            white_accel_density=(process_noise**2 / dt).mean()
        )
        
        kf = IMMConstantAccelerationKF(dim, 7, dim, dt=dt, initialization_data=init_data)
        kf.initialize()
        return kf

@dataclass
class CTSpec(ModelSpec):
    """Coordinated Turn model specification"""
    omega: ParamRange | float  # Turn rate
    omega_noise_std: ParamRange | float  # Turn rate noise
    z_acceleration: Optional[ParamRange | float] = None  # Z-axis acceleration for 3D
    
    @property
    def model_type(self) -> str:
        return 'CT'
    
    def get_generation_params(self, dim: int) -> dict:
        params = {
            'omega': sample_vec(self.omega, 1, per_axis=False)[0],
            'omega_noise_std': sample_vec(self.omega_noise_std, 1, per_axis=False)[0],
            'measurement_noise_std': sample_vec(self.measurement_noise_std, dim, per_axis=True)
        }
        if dim == 3 and self.z_acceleration is not None:
            params['z_acceleration'] = sample_vec(self.z_acceleration, 1, per_axis=False)[0]
        return params
    
    def create_kf(self, dt: float, dim: int):
        """Create a Kalman Filter for this CT model specification
        
        Oracle IMM: Uses exact noise parameters from spec (assumes they are sampled values, not ranges)
        """
        from src.imm_models.models_for_imm import IMMCoordinatedTurnKF, InitializationData
        
        # Get omega noise std
        omega_std = float(self.omega_noise_std)
        
        # Get measurement noise
        if isinstance(self.measurement_noise_std, np.ndarray):
            obs_noise = self.measurement_noise_std
        else:
            obs_noise = np.full(dim, float(self.measurement_noise_std))
        
        # For CT, process noise is typically lower
        process_noise = np.full(dim, omega_std * 0.5)
        
        init_data = InitializationData(
            observation_noise_std=obs_noise,
            process_noise_std=process_noise,
            omega_std=omega_std
        )
        
        kf = IMMCoordinatedTurnKF(dim, 7, dim, dt=dt, initialization_data=init_data)
        kf.initialize()
        return kf

@dataclass
class SegmentSpec:
    """Single segment in a composite (IMM) trajectory"""
    model_spec: ModelSpec  # CV, CA, CT, etc.
    T: int  # Time steps for this segment (fixed value for batching)

@dataclass
class IMMSpec(ModelSpec):
    """Interacting Multiple Model (composite trajectory) specification"""
    segments: List[SegmentSpec]  # List of segments that compose the trajectory
    randomize_order: bool = False  # Simple randomization: shuffle segment order
    randomize_blueprint: bool = False  # Advanced: treat as blueprints with random lengths
    min_segment_length: int = 10  # Min segment length for blueprint mode
    max_segment_length: int = 40  # Max segment length for blueprint mode
    
    @property
    def model_type(self) -> str:
        return 'IMM'
    
    def get_generation_params(self, dim: int) -> dict:
        """Generate parameters for all segments"""
        segment_params = []
        for seg in self.segments:
            params = seg.model_spec.get_generation_params(dim)
            segment_params.append({
                'model_type': seg.model_spec.model_type,
                'T': seg.T,
                'params': params
            })
        return {
            'segments': segment_params,
            'randomize_order': self.randomize_order,
            'randomize_blueprint': self.randomize_blueprint,
            'min_segment_length': self.min_segment_length,
            'max_segment_length': self.max_segment_length
        }
    
    def create_imm(self, dt: float, dim: int):
        """
        Create an IMM estimator by instantiating a KF for EACH segment using the
        segment's ModelSpec.create_kf(), then building an IMM with a diagonal-dominant
        transition matrix and uniform initial mode probabilities.
        """
        from filterpy.kalman import IMMEstimator

        # Build filters: one per segment (no dedup unless desired explicitly)
        filters = []
        for seg in self.segments:
            if not hasattr(seg.model_spec, "create_kf"):
                raise ValueError(f"ModelSpec of type '{seg.model_spec.model_type}' does not implement create_kf(dt, dim)")
            filters.append(seg.model_spec.create_kf(dt=dt, dim=dim))

        n_filters = len(filters)
        if n_filters == 0:
            raise ValueError("IMMSpec.create_imm() called with no segments defined")

        # Transition matrix M: diagonal-dominant, small off-diagonal (e.g., 0.01)
        off_diag = 0.01
        M = np.full((n_filters, n_filters), off_diag, dtype=float)
        np.fill_diagonal(M, 1.0 - off_diag * (n_filters - 1))

        # Uniform initial mode probabilities
        mu = np.full(n_filters, 1.0 / n_filters, dtype=float)

        return IMMEstimator(filters, mu, M)

# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class ClassConfig:
    """Configuration for a single trajectory class"""
    name: str
    model_spec: ModelSpec  # Polymorphic - can be any ModelSpec subclass
    n_trajectories: int = 1000

@dataclass
class DatasetConfig:
    """Configuration for the entire dataset"""
    seed: int = 7
    dim: int = 3
    dt: float = 0.04
    T: int = 100  # Fixed time steps for all trajectories (for easier batching)
    # aligned initial-condition priors (E)
    init_pos_range: Tuple[float, float] = (-100.0, 100.0)
    init_speed_range: Tuple[float, float] = (-10.0, 25.0)
    # classes
    classes: List[ClassConfig] = field(default_factory=list)
    store_clean: bool = True
    # regime bins for parameter-aware split (F)
    split_bins: Dict[str, List[float]] = field(default_factory=lambda: {
        "vchange": [0.0, 0.05, 0.12, 0.25, 0.45, 0.8],
        "meas":    [0.0, 0.05, 0.12, 0.25, 0.45, 0.8],
        "accel":   [0.0, 0.05, 0.12, 0.25, 0.45, 0.8],
        "omega":   [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0],
    })

# -----------------------------
# Trajectory Generator Registry
# -----------------------------
class TrajectoryGenerator:
    """Registry of trajectory generation functions"""
    
    @staticmethod
    def generate(
        model_type: str,
        T: Optional[int], #Will be ignored for IMM models
        dt: float,
        initial_state: TrajectoryState,
        params: dict,
        seed: int,
        dim: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, TrajectoryState]:
        """Dispatch to appropriate generator based on model type"""
        if model_type == 'CV':
            return generate_cv_trajectory(
                T=T, dt=dt, initial_state=initial_state,
                vel_change_std=params['vel_change_std'],
                measurement_noise_std=params['measurement_noise_std'],
                number_of_trajectories=1, seed=seed
            )[0]
        elif model_type == 'CA':
            # Set acceleration (always present for CA)
            initial_state.acceleration = params['acceleration']
            return generate_ca_trajectory(
                T=T, dt=dt, initial_state=initial_state,
                accel_noise_std=params['accel_noise_std'],
                measurement_noise_std=params['measurement_noise_std'],
                number_of_trajectories=1, seed=seed
            )[0]
        elif model_type == 'CT':
            return generate_ct_trajectory_simple(
                T=T, dt=dt, omega=params['omega'], dim=dim,
                initial_state=initial_state,
                omega_noise_std=params['omega_noise_std'],
                measurement_noise_std=params['measurement_noise_std'],
                z_acceleration=params.get('z_acceleration'),
                number_of_trajectories=1, seed=seed
            )[0]
        elif model_type == 'IMM':
            # Build segment list for composite generation
            segments = [(seg['model_type'], seg['T'], seg['params']) 
                       for seg in params['segments']]
            return generate_composite_trajectory(
                trajectory_segments=segments,
                dt=dt, dim=dim, initial_state=initial_state, seed=seed,
                randomize_order=params.get('randomize_order', False),
                randomize_blueprint=params.get('randomize_blueprint', False),
                min_segment_length=params.get('min_segment_length', 10),
                max_segment_length=params.get('max_segment_length', 40),
                target_T=T  # Pass the dataset-level T for blueprint mode
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# -----------------------------
# Default: 4 CV classes with overlapping ranges (B)
#   All four differ, but there's intentional overlap in both process & measurement noise.
# -----------------------------
def default_config(dim: int = 2) -> DatasetConfig:
    """Create default configuration with 4 CV classes"""
    # process noise std ranges (per-axis accel noise in CV generator)
    # measurement noise std ranges (per-axis)
    CV_A = ClassConfig(
        name="CV_A",
        model_spec=CVSpec(
            vel_change_std=np.array([0.2, 0.2], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_B = ClassConfig(
        name="CV_B",
        model_spec=CVSpec(
            vel_change_std=np.array([0.4, 0.4], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_C = ClassConfig(
        name="CV_C",
        model_spec=CVSpec(
            vel_change_std=np.array([0.3, 0.3], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_D = ClassConfig(
        name="CV_D",
        model_spec=CVSpec(
            vel_change_std=np.array([0.05, 0.05], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )

    return DatasetConfig(
        seed=7,
        dim=dim,
        dt=0.04,
        T=100,  # Fixed time steps
        init_pos_range=(-100.0, 100.0),
        init_speed_range=(-25.0, 25.0),
        classes=[CV_A, CV_B, CV_C, CV_D],
        store_clean=True,
    )

# -----------------------------
# Single unified RNG + pipeline (C)
# -----------------------------
def sample_initial_state(
    dim: int,
    pos_r: Tuple[float, float],
    speed_r: Tuple[float, float],
) -> TrajectoryState:
    rng = random_generator.get_rng()
    pos = rng.uniform(pos_r[0], pos_r[1], size=dim)
    vel = rng.uniform(speed_r[0], speed_r[1], size=dim)
    return TrajectoryState(position=pos, velocity=vel)

# -----------------------------
# Core dataset generation (Generic)
# -----------------------------
def generate_dataset(cfg: DatasetConfig) -> Dict[str, Any]:
    """Generate dataset with support for CV, CA, CT, and IMM models
    
    Returns:
        Dictionary mapping class names to their data:
        {
            "ClassName1": {"X": [...], "Y": [...], "meta": [...]},
            "ClassName2": {"X": [...], "Y": [...], "meta": [...]},
            ...
            "config": DatasetConfig as dict
        }
    """
    # Initialize singleton RNG with dataset seed
    
    # Initialize dictionary to store data per class
    dataset = {}

    for cls in cfg.classes:
        # Initialize lists for this class
        X_class = []
        Y_class = []
        META_class = []
        
        for _ in range(cls.n_trajectories):
            # For non-IMM models, use dataset-level T
            # For IMM with blueprint randomization, use dataset-level T as target
            # For IMM without randomization, T is determined by sum of segment specs
            if cls.model_spec.model_type != 'IMM':
                T = cfg.T
            elif hasattr(cls.model_spec, 'randomize_blueprint') and cls.model_spec.randomize_blueprint:
                T = cfg.T  # Blueprint mode needs target length
            else:
                T = None  # Sequential IMM uses segment lengths
            
            tr_seed = random_generator.get_rng().integers(0, 2**32 - 1)

            # Sample initial state
            init_state = sample_initial_state(
                cfg.dim, cfg.init_pos_range, cfg.init_speed_range
            )
            
            # Get model-specific parameters (polymorphic call)
            params = cls.model_spec.get_generation_params(cfg.dim)
            
            # Generate trajectory using registry
            noisy, clean, final_state = TrajectoryGenerator.generate(
                model_type=cls.model_spec.model_type,
                T=T,
                dt=cfg.dt,
                initial_state=init_state,
                params=params,
                seed=tr_seed,
                dim=cfg.dim
            )

            # Copy arrays to avoid reference issues
            X_class.append(noisy.copy())
            if cfg.store_clean:
                Y_class.append(clean.copy())
            
            # Generic metadata storage
            initial_state_dict = {
                "position": init_state.position.tolist(),
                "velocity": init_state.velocity.tolist(),
            }
            if init_state.acceleration is not None:
                initial_state_dict["acceleration"] = init_state.acceleration.tolist()
            
            # Store all parameters generically
            META_class.append({
                "class": cls.name,
                "model_type": cls.model_spec.model_type,
                "T": T if T is not None else len(noisy),
                "dt": cfg.dt,
                "seed": tr_seed,
                "initial_state": initial_state_dict,
                "params": _serialize_params(params),  # Recursively serialize all params
                #"bins": _compute_bins(params, cfg.split_bins, cls.model_spec.model_type)
            })
        
        # Store this class's data in the dataset dictionary
        dataset[cls.name] = {
            "X": X_class,
            "Y": Y_class if cfg.store_clean else None,
            "meta": META_class
        }
    
    # Add config to the dataset
    dataset["config"] = _cfg_to_py(cfg)
    
    return dataset

# -----------------------------
# Helper Functions
# -----------------------------
def _serialize_params(params: dict) -> dict:
    """Recursively convert numpy arrays to lists for JSON serialization"""
    result = {}
    for key, val in params.items():
        if isinstance(val, np.ndarray):
            result[key] = val.tolist()
        elif isinstance(val, list):
            result[key] = [_serialize_params(v) if isinstance(v, dict) else v for v in val]
        elif isinstance(val, dict):
            result[key] = _serialize_params(val)
        else:
            result[key] = float(val) if isinstance(val, (np.floating, np.integer)) else val
    return result

def _compute_bins(params: dict, split_bins: dict, model_type: str) -> dict:
    """Compute regime bins based on model type and parameters"""
    bins = {}
    
    if model_type == 'CV':
        if 'vel_change_std' in params:
            bins['vchange'] = _digitize_safe(
                float(np.mean(params['vel_change_std'])), 
                split_bins.get('vchange', [])
            )
        if 'measurement_noise_std' in params:
            bins['meas'] = _digitize_safe(
                float(np.mean(params['measurement_noise_std'])), 
                split_bins.get('meas', [])
            )
    elif model_type == 'CA':
        if 'accel_noise_std' in params:
            bins['accel'] = _digitize_safe(
                float(np.mean(params['accel_noise_std'])), 
                split_bins.get('accel', [])
            )
        if 'measurement_noise_std' in params:
            bins['meas'] = _digitize_safe(
                float(np.mean(params['measurement_noise_std'])), 
                split_bins.get('meas', [])
            )
    elif model_type == 'CT':
        if 'omega' in params:
            bins['omega'] = _digitize_safe(
                float(params['omega']), 
                split_bins.get('omega', [])
            )
        if 'measurement_noise_std' in params:
            bins['meas'] = _digitize_safe(
                float(np.mean(params['measurement_noise_std'])), 
                split_bins.get('meas', [])
            )
    elif model_type == 'IMM':
        # For IMM, bin based on aggregated segment properties
        # Could be extended to track segment-specific bins
        bins['composite'] = 0  # Placeholder for now
    
    return bins

def _digitize_safe(x: float, edges: List[float]) -> int:
    """Safely bin a value into discrete bins"""
    if not edges:
        return 0
    return int(np.clip(np.digitize([x], edges, right=False)[0] - 1, 0, len(edges) - 2))

def _cfg_to_py(cfg: DatasetConfig) -> Dict[str, Any]:
    def dc(x):
        if hasattr(x, "__dataclass_fields__"):
            return {k: dc(getattr(x, k)) for k in x.__dataclass_fields__}
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: dc(v) for k, v in x.items()}
        if isinstance(x, list):
            return [dc(v) for v in x]
        return x
    return dc(cfg)

# -----------------------------
# Parameter-aware split (F)
# -----------------------------
def stratified_split_by_bins(meta: List[Dict[str, Any]], ratios=(0.7, 0.15, 0.15)):
    """
    Split indices into train/val/test with minimal overlap in (vchange, meas) bins
    and preserving class balance.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, m in enumerate(meta):
        b = m["bins"]
        key = f'{m["class"]}|{b["vchange"]}|{b["meas"]}'
        buckets[key].append(i)

    train, val, test = [], [], []
    for key, idxs in buckets.items():
        rng = np.random.default_rng(hash(key) & 0xFFFFFFFF)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train += idxs[:n_train]
        val += idxs[n_train:n_train + n_val]
        test += idxs[n_train + n_val:]
    return train, val, test

# -----------------------------
# Quick overlap diagnostics (B/E/G) - Model-Agnostic
# -----------------------------
def quick_overlap_checks(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize per-class means and pairwise histogram overlaps.
    Works with any model type (CV, CA, CT, IMM).
    If overlaps < ~0.3 persistently, your ranges may leak labels.
    """
    from collections import defaultdict

    def extract(meta, cls, key):
        """Extract parameter values across all trajectories of a class"""
        vals = []
        for m in meta:
            if m["class"] != cls:
                continue
            if key in m["params"]:
                arr = m["params"][key]
                if isinstance(arr, list):
                    vals.append(float(np.mean(np.array(arr))))
                else:
                    vals.append(float(arr))
        return np.array(vals, dtype=float) if vals else np.array([])

    classes = sorted(set(m["class"] for m in meta))
    per_class = {}
    
    for c in classes:
        # Get model type for this class
        model_type = next((m["model_type"] for m in meta if m["class"] == c), None)
        
        # Extract relevant parameters based on model type
        class_stats = {
            "model_type": model_type,
            "T_mean": float(np.mean([m["T"] for m in meta if m["class"] == c])),
        }
        
        # Add model-specific parameters
        if model_type == 'CV':
            vc = extract(meta, c, "vel_change_std")
            ms = extract(meta, c, "measurement_noise_std")
            if len(vc): class_stats["vchange_mean"] = float(vc.mean())
            if len(ms): class_stats["meas_mean"] = float(ms.mean())
        elif model_type == 'CA':
            ac = extract(meta, c, "accel_noise_std")
            ms = extract(meta, c, "measurement_noise_std")
            if len(ac): class_stats["accel_mean"] = float(ac.mean())
            if len(ms): class_stats["meas_mean"] = float(ms.mean())
        elif model_type == 'CT':
            om = extract(meta, c, "omega")
            ms = extract(meta, c, "measurement_noise_std")
            if len(om): class_stats["omega_mean"] = float(om.mean())
            if len(ms): class_stats["meas_mean"] = float(ms.mean())
        
        per_class[c] = class_stats

    def overlap(a: np.ndarray, b: np.ndarray, bins=30):
        """Calculate histogram overlap between two distributions"""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if lo == hi:
            return 1.0
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        return float(np.sum(np.minimum(ha, hb)) * (hi - lo) / bins)

    # Compute pairwise overlaps for common parameters
    pairwise = []
    for i, ci in enumerate(classes):
        for cj in classes[i+1:]:
            # Measurement noise is common to all models
            mci = extract(meta, ci, "measurement_noise_std")
            mcj = extract(meta, cj, "measurement_noise_std")
            
            overlap_data = {
                "pair": (ci, cj),
                "meas_overlap": overlap(mci, mcj) if len(mci) and len(mcj) else None,
            }
            
            # Add model-specific overlaps if both classes use same model type
            model_i = next((m["model_type"] for m in meta if m["class"] == ci), None)
            model_j = next((m["model_type"] for m in meta if m["class"] == cj), None)
            
            if model_i == model_j == 'CV':
                vci = extract(meta, ci, "vel_change_std")
                vcj = extract(meta, cj, "vel_change_std")
                overlap_data["vchange_overlap"] = overlap(vci, vcj) if len(vci) and len(vcj) else None
            elif model_i == model_j == 'CA':
                aci = extract(meta, ci, "accel_noise_std")
                acj = extract(meta, cj, "accel_noise_std")
                overlap_data["accel_overlap"] = overlap(aci, acj) if len(aci) and len(acj) else None
            elif model_i == model_j == 'CT':
                omi = extract(meta, ci, "omega")
                omj = extract(meta, cj, "omega")
                overlap_data["omega_overlap"] = overlap(omi, omj) if len(omi) and len(omj) else None
            
            pairwise.append(overlap_data)

    return {"per_class": per_class, "pairwise_overlap": pairwise}

# -----------------------------
# Example Configurations for Different Models
# -----------------------------
def mixed_model_config(dim: int = 2) -> DatasetConfig:
    """Example configuration with CV, CA, CT, and IMM models"""
    
    # CV classes
    cv_fast = ClassConfig(
        name="CV_Fast",
        model_spec=CVSpec(
            vel_change_std=ParamRange(0.3, 0.5),
            measurement_noise_std=np.array([0.4, 0.4]),
        ),
        n_trajectories=25
    )
    
    cv_slow = ClassConfig(
        name="CV_Slow",
        model_spec=CVSpec(
            vel_change_std=np.array([0.05, 0.05]),
            measurement_noise_std=np.array([0.4, 0.4]),
        ),
        n_trajectories=25
    )
    
    # CA class
    ca_smooth = ClassConfig(
        name="CA_Smooth",
        model_spec=CASpec(
            accel_noise_std=ParamRange(0.05, 0.15),
            acceleration=np.array([0.5, 0.5]),  # Fixed initial acceleration
            measurement_noise_std=np.array([0.3, 0.3]),
        ),
        n_trajectories=25
    )
    
    # CT class
    ct_turning = ClassConfig(
        name="CT_Turning",
        model_spec=CTSpec(
            omega=ParamRange(-0.3, 0.3),  # Variable turn rate
            omega_noise_std=0.01,
            measurement_noise_std=np.array([0.3, 0.3]),
        ),
        n_trajectories=25
    )
    
    # IMM class (composite: CV -> CA -> CV)
    imm_mixed = ClassConfig(
        name="IMM_Mixed",
        model_spec=IMMSpec(
            segments=[
                SegmentSpec(
                    model_spec=CVSpec(
                        vel_change_std=np.array([0.1, 0.1]),
                        measurement_noise_std=np.array([0.3, 0.3]),
                    ),
                    T=40
                ),
                SegmentSpec(
                    model_spec=CASpec(
                        accel_noise_std=np.array([0.1, 0.1]),
                        acceleration=np.array([1.0, 1.0]),
                        measurement_noise_std=np.array([0.3, 0.3]),
                    ),
                    T=30
                ),
                SegmentSpec(
                    model_spec=CVSpec(
                        vel_change_std=np.array([0.1, 0.1]),
                        measurement_noise_std=np.array([0.3, 0.3]),
                    ),
                    T=30
                ),
            ]
        ),
        n_trajectories=25
    )
    
    return DatasetConfig(
        seed=42,
        dim=dim,
        dt=0.04,
        T=100,  # Fixed time for non-IMM models
        init_pos_range=(-50.0, 50.0),
        init_speed_range=(2.0, 15.0),
        classes=[cv_fast, cv_slow, ca_smooth, ct_turning, imm_mixed],
        store_clean=True,
    )

# -----------------------------
# Main (example)
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Default CV-only configuration")
    print("=" * 60)
    cfg = default_config(dim=2)
    data = generate_dataset(cfg)
    print(f"Generated {len(data['X'])} trajectories")
    print(f"Classes: {[c.name for c in cfg.classes]}")
    print(f"Model types: {set(m['model_type'] for m in data['meta'])}")
    
    diag = quick_overlap_checks(data["meta"])
    print("\nPer-class statistics:")
    print(json.dumps(diag["per_class"], indent=2))

    train, val, test = stratified_split_by_bins(data["meta"])
    print(f"\nSplit sizes: train={len(train)}  val={len(val)}  test={len(test)}")
    
    print("\n" + "=" * 60)
    print("Example 2: Mixed model configuration (CV, CA, CT, IMM)")
    print("=" * 60)
    cfg_mixed = mixed_model_config(dim=2)
    data_mixed = generate_dataset(cfg_mixed)
    print(f"Generated {len(data_mixed['X'])} trajectories")
    print(f"Classes: {[c.name for c in cfg_mixed.classes]}")
    print(f"Model types: {set(m['model_type'] for m in data_mixed['meta'])}")
    
    diag_mixed = quick_overlap_checks(data_mixed["meta"])
    print("\nPer-class statistics:")
    print(json.dumps(diag_mixed["per_class"], indent=2))
    
    # Check IMM metadata to see segment information
    imm_sample = next((m for m in data_mixed['meta'] if m['model_type'] == 'IMM'), None)
    if imm_sample:
        print("\nIMM trajectory sample metadata:")
        print(json.dumps(imm_sample, indent=2))
    
    # Optional save:
    # np.savez_compressed("trajectories.npz",
    #     X=[x.astype(np.float32) for x in data['X']],
    #     Y=[y.astype(np.float32) for y in (data['Y'] or [])])
    # with open("trajectories_meta.json", "w") as f:
    #     json.dump(data["meta"], f, indent=2)
