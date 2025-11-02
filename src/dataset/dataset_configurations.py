# cv_traj_config.py
"""
Structured config/orchestrator for CV-only synthetic trajectories with 4 classes.
Keeps anti-bias rules B/C/E/F/G while staying lean.

Requires in scope/import:
  - TrajectoryState
  - generate_cv_trajectory(T, dt, initial_state, vel_change_std, measurement_noise_std, number_of_trajectories=1, seed)
"""

from __future__ import annotations
import math, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Literal
import numpy as np
from src.motion_models.trajectory_generation.route_generation import TrajectoryState, generate_cv_trajectory

# ---- If your generators are in another file, uncomment and edit this import ----
# from your_generators_file import TrajectoryState, generate_cv_trajectory


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class ParamRange:
    low: float
    high: float
    kind: Literal["uniform", "loguniform"] = "uniform"

    def sample(self, rng: np.random.Generator) -> float:
        if self.kind == "uniform":
            return float(rng.uniform(self.low, self.high))
        elif self.kind == "loguniform":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(np.exp(rng.uniform(lo, hi)))
        else:
            raise ValueError(f"Unknown range kind: {self.kind}")

def sample_vec(
    rng: np.random.Generator,
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
            return np.array([spec.sample(rng) for _ in range(dim)], dtype=float)
        v = spec.sample(rng)
        return np.full(dim, v, dtype=float)
    elif isinstance(spec, tuple):
        return sample_vec(rng, ParamRange(spec[0], spec[1]), dim, per_axis)
    else:
        return np.full(dim, float(spec), dtype=float)

# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class CVSpec:
    vel_change_std: np.ndarray | ParamRange      # per-axis accel-noise std (process)
    measurement_noise_std: np.ndarray | ParamRange  # per-axis measurement noise std

@dataclass
class ClassConfig:
    name: str
    cv: CVSpec
    n_trajectories: int = 1000

@dataclass
class DatasetConfig:
    seed: int = 7
    dim: int = 3
    dt: float = 0.04
    T_range: Tuple[int, int] = (100, 150)            # aligned across classes (E)
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
    })

# -----------------------------
# Default: 4 CV classes with overlapping ranges (B)
#   All four differ, but there’s intentional overlap in both process & measurement noise.
# -----------------------------
def default_config(dim: int = 2) -> DatasetConfig:
    # process noise std ranges (per-axis accel noise in CV generator)
    # measurement noise std ranges (per-axis)
    CV_A = ClassConfig(
        name="CV_A",
        cv=CVSpec(
            vel_change_std=np.array([0.2, 0.2], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_B = ClassConfig(
        name="CV_B",
        cv=CVSpec(
            vel_change_std=np.array([0.4, 0.4], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_C = ClassConfig(
        name="CV_C",
        cv=CVSpec(
            vel_change_std=np.array([0.3, 0.3], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )
    CV_D = ClassConfig(
        name="CV_D",
        cv=CVSpec(
            vel_change_std=np.array([0.05, 0.05], dtype=float),
            measurement_noise_std=np.array([0.4, 0.4], dtype=float),
        ),
        n_trajectories=50
    )

    return DatasetConfig(
        seed=7,
        dim=dim,
        dt=0.04,
        T_range=(100, 100),
        init_pos_range=(-100.0, 100.0),
        init_speed_range=(3.0, 25.0),
        classes=[CV_A, CV_B, CV_C, CV_D],
        store_clean=True,
    )

# -----------------------------
# Single unified RNG + pipeline (C)
# -----------------------------
def sample_initial_state(
    rng: np.random.Generator,
    dim: int,
    pos_r: Tuple[float, float],
    speed_r: Tuple[float, float],
) -> TrajectoryState:
    pos = rng.uniform(pos_r[0], pos_r[1], size=dim)
    vel = rng.uniform(speed_r[0], speed_r[1], size=dim)
    return TrajectoryState(position=pos, velocity=vel)

# -----------------------------
# Core dataset generation
# -----------------------------
def generate_dataset(cfg: DatasetConfig) -> Dict[str, Any]:
    master = np.random.default_rng(cfg.seed)
    X, Y, META = [], [], []

    for cls in cfg.classes:
        for _ in range(cls.n_trajectories):
            T = int(master.integers(cfg.T_range[0], cfg.T_range[1] + 1))
            tr_seed = int(master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(tr_seed)

            init_state = sample_initial_state(
                rng, cfg.dim, cfg.init_pos_range, cfg.init_speed_range
            )
            # sample per-axis params with overlap (B)
            vchange = sample_vec(rng, cls.cv.vel_change_std, cfg.dim, per_axis=True)
            meas = sample_vec(rng, cls.cv.measurement_noise_std, cfg.dim, per_axis=True)

            noisy, clean, _ = generate_cv_trajectory(
                T=T, dt=cfg.dt, initial_state=init_state,
                vel_change_std=vchange, measurement_noise_std=meas,
                number_of_trajectories=1, seed=int(rng.integers(0, 2**32 - 1))
            )[0]

            X.append(noisy)
            if cfg.store_clean:
                Y.append(clean)
            initial_state_dict = {
                "position": init_state.position.tolist(),
                "velocity": init_state.velocity.tolist(),
            }
            if init_state.acceleration is not None:
                initial_state_dict["acceleration"] = init_state.acceleration.tolist()
            
            META.append({
                "class": cls.name,
                "T": T,
                "dt": cfg.dt,
                "seed": tr_seed,
                "initial_state": initial_state_dict,
                "params": {
                    "vel_change_std": vchange.tolist(),
                    "measurement_noise_std": meas.tolist(),
                },
                # bins for split-by-regime (F)
                "bins": {
                    "vchange": _digitize_safe(float(np.mean(vchange)), cfg.split_bins["vchange"]),
                    "meas":    _digitize_safe(float(np.mean(meas)), cfg.split_bins["meas"]),
                }
            })

    return {"X": X, "Y": Y if cfg.store_clean else None, "meta": META, "config": _cfg_to_py(cfg)}

def _digitize_safe(x: float, edges: List[float]) -> int:
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
# Quick overlap diagnostics (B/E/G)
# -----------------------------
def quick_overlap_checks(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize per-class means and pairwise histogram overlaps for vchange & meas.
    If overlaps < ~0.3 persistently, your ranges may leak labels.
    """
    from collections import defaultdict

    def extract(meta, cls, key):
        vals = []
        for m in meta:
            if m["class"] != cls: continue
            arr = m["params"][key]
            vals.append(float(np.mean(np.array(arr))))
        return np.array(vals, dtype=float) if vals else np.array([])

    classes = sorted(set(m["class"] for m in meta))
    per_class = {}
    for c in classes:
        vc = extract(meta, c, "vel_change_std")
        ms = extract(meta, c, "measurement_noise_std")
        T = np.array([m["T"] for m in meta if m["class"] == c], dtype=float)
        per_class[c] = {
            "vchange_mean": float(vc.mean()) if len(vc) else None,
            "meas_mean": float(ms.mean()) if len(ms) else None,
            "T_mean": float(T.mean()) if len(T) else None,
        }

    def overlap(a: np.ndarray, b: np.ndarray, bins=30):
        if len(a)==0 or len(b)==0: return 0.0
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if lo == hi: return 1.0
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        return float(np.sum(np.minimum(ha, hb)) * (hi - lo) / bins)

    pairwise = []
    for i, ci in enumerate(classes):
        for cj in classes[i+1:]:
            vci = extract(meta, ci, "vel_change_std")
            vcj = extract(meta, cj, "vel_change_std")
            mci = extract(meta, ci, "measurement_noise_std")
            mcj = extract(meta, cj, "measurement_noise_std")
            pairwise.append({
                "pair": (ci, cj),
                "vchange_overlap": overlap(vci, vcj),
                "meas_overlap": overlap(mci, mcj),
            })

    return {"per_class": per_class, "pairwise_overlap": pairwise}

# -----------------------------
# Main (example)
# -----------------------------
if __name__ == "__main__":
    cfg = default_config(dim=2)
    data = generate_dataset(cfg)
    print(f"Generated {len(data['X'])} trajectories "
          f"(classes: {[c.name for c in cfg.classes]})")

    diag = quick_overlap_checks(data["meta"])
    print(json.dumps(diag, indent=2))

    train, val, test = stratified_split_by_bins(data["meta"])
    print(f"Split sizes: train={len(train)}  val={len(val)}  test={len(test)}")

    # Optional save:
    # np.savez_compressed("cv_only_positions.npz",
    #     X=[x.astype(np.float32) for x in data['X']],
    #     Y=[y.astype(np.float32) for y in (data['Y'] or [])])
    # with open("cv_only_meta.json", "w") as f:
    #     json.dump(data["meta"], f, indent=2)
