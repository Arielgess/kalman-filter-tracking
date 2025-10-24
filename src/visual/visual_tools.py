import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Optional, Union


def plot_trajectory_simple(
        traj,
        show_clean=None,  # (T,d) optional
        kf_filt=None,  # (T_filt,d) positions OR (T_filt,2d) states [pos,vel] per axis
        title="Trajectory",
        equal_xy=True,
        legend_loc="best",
):
    """
    If kf_filt has different length than traj, it will be padded at the beginning
    to match the length of traj.

    Args:
        traj       : np.ndarray (T,2) or (T,3) noisy positions.
        show_clean : np.ndarray (T,d) clean positions (optional).
        kf_filt    : np.ndarray (T_filt,d) or (T_filt,2d) filtered (optional).
        title      : str
        equal_xy   : bool. Keep equal aspect for XY (and cubic box in 3D if supported).
        legend_loc : str
    """
    traj = np.asarray(traj, dtype=float)
    assert traj.ndim == 2 and traj.shape[1] in (2, 3), "traj must be (T,2) or (T,3)"
    T, d = traj.shape

    def _coerce_positions(arr, name):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.shape == (T, d):
            return arr
        # Accept full state (T_filt, 2d): [pos0, vel0, pos1, vel1, ...]
        if arr.ndim == 2 and arr.shape[1] == 2 * d:
            return arr[:, 0::2]
        raise ValueError(f"{name} must be (T,{d}) positions or (T,{2 * d}) full states; got {arr.shape}")

    def _pad_to_length(arr, target_length, name):
        """Pad array at the beginning to match target_length."""
        if arr is None:
            return None
        T_filt = arr.shape[0]
        if T_filt == target_length:
            return arr
        elif T_filt < target_length:
            # Pad at the beginning with the first value
            pad_length = target_length - T_filt
            padding = np.tile(arr[0:1, :], (pad_length, 1))
            return np.vstack([padding, arr])
        else:
            # Truncate from the end if longer
            return arr[:target_length, :]

    show_clean = _coerce_positions(show_clean, "show_clean") if show_clean is not None else None
    kf_filt = _coerce_positions(kf_filt, "kf_filt") if kf_filt is not None else None

    # Pad filtered data to match trajectory length
    kf_filt = _pad_to_length(kf_filt, T, "kf_filt")

    # Decide 3D only from traj
    has_z = (d == 3) and (np.ptp(traj[:, 2]) > 1e-12)

    if has_z:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="noisy", lw=1.8)
        if show_clean is not None:
            ax.plot(show_clean[:, 0], show_clean[:, 1], show_clean[:, 2],
                    "--", label="clean", lw=1.5, alpha=0.8)
        if kf_filt is not None:
            ax.plot(kf_filt[:, 0], kf_filt[:, 1], kf_filt[:, 2],
                    "-.", label="filtered", lw=1.5)

        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_zlabel("z")
        ax.set_title(title)
        if equal_xy:
            try:
                ax.set_box_aspect((1, 1, 1))  # mpl >= 3.3
            except Exception:
                pass
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()
        return

    # 2D layout: XY path + components over time
    fig, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

    # XY path
    ax_xy.plot(traj[:, 0], traj[:, 1], label="noisy", lw=1.8)
    if show_clean is not None:
        ax_xy.plot(show_clean[:, 0], show_clean[:, 1], "--", label="clean", lw=1.5, alpha=0.8)
    if kf_filt is not None:
        ax_xy.plot(kf_filt[:, 0], kf_filt[:, 1], "-.", label="filtered", lw=1.5)

    ax_xy.set_xlabel("x");
    ax_xy.set_ylabel("y");
    ax_xy.set_title("XY path")
    if equal_xy:
        ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True);
    ax_xy.legend(loc=legend_loc)

    # Components vs time
    t = np.arange(T)
    labels = ["x", "y"] if d == 2 else ["x", "y", "z"]

    def _plot_series(ax, series, style, prefix, **kw):
        for i, name in enumerate(labels):
            ax.plot(t, series[:, i], style, label=f"{prefix} {name}", **kw)

    _plot_series(ax_t, traj, "-", "noisy")
    if show_clean is not None:
        _plot_series(ax_t, show_clean, "--", "clean", alpha=0.8)
    if kf_filt is not None:
        _plot_series(ax_t, kf_filt, "-.", "filtered")

    ax_t.set_xlabel("t (steps)");
    ax_t.set_ylabel("position");
    ax_t.set_title("Components")
    ax_t.grid(True);
    ax_t.legend(loc=legend_loc, ncols=2 if d == 3 else 1)

    plt.tight_layout()
    plt.show()


def plot_trajectories_with_predictions_shifted(
    noisy,                         # (T0,d)
    clean=None,                    # (T?,d) optional
    predictions=None,              # list of (traj_(Ti,d) or (Ti,2d), k_int)
    title="Trajectories",
    show_time_series=True,         # also plot components vs time
    equal_axes=True,               # equal aspect in 2D / cubic in 3D
    legend_loc="best",
):
    """
    Plot noisy & clean trajectories plus k-step prediction trajectories.
    Supports 2D and 3D. Each trajectory can have a different length.

    IMPORTANT: k-step series are plotted against t shifted by +k:
        t_pred = np.arange(T_pred) + k
    so that pred[i] aligns with target at time (i+k).

    Parameters
    ----------
    noisy : np.ndarray of shape (T0, d)
        The measured (noisy) trajectory, d ∈ {2,3}.
    clean : np.ndarray of shape (T?, d), optional
        The underlying clean trajectory (length may differ).
    predictions : list[tuple[np.ndarray, int]], optional
        Each item is (traj, k) where traj is (Ti, d) or (Ti, 2d) full state
        (pos, vel, [pos, vel, ...]). k is the k-step horizon label.
    title : str
    show_time_series : bool
        If True, also show per-component time plots (each with its own time axis).
    equal_axes : bool
    legend_loc : str

    Notes
    -----
    - All supplied trajectories must agree on dimensionality d (2 or 3).
    - If a trajectory is (Ti, 2d), positions are auto-extracted at columns 0,2,(4).
    - Variable lengths are fully supported; each is plotted on its own (shifted) time base.
    """
    def _as_array(x, name):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] not in (2, 3, 4, 6):  # allow (d) or (2d)
            raise ValueError(f"{name} must be 2D array with second dim 2/3 or 4/6; got {x.shape}")
        return x

    def _coerce_positions(arr, d_expected=None, name="array"):
        if arr is None:
            return None, None
        arr = _as_array(arr, name)
        Ti, dj = arr.shape
        # Decide target dimension d
        if d_expected is None:
            if dj in (2, 3): d = dj
            elif dj in (4, 6): d = dj // 2
            else: raise ValueError(f"{name}: unsupported shape {arr.shape}")
        else:
            d = d_expected
        # Extract positions
        if dj == d:
            pos = arr
        elif dj == 2 * d:
            pos = arr[:, 0::2]
        else:
            raise ValueError(f"{name}: dims mismatch; expected d={d} or 2d={2*d}, got {arr.shape}")
        return pos, d

    # --- Ingest and validate dims ---
    noisy_pos, d = _coerce_positions(noisy, None, "noisy")
    if d not in (2, 3):
        raise ValueError("Only 2D or 3D supported.")
    clean_pos, _ = _coerce_positions(clean, d, "clean") if clean is not None else (None, None)

    preds = predictions or []
    pred_pos_list = []
    for idx, (arr, k) in enumerate(preds):
        pos, _ = _coerce_positions(arr, d, f"predictions[{idx}].traj")
        if not (isinstance(k, (int, np.integer)) and k >= 0):
            raise ValueError(f"predictions[{idx}].k must be a nonnegative integer")
        pred_pos_list.append((pos, int(k)))

    # --- Plot spatial path(s) ---
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], noisy_pos[:, 2], "-", lw=1.8, label="noisy")
        if clean_pos is not None:
            ax.plot(clean_pos[:, 0], clean_pos[:, 1], clean_pos[:, 2], "--", lw=1.5, alpha=0.9, label="clean")

        # color/linestyle cycling that remains readable with many k's
        ls_cycle = ["-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
        for i, (pp, k) in enumerate(pred_pos_list):
            # Spatial plot has no explicit time axis; overlay as paths
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], ls_cycle[i % len(ls_cycle)], lw=1.6, label=f"k={k}")

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title)
        if equal_axes:
            try: ax.set_box_aspect((1, 1, 1))
            except Exception: pass
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()

    else:  # d == 2
        ncols = 2 if show_time_series else 1
        fig, axes = plt.subplots(1, ncols, figsize=(12 if show_time_series else 7, 6))
        ax_xy = axes if ncols == 1 else axes[0]

        ax_xy.plot(noisy_pos[:, 0], noisy_pos[:, 1], "-", lw=1.8, label="noisy")
        if clean_pos is not None:
            ax_xy.plot(clean_pos[:, 0], clean_pos[:, 1], "--", lw=1.5, alpha=0.9, label="clean")
        ls_cycle = ["-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
        for i, (pp, k) in enumerate(pred_pos_list):
            ax_xy.plot(pp[:, 0], pp[:, 1], ls_cycle[i % len(ls_cycle)], lw=1.6, label=f"k={k}")

        ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("XY path")
        if equal_axes:
            ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.grid(True); ax_xy.legend(loc=legend_loc)

        # --- Optional time series (with +k shift for predictions) ---
        if show_time_series:
            ax_t = axes[1]
            def _plot_t(ax, series, label_prefix, style="-", **kw):
                t = np.arange(series.shape[0])
                ax.plot(t, series[:, 0], style, lw=1.5, label=f"{label_prefix} x", **kw)
                ax.plot(t, series[:, 1], style, lw=1.5, label=f"{label_prefix} y", **kw)

            # No shift for observed/clean
            _plot_t(ax_t, noisy_pos, "noisy", "-")
            if clean_pos is not None:
                _plot_t(ax_t, clean_pos, "clean", "--", alpha=0.9)

            # Shifted time axis for each k-step prediction
            for i, (pp, k) in enumerate(pred_pos_list):
                t_pred = np.arange(pp.shape[0]) + k   # <<------- Method A: align pred[i] with time i+k
                ax_t.plot(t_pred, pp[:, 0], ls_cycle[i % len(ls_cycle)], lw=1.6, label=f"k={k} x")
                ax_t.plot(t_pred, pp[:, 1], ls_cycle[i % len(ls_cycle)], lw=1.6, label=f"k={k} y")

            ax_t.set_xlabel("t (steps)"); ax_t.set_ylabel("position"); ax_t.set_title("Components vs time")
            ax_t.grid(True)
            handles, labels = ax_t.get_legend_handles_labels()
            if len(labels) <= 12:
                ax_t.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.show()


def calculate_mse_sliding_window(
    predictions: np.ndarray,
    true_values: np.ndarray,
    k_lag: int = 1,
    window_size: int = 10,
    use_clean_signal: bool = True,
    normalize_mse: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MSE in a sliding window with k-lag support.
    
    The function calculates MSE by comparing predictions at time t with true values at time t+k_lag,
    considering the k-lag offset. For example, prediction at t=0 with k_lag=5 should be compared
    to true value at t=5.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (T_pred, d) containing predictions
    true_values : np.ndarray  
        Array of shape (T_true, d) containing true values (measurements or clean signal)
    k_lag : int, default=1
        The k-step lag. Prediction at time t is compared to true value at time t+k_lag
    window_size : int, default=10
        Size of the sliding window for MSE calculation
    use_clean_signal : bool, default=True
        If True, use clean signal for MSE calculation. If False, use measurements.
    normalize_mse : bool, default=True
        If True, normalize MSE by the second moment of the true values
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (mse_values, time_indices) where:
        - mse_values: Array of MSE values for each window position
        - time_indices: Array of time indices corresponding to the center of each window
    """
    predictions = np.asarray(predictions, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    
    if predictions.ndim != 2 or true_values.ndim != 2:
        raise ValueError("predictions and true_values must be 2D arrays")
    
    if predictions.shape[1] != true_values.shape[1]:
        raise ValueError("predictions and true_values must have the same number of dimensions")
    
    T_true = true_values.shape[0]    
    
    # Calculate squared residuals for each valid prediction
    squared_residuals = []
    for t in range(len(predictions)):
        # Prediction at time t should be compared to true value at time t + k_lag
        true_idx = t + k_lag
        if true_idx < T_true:
            residual = true_values[true_idx] - predictions[t]
            squared_residuals.append(residual @ residual)  # Dot product for squared norm
    
    squared_residuals = np.array(squared_residuals)
    
    # Calculate sliding window MSE
    if len(squared_residuals) < window_size:
        # If we have fewer points than window size, return single MSE value
        mse_values = np.array([np.mean(squared_residuals)])
        time_indices = np.array([len(squared_residuals) // 2])
    else:
        # Calculate MSE for each window position
        mse_values = []
        time_indices = []
        
        for i in range(len(squared_residuals) - window_size + 1):
            window_residuals = squared_residuals[i:i + window_size]
            mse = np.mean(window_residuals)
            mse_values.append(mse)
            time_indices.append(i + window_size // 2)  # Center of window
        
        mse_values = np.array(mse_values)
        time_indices = np.array(time_indices)
    
    # Normalize MSE if requested
    if normalize_mse:
        # Use the same normalization as in base_kalman_filter.py
        measurement_second_moment = np.mean((true_values - true_values.mean(axis=0, keepdims=True)) ** 2)
        mse_values = mse_values / measurement_second_moment
    
    return mse_values, time_indices


def plot_mse_over_time(
    mse_values: np.ndarray,
    time_indices: np.ndarray,
    title: str = "MSE Over Time",
    window_size: int = 10,
    k_lag: int = 1,
    figsize: Tuple[int, int] = (10, 6),
    show_grid: bool = True
) -> None:
    """
    Plot MSE values over time in a sliding window.
    
    Parameters
    ----------
    mse_values : np.ndarray
        Array of MSE values for each window position
    time_indices : np.ndarray
        Array of time indices corresponding to the center of each window
    title : str, default="MSE Over Time"
        Title for the plot
    window_size : int, default=10
        Size of the sliding window (for display purposes)
    k_lag : int, default=1
        The k-step lag (for display purposes)
    figsize : Tuple[int, int], default=(10, 6)
        Figure size for the plot
    show_grid : bool, default=True
        Whether to show grid on the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time_indices, mse_values, 'b-', linewidth=2, label=f'MSE (window={window_size}, k={k_lag})')
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend()
    plt.tight_layout()
    plt.show()


def calculate_and_plot_mse_sliding_window(
    predictions: np.ndarray,
    true_values: np.ndarray,
    k_lag: int = 1,
    window_size: int = 10,
    use_clean_signal: bool = True,
    normalize_mse: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MSE in a sliding window and plot it over time.
    
    This is a convenience function that combines calculate_mse_sliding_window and plot_mse_over_time.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (T_pred, d) containing predictions
    true_values : np.ndarray  
        Array of shape (T_true, d) containing true values (measurements or clean signal)
    k_lag : int, default=1
        The k-step lag. Prediction at time t is compared to true value at time t+k_lag
    window_size : int, default=10
        Size of the sliding window for MSE calculation
    use_clean_signal : bool, default=True
        If True, use clean signal for MSE calculation. If False, use measurements.
    normalize_mse : bool, default=True
        If True, normalize MSE by the second moment of the true values
    title : str, optional
        Title for the plot. If None, a default title will be generated.
    figsize : Tuple[int, int], default=(10, 6)
        Figure size for the plot
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (mse_values, time_indices) from calculate_mse_sliding_window
    """
    # Calculate MSE
    mse_values, time_indices = calculate_mse_sliding_window(
        predictions, true_values, k_lag, window_size, use_clean_signal, normalize_mse
    )
    
    # Generate title if not provided
    if title is None:
        signal_type = "clean" if use_clean_signal else "measurements"
        norm_str = "normalized" if normalize_mse else "raw"
        title = f"MSE Over Time (k={k_lag}, window={window_size}, {signal_type}, {norm_str})"
    
    # Plot MSE
    plot_mse_over_time(mse_values, time_indices, title, window_size, k_lag, figsize)
    
    return mse_values, time_indices


