import numpy as np
from scipy.linalg import expm

from src.filters.base_kalman_filter import BaseKalmanFilter, InitializationData


class SingerKF(BaseKalmanFilter):
    """
    Singer model per-axis state: [x, v, a]; total state is block-diag across axes.
    Measurement: positions only (dim_measurement == dim).
    """

    def __init__(self, dim: int, dt: float, tau, initialization_data: InitializationData):
        # dim_state = 3 * dim, dim_measurement = dim
        super().__init__(dim=dim, dim_state=3 * dim, dim_measurement=dim, dt=dt,
                         initialization_data=initialization_data)

        # tau can be scalar or array-like of length dim
        tau = np.asarray(tau, dtype=float)
        if tau.size == 1:
            self.tau = np.full((dim,), float(tau))
        elif tau.size == dim:
            self.tau = tau
        else:
            raise ValueError("tau must be a scalar or an array of length 'dim'")

    def initialize(self, x0=None, P0=None):
        super().initialize(x0, P0)

        if self.initialization_data.process_noise_std.shape != (self.dim,):
            raise ValueError("process_noise_std must have shape (dim,) — interpreted as sigma_a per axis")
        if self.initialization_data.observation_noise_std.shape != (self.dim,):
            raise ValueError("observation_noise_std must have shape (dim,)")

        sig_a = self.initialization_data.process_noise_std.astype(float)  # sigma_a per axis
        alpha = 1.0 / self.tau  # alpha per axis
        dt = float(self.dt)

        # Build per-axis 3x3 Singer F and Q, then block-diag them
        F_blocks = []
        Q_blocks = []
        for ax in range(self.dim):
            F_blocks.append(_singer_F(dt, alpha[ax]))
            Q_blocks.append(_singer_Q(dt, alpha[ax], sig_a[ax]))

        self.kf.F = _blk_diag(F_blocks)
        self.kf.Q = _blk_diag(Q_blocks)

        # Measurement matrix H: measure position in each axis
        if self.dim == 2:
            H = np.array([
                [1, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 1, 0, 0],  # y
            ], float)
        else:
            H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 1, 0, 0, 0, 0, 0],  # y
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  # z
            ], float)
        self.kf.H = H

        # Observation covariance R (diagonal with per-axis variances)
        self.kf.R = np.diag((self.initialization_data.observation_noise_std ** 2).astype(float))

        # If user did not pass initial x/P in Base.initialize, keep FilterPy defaults.

    def _get_x0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        estimated_v0 = (measurements[1] - measurements[0]) / self.dt
        if self.dim == 2:
            # this means that x0 should be of the form [x, v_x, a_x y, v_y, a_y]
            x0 = np.array([measurements[0, 0], estimated_v0[0], 0, measurements[0, 1], estimated_v0[1], 0])
        else:
            # then dim==3, so x0 should be of the form [x, v_x, a_x y, v_y, a_y, z, v_z a_z]
            x0 = np.array([measurements[0, 0], estimated_v0[0], 0,
                           measurements[0, 1], estimated_v0[1], 0,
                           measurements[0, 2], estimated_v0[2], 0])
        return x0

    def _get_P0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        return np.eye(self.dim * 3) * 1e1

    def _project_Q(self, raw_Q):
        """
        Project Q to the Singer model theoretical form.
        Find the best scalar multiplier that fits the empirical Q to the theoretical structure.
        """
        q_bounds = (1e-10, 1e+1)
        tie_axes = True

        Q = 0.5 * (raw_Q + raw_Q.T)  # Make symmetric
        n = 3 * self.dim
        dt = self.dt

        Q_projected = np.zeros((n, n), dtype=float)
        q_values = []

        for axis in range(self.dim):
            # Extract the 3x3 block for this axis
            i = 3 * axis
            Q_axis = Q[i:i + 3, i:i + 3]

            # Get the theoretical Singer structure for this axis (without scaling)
            tau_axis = self.tau[axis]
            alpha_axis = 1.0 / tau_axis
            B_axis = _singer_Q(dt, alpha_axis, 1.0)

            # Find the best scalar multiplier using least-squares fitting
            b_flat = B_axis.reshape(-1)
            denom = float(b_flat @ b_flat)
            q_axis = float((Q_axis.reshape(-1) @ b_flat) / denom)
            q_axis = float(np.clip(q_axis, *q_bounds))
            q_values.append(q_axis)

            Q_projected[i:i + 3, i:i + 3] = q_axis * B_axis

        # Tie axes together using median q value for consistency
        if tie_axes and len(q_values) > 1:
            q_median = float(np.clip(np.median(q_values), *q_bounds))
            for axis in range(self.dim):
                i = 3 * axis
                tau_axis = self.tau[axis]
                alpha_axis = 1.0 / tau_axis
                B_axis = _singer_Q(dt, alpha_axis, 1.0)
                Q_projected[i:i + 3, i:i + 3] = q_median * B_axis

        return Q_projected


def _blk_diag(mats):
    """Block-diagonal concatenation for a list of square matrices."""
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n), dtype=float)
    i = 0
    for M in mats:
        k = M.shape[0]
        out[i:i + k, i:i + k] = M
        i += k
    return out


def _singer_F(dt: float, alpha: float) -> np.ndarray:
    """
    Closed-form Singer state transition per axis for state [x, v, a]^T.
    Discrete-time from Singer (1970), Eq. (13) — with numerically stable small-x handling.
    """
    x = alpha * dt
    if x < 1e-6:
        # Series expansion around x=0 is a good approximation
        e = 1 - x + 0.5 * x * x - (x ** 3) / 6.0
        F = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0 - x + 0.5 * x * x]  # e^{-x}
        ], float)
        return F

    e = np.exp(-x)
    # Singer’s D(T, alpha) for [x, v, a]
    F = np.array([
        [1.0, dt, (1.0 - e) / alpha],
        [0.0, 1.0, e - 1.0 + x],
        [0.0, 0.0, e]
    ], float)
    return F


def _singer_Q(dt: float, alpha: float, sigma_a: float) -> np.ndarray:
    """
    Compute Singer model discrete Q matrix using Van Loan's method.

    This is the most reliable approach - uses matrix exponential to convert
    continuous-time system to discrete-time.

    Continuous system:
        dx/dt = Ac*x + Gc*w
        where Ac = [[0, 1, 0], [0, 0, 1], [0, 0, -alpha]]
        and Gc = [[0], [0], [1]]
        and w ~ N(0, 2*alpha*sigma_a^2)  (continuous white noise)
    """
    # Continuous-time system matrices
    Ac = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -alpha]
    ], dtype=float)

    Gc = np.array([[0.0], [0.0], [1.0]], dtype=float)

    # Continuous process noise spectral density
    Qc = 2.0 * alpha * sigma_a ** 2

    # Van Loan method: build augmented matrix
    # M = [[-Ac, Gc*Qc*Gc.T], [0, Ac.T]] * dt
    n = 3
    M = np.zeros((2 * n, 2 * n), dtype=float)
    M[:n, :n] = -Ac * dt
    M[:n, n:] = (Gc @ np.array([[Qc]]) @ Gc.T) * dt
    M[n:, n:] = Ac.T * dt

    # Compute matrix exponential
    EM = expm(M)

    # Extract discrete F and Q
    # F_discrete = exp(Ac * dt) = EM[n:, n:].T
    # Q_discrete = F_discrete @ EM[:n, n:]
    F_discrete = EM[n:, n:].T
    Q_discrete = F_discrete @ EM[:n, n:]

    return Q_discrete
