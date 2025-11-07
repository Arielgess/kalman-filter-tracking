import numpy as np

from src.filters.base_extended_kalman_filter import BaseExtendedKalmanFilter
from src.filters.base_kalman_filter import InitializationData


def f_ct(state: np.ndarray, dt: float) -> np.ndarray:
    """
    Continuous-Turn (CT) nonlinear transition function.
    State: [x, y, v_x, v_y, omega]

    The turn rate omega is assumed constant over the time step dt.
    Positions and velocities are rotated according to omega * dt.

    This method follows the nonlinear dynamic equation introduced in the example in notebook 3.
    """
    x, y, vx, vy, omega = state
    omega_time = omega * dt
    # Near-zero omega → straight-line motion (avoid numerical instability)
    if abs(omega_time) < 1e-6:
        return np.array([x + vx * dt,
                         y + vy * dt,
                         vx,
                         vy,
                         omega])

    c, s = np.cos(omega_time), np.sin(omega_time)
    A = np.sin(omega_time) / omega
    B = (1.0 - np.cos(omega_time)) / omega

    x_next = x + A * vx + B * vy
    y_next = y - B * vx + A * vy
    vx_next = c * vx - s * vy
    vy_next = s * vx + c * vy

    return np.array([x_next, y_next, vx_next, vy_next, omega])


def F_ct(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of the CT motion model: ∂f/∂x.
    State: [x, y, v_x, v_y, ω]

    Derived analytically from f_ct().
    This follows the F_k matrix introduced in the example in notebook 3
    """
    _, _, vx, vy, omega = x
    omega_time = omega * dt
    c, s = np.cos(omega_time), np.sin(omega_time)

    # Handle small omega_time via Taylor expansion. This is used for the terms not to explode
    if abs(omega_time) < 1e-6:
        A = dt
        B = 0.5 * dt ** 2
        Ap = -dt ** 3 / 3 * omega
        Bp = 0.5 * dt ** 2
    else:
        A = np.sin(omega_time) / omega
        B = (1.0 - np.cos(omega_time)) / omega
        Ap = (omega_time * np.cos(omega_time) - np.sin(omega_time)) / (omega ** 2 + 1e-300)
        Bp = (omega_time * np.sin(omega_time) - (1.0 - np.cos(omega_time))) / (omega ** 2 + 1e-300)

    F = np.eye(5, dtype=float)
    F[0, 2] = A
    F[0, 3] = B
    F[0, 4] = vx * Ap + vy * Bp
    F[1, 2] = -B
    F[1, 3] = A
    F[1, 4] = -vx * Bp + vy * Ap
    F[2, 2] = c
    F[2, 3] = -s
    F[2, 4] = -dt * (s * vx + c * vy)
    F[3, 2] = s
    F[3, 3] = c
    F[3, 4] = dt * (c * vx - s * vy)
    return F


def build_Q_ct_bla(omega, dt: float, q_acc: float, q_omega: float) -> np.ndarray:
    """
    Discrete-time process noise for the CT model.
    The omega is not used in this implementation, but can be used if we want a Q matrix that changes over time.
    In our case, it is sufficient to assume that Q is constant
    Q shape: (5,5)
    This Q only represents the process noise of the CT model. In 3-D, the z-axis is CA model,
    so this matrix will be expanded to support the CA model noise in z axis if necessary
    """
    # Small, constant approximate discretization
    qv = q_acc * np.array([
        [dt ** 4 / 4, dt ** 3 / 2],
        [dt ** 3 / 2, dt ** 2]
    ])
    Q = np.zeros((5, 5))
    Q[0:2, 0:2] = qv
    Q[2:4, 2:4] = qv
    Q[4, 4] = q_omega * dt
    return Q

def build_Q_ct(omega: float, dt: float, q_acc: float, q_omega: float) -> np.ndarray:
    """
    Build the discrete-time process noise covariance Q for the Coordinated Turn (CT) model.

    Parameters
    ----------
    omega : float
        Turn rate [rad/s].
    dt : float
        Sampling time [s].
    q_acc : float
        Power spectral density (PSD) of the white acceleration noise [m²/s³].
    q_omega : float
        Power spectral density of the angular rate noise [rad²/s].

    Returns
    -------
    Q : np.ndarray, shape (5,5)
        Discrete process noise covariance for state [x, y, v_x, v_y, ω].

    Notes
    -----
    This is the ω-dependent form of Q, consistent with the continuous-time dynamics:

        ẋ = v_x
        ẏ = v_y
        v̇_x = -ω v_y + a_x
        v̇_y =  ω v_x + a_y
        ω̇ = w_ω

    It assumes:
        - Acceleration noise PSD = q_acc (same in x/y)
        - Angular acceleration noise PSD = q_omega
    """

    # Handle near-zero omega for numerical stability
    omega_dt = omega * dt
    if abs(omega_dt) < 1e-6:
        # When omega ≈ 0 → straight-line motion (use CV-like approximation)
        qv = q_acc * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])
        Q = np.zeros((5, 5))
        Q[0:2, 0:2] = qv
        Q[2:4, 2:4] = qv
        Q[4, 4] = q_omega * dt
        return Q

    s = np.sin(omega_dt)
    c = np.cos(omega_dt)

    # These are derived from the continuous-time integration of rotated acceleration noise
    # See: Bar-Shalom, "Tracking and Data Association", Sec. 6.5.3
    q11 = (4 - 3 * c - omega_dt * s) / (2 * omega**4)
    q12 = (omega_dt - s) / (2 * omega**3)
    q13 = (1 - c) / omega**2
    q22 = (2 * omega_dt - 4 * s + omega_dt * c) / (2 * omega**4)

    Qv = q_acc * np.array([
        [q11, q12, q13, 0],
        [q12, q22, 0, q13],
        [q13, 0, dt, 0],
        [0, q13, 0, dt]
    ])

    # Assemble full 5×5 Q
    Q = np.zeros((5, 5))
    Q[0:4, 0:4] = Qv
    Q[4, 4] = q_omega * dt
    return Q


class CTKalmanFilter(BaseExtendedKalmanFilter):
    """
    Coordinated Turn (CT) model with IMM-compatible 2D/3D state structure.
    """

    def initialize(self, x0=None, P0=None):
        super().initialize(x0, P0)
        if self.initialization_data.observation_noise_std.shape[0] != self.dim:
            raise ValueError(f"Expected observation_noise_std with shape ({self.dim},)")

        self.kf.H, self.kf.R = self._get_matrices(self.initialization_data)

    def _get_matrices(self, initialization_data: InitializationData):
        """
        Build measurement model matrices H and R for the Coordinated Turn (CT) filter.
        2-D state: [x, v_x, a_x, y, v_y, a_y, ω]
          H shape = (2, 7)
          state = [x, y]

        3-D state: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
          H shape = (3, 10)
          state = [x, y, z]

        R is diagonal observation_noise_std**2.
        """
        if self.dim == 2:
            H = np.zeros((2, 7), dtype=float)
            H[0, 0] = 1.0  # x
            H[1, 3] = 1.0  # y

        else:  # 3-D
            H = np.zeros((3, 10), dtype=float)
            H[0, 0] = 1.0  # x
            H[1, 3] = 1.0  # y
            H[2, 6] = 1.0  # z

        R = np.diag(initialization_data.observation_noise_std ** 2)
        return H, R

    def f(self, x):
        # nonlinear transition for 2D or 3D CT
        dt = self.dt
        if self.dim == 2:
            x_ct = np.array([x[0], x[3], x[1], x[4], x[6]])
            fx = f_ct(x_ct, dt)
            x_new = x.copy()
            x_new[0] = fx[0];
            x_new[3] = fx[1]
            x_new[1] = fx[2];
            x_new[4] = fx[3]
            x_new[6] = fx[4]
            return x_new
        else:
            # 3D: XY follows CT, Z follows CA
            x_ct = np.array([x[0], x[3], x[1], x[4], x[9]])
            fx_ct = f_ct(x_ct, dt)
            Fz = np.array([[1, dt, 0.5 * dt ** 2],
                           [0, 1, dt],
                           [0, 0, 1]])
            x_new = x.copy()
            x_new[0] = fx_ct[0];
            x_new[3] = fx_ct[1]
            x_new[1] = fx_ct[2];
            x_new[4] = fx_ct[3]
            x_new[9] = fx_ct[4]
            x_new[6:9] = Fz @ x[6:9]
            return x_new

    def linearize_F(self, x):
        dt = self.dt
        if self.dim == 2:
            x_ct = np.array([x[0], x[3], x[1], x[4], x[6]])
            F_ct_local = F_ct(x_ct, dt)
            F = np.eye(7)
            F[np.ix_([0, 3, 1, 4, 6], [0, 3, 1, 4, 6])] = F_ct_local
            return F
        else:
            x_ct = np.array([x[0], x[3], x[1], x[4], x[9]])
            F_ct_local = F_ct(x_ct, dt)
            Fz = np.array([[1, dt, 0.5 * dt ** 2],
                           [0, 1, dt],
                           [0, 0, 1]])
            F = np.eye(10)
            F[np.ix_([0, 3, 1, 4, 9], [0, 3, 1, 4, 9])] = F_ct_local
            F[6:9, 6:9] = Fz
            return F

    def get_current_Q(self, x):
        dt = self.dt
        if self.dim == 2:
            q_acc = float(self.initialization_data.process_noise_std[0]) ** 2
            q_omega = float(
                self.initialization_data.omega_std) ** 2

            # Build the 5×5 Q for the CT core
            Q_ct_core = build_Q_ct(x[6], dt, q_acc, q_omega)  # (5×5) for [x, y, vx, vy, ω]

            # Embed it in a 7×7 matrix for [x, vx, ax, y, vy, ay, ω]
            Q_full = np.zeros((7, 7), dtype=float)
            # Same index mapping you use for F:
            idx = [0, 3, 1, 4, 6]  # [x, y, vx, vy, ω]
            Q_full[np.ix_(idx, idx)] = Q_ct_core

            return Q_full

        else:
            # 3-D
            q_acc_xy = float(self.initialization_data.process_noise_std[0]) ** 2
            q_acc_z = float(self.initialization_data.process_noise_std[2]) ** 2
            q_omega = float(
                self.initialization_data.omega_std) ** 2

            Q_ct_xy = build_Q_ct(x[9], dt, q_acc_xy, q_omega)

            # Add to it the Q of the constant acceleration model
            Qz = q_acc_z * np.array([
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt]
            ], dtype=float)

            Q = np.zeros((10, 10), dtype=float)
            Q[np.ix_([0, 3, 1, 4, 9], [0, 3, 1, 4, 9])] = Q_ct_xy
            Q[6:9, 6:9] = Qz

            return Q

    def _get_x0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        estimated_v0 = (measurements[1] - measurements[0]) / self.dt
        if self.dim == 2:
            # this means that x0 should be of the form [x, v_x, a_x, y, v_y, a_y, ω]
            x0 = np.array([measurements[0, 0], estimated_v0[0], 0, measurements[0, 1], estimated_v0[1], 0, 0])
        else:
            # then dim==3, so x0 should be of the form [x, v_x, a_x y, v_y, a_y, z, v_z a_z]
            x0 = np.array([measurements[0, 0], estimated_v0[0], 0,
                           measurements[0, 1], estimated_v0[1], 0,
                           measurements[0, 2], estimated_v0[2], 0, 0])
        return x0

    def _get_P0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        return np.eye(self.dim * 3 + 1) * 1e1
