
import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator, ExtendedKalmanFilter, UnscentedKalmanFilter
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.filters.base_extended_kalman_filter import BaseExtendedKalmanFilter

# State dimension constants for IMM compatibility
STATE_DIM_2D = 7   # [x, v_x, a_x, y, v_y, a_y, ω]
STATE_DIM_3D = 10  # [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]

@dataclass
class InitializationData:
    observation_noise_std: np.array
    process_noise_std: np.array
    white_accel_density: Optional[int] = None
    tau: Optional[float] = None
    omega_std: Optional[float] = None


#Matrices for choosing measurements from the state (updated for new state dimension)
# State: [x, v_x, a_x, y, v_y, a_y, ω] for 2D or [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω] for 3D
X = np.array([1, 0, 0, 0, 0, 0, 0])  # 2D: measure x
Y = np.array([0, 0, 0, 1, 0, 0, 0])  # 2D: measure y
Z = np.array([0, 0, 0, 0, 0, 0, 0])  # 2D: no z measurement

class BaseKalmanFilter:
    """
    This is an interface for the Kalman Filters that will used in this project.
    They adhere to the interface the IMM in the filterpy package expects, so they are able to be used in the IMM model
    """
    def __init__(self, dim, dim_state, dim_measurement, dt, initialization_data: InitializationData):
        if dim not in [2, 3]:
            raise ValueError("Supporting only 2-D or 3-D")
        self.dim = dim
        self.dt = dt
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.kf = None
        self.initialization_data = initialization_data

    def initialize(self, x0=None, P0=None):
        """
        This should always be overriden, becaues the KalmanFilter given by filterpy is not a functional
        KF without changing the matrices first.
        :return:
        """
        self.kf = KalmanFilter(dim_x=self.dim_state, dim_z=self.dim_measurement)
        if x0 is not None:
            self.kf.x = x0
        if P0 is not None:
            self.kf.P = P0

    def predict(self, u=None):
        """
        This method should be overriden if the KF that is used is doing another steps when predicting.
        For example, a Coordinated-Turn EKF would need to override this method because it needs to linearize f first.
        :return:
        """
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)

    @property
    def x(self):
        return self.kf.x

    @x.setter
    def x(self, value):
        self.kf.x = value

    @property
    def P(self):
        return self.kf.P

    @P.setter
    def P(self, value):
        self.kf.P = value

    @property
    def likelihood(self):
        """
        This should return the likelihood of the last measurement
        :return:
        """
        return self.kf.likelihood

    def run_k_lag(self,
                  measurements: np.ndarray,
                  X0,
                  P0,
                  k: int = 1,
                  clean_signal: np.ndarray=None,
                  normalize_mse: bool = True):
        """
        Runs the KF in an Online method (feed another observation after every calculation) and then computes the MSE or the NMSE
        """
        # Ensure X0 and P0 have the correct dimensions
        if X0.shape[0] != self.dim_state:
            raise ValueError(f"X0 must have {self.dim_state} elements, got {X0.shape[0]}")
        if P0.shape != (self.dim_state, self.dim_state):
            raise ValueError(f"P0 must be ({self.dim_state}, {self.dim_state}), got {P0.shape}")
            
        self.kf.x = X0.copy()
        self.kf.P = P0.copy()
        self.kf._alpha_sq = 1.0 #Set the fading memory control to be 1, just in case. This makes the KF behave normal
        self.kf.M[:] = 0

        T = measurements.shape[0]
        #predict -> update for the first value
        #self.kf.predict()
        self.kf.update(measurements[0])

        predictions = np.empty((T - k, self.dim_measurement))
        squared_residuals = []
        for t in range(T - k):
            #calculate x_{t+k|t}, P_{t+k|t}
            x_t_t = self.kf.x.copy()  # x_{t|t}
            P_t_t = self.kf.P.copy()  # P_{t|t}
            x_k, P_k = self._kstep_predict_state(x_t_t, P_t_t, k)

            yhat_k = (self.kf.H @ x_k).reshape(-1)  # calculate yhat_{t+k|t} by H @ x_{t+k|t}
            #calculate the residual - e = measurements[t++k] - yhat_{t+k|t}
            e = measurements[t + k].reshape(-1) - yhat_k
            predictions[t] = yhat_k
            squared_residuals.append(e@e)
            self.kf.predict()
            if t + max(k, 1) < T: # the max is for the case that k=0
                self.kf.update(measurements[t+1])

        squared_residuals = np.asarray(squared_residuals)
        measurements_for_mse = measurements if clean_signal is None else clean_signal
        return self.calculate_mse(measurements_for_mse, squared_residuals, normalized=normalize_mse), predictions

    def _kstep_predict_state(self,
                            x_t_t: np.ndarray,
                            P_t_t: np.ndarray,
                            k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Pure k-step time update (no measurements):
          x_{t+k|t} = F^k x_{t|t}
          P_{t+k|t} = F^k P_{t|t} (F^T)^k + sum_{i=0}^{k-1} F^i Q (F^T)^i
        """
        n = self.kf.F.shape[0]
        Fk = np.eye(n)
        Fi = np.eye(n)  # will hold F^i
        acc = np.zeros_like(self.kf.Q)

        # build F^k and ∑_{i=0}^{k-1} F^i Q (F^i)^T
        for _ in range(k):
            Fk = self.kf.F @ Fk
            acc = acc + Fi @ self.kf.Q @ Fi.T
            Fi = self.kf.F @ Fi

        x_k = Fk @ x_t_t
        P_k = Fk @ P_t_t @ Fk.T + acc
        return x_k, P_k

    @staticmethod
    def calculate_mse(measurements, squared_residuals, normalized: bool=True):
        measurement_second_moment = np.mean((measurements - measurements.mean(axis=0, keepdims=True)) ** 2)

        mse = float(np.mean(squared_residuals))

        if normalized:
            return mse / measurement_second_moment
        return mse

class IMMConstantVelocityKF(BaseKalmanFilter):
    """
    The state is [x, v_x, a_x, y, v_y, a_y, ω] in case of 2-D, and [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω] in case of 3-D
    For CV model: a_x = a_y = a_z = 0, ω = 0
    """
    def initialize(self, x0 = None, P0 = None):
        super().initialize(x0, P0)
        if self.initialization_data.process_noise_std.shape != (self.dim,):
            raise ValueError("Process noise is not in the correct dimensions")
        if self.initialization_data.observation_noise_std.shape != (self.dim,):
            raise ValueError("Observation noise is not in the correct dimensions")

        F, H, Q, R = self._get_motion_model_matrices(self.initialization_data)
        self.kf.F = F
        self.kf.H = H
        self.kf.Q = Q
        self.kf.R = R

    def _get_motion_model_matrices(self, initialization_data:InitializationData):
        dt = self.dt
        # For CV model: acceleration and omega are always 0
        F1 = np.array([
            [1, dt, 0],  # x, vx, ax
            [0, 1, 0],   # vx stays vx, ax stays 0
            [0, 0, 0]    # ax stays 0
        ])
        
        Z = np.zeros((3,3))
        Z_omega = np.zeros((3,1))  # For omega column
        Z_omega_row = np.zeros((1,3))  # For omega row

        if self.dim == 2:
            F = np.block([
                [F1, Z, Z_omega],
                [Z, F1, Z_omega],
                [Z_omega_row, Z_omega_row, np.array([[1]])]  # omega stays omega (0 for CV)
            ])

            H = np.array([
                [1, 0, 0, 0, 0, 0, 0],  # measure x
                [0, 0, 0, 1, 0, 0, 0],  # measure y
            ])

            Q = np.zeros((STATE_DIM_2D, STATE_DIM_2D), dtype=float)
            Q[1,1] = initialization_data.process_noise_std[0]**2  # velocity noise in x
            Q[4,4] = initialization_data.process_noise_std[1]**2  # velocity noise in y

        else:
            F = np.block([
                [F1, Z, Z, Z_omega],
                [Z, F1, Z, Z_omega],
                [Z, Z, F1, Z_omega],
                [Z_omega_row, Z_omega_row, Z_omega_row, np.array([[1]])]  # omega stays omega (0 for CV)
            ])

            H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # y
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
            ], dtype=float)

            Q = np.zeros((STATE_DIM_3D, STATE_DIM_3D), dtype=float)
            Q[1,1] = initialization_data.process_noise_std[0]**2  # velocity noise in x
            Q[4,4] = initialization_data.process_noise_std[1]**2  # velocity noise in y
            Q[7,7] = initialization_data.process_noise_std[2]**2  # velocity noise in z

        R = np.diag(initialization_data.observation_noise_std**2)
        return F, H, Q, R


class IMMConstantAccelerationKF(BaseKalmanFilter):
    """
    The state is [x, v_x, a_x, y, v_y, a_y, ω] in case of 2-D, and [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω] in case of 3-D
    For CA model: ω = 0
    """
    def initialize(self,x0 = None, P0 = None):
        super().initialize(x0, P0)
        if self.initialization_data.process_noise_std.shape != (self.dim,):
            raise ValueError("Process noise is not in the correct dimensions")
        if self.initialization_data.observation_noise_std.shape != (self.dim,):
            raise ValueError("Observation noise is not in the correct dimensions")

        F, H, Q, R = self._get_motion_model_matrices(self.initialization_data)
        self.kf.F = F
        self.kf.H = H
        self.kf.Q = Q
        self.kf.R = R

    def _get_motion_model_matrices(self, initialization_data:InitializationData):
        dt = self.dt
        F1 = np.array([[1, dt, 0.5 * dt * dt],
                       [0, 1,  dt],
                       [0, 0,   1]], float)
        Z = np.zeros((3,3))
        Z_omega = np.zeros((3,1))  # For omega column
        Z_omega_row = np.zeros((1,3))  # For omega row
        Q_axis = initialization_data.white_accel_density * \
                  np.array([
                  [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                  [dt ** 4 / 8 , dt ** 3 / 3, dt ** 2 / 2],
                  [dt ** 3 / 6 , dt ** 2 / 2, dt         ]
                  ])

        if self.dim == 2:
            F = np.block([
                [F1, Z, Z_omega],
                [Z, F1, Z_omega],
                [Z_omega_row, Z_omega_row, np.array([[1]])]  # omega stays omega (0 for CA)
            ])

            H = np.array([
                [1, 0, 0, 0, 0, 0, 0],  # measure x
                [0, 0, 0, 1, 0, 0, 0],  # measure y
            ])

            Q = np.block([
                [Q_axis, Z, Z_omega],
                [Z, Q_axis, Z_omega],
                [Z_omega_row, Z_omega_row, np.array([[0]])]  # No noise for omega in CA model
            ])

        else:
            F = np.block([
                [F1, Z, Z, Z_omega],
                [Z, F1, Z, Z_omega],
                [Z, Z, F1, Z_omega],
                [Z_omega_row, Z_omega_row, Z_omega_row, np.array([[1]])]  # omega stays omega (0 for CA)
            ])

            H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # y
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
            ], dtype=float)

            Q = np.block([[Q_axis, Z, Z, Z_omega],
                          [Z, Q_axis, Z, Z_omega],
                          [Z, Z, Q_axis, Z_omega],
                          [Z_omega_row, Z_omega_row, Z_omega_row, np.array([[0]])]  # No noise for omega in CA model
            ])

        R = np.diag(initialization_data.observation_noise_std**2)
        return F, H, Q, R


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
        B = 0.5 * dt**2
        Ap = -dt**3 / 3 * omega
        Bp = 0.5 * dt**2
    else:
        A = np.sin(omega_time) / omega
        B = (1.0 - np.cos(omega_time)) / omega
        Ap = (omega_time * np.cos(omega_time) - np.sin(omega_time)) / (omega**2 + 1e-300)
        Bp = (omega_time * np.sin(omega_time) - (1.0 - np.cos(omega_time))) / (omega**2 + 1e-300)

    F = np.eye(5)
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


def build_Q_ct(omega, dt: float, q_acc: float, q_omega: float) -> np.ndarray:
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
        [dt**4/4, dt**3/2],
        [dt**3/2, dt**2]
    ])
    Q = np.zeros((5, 5))
    Q[0:2, 0:2] = qv
    Q[2:4, 2:4] = qv
    Q[4, 4] = q_omega * dt
    return Q


def build_Q_ct_bla(omega: float, dt: float, q_acc: float, q_omega: float) -> np.ndarray:
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


class IMMCoordinatedTurnKF(BaseExtendedKalmanFilter):
    """
    Coordinated Turn (CT) model with IMM-compatible 2D/3D state structure.
    The state is [x, v_x, a_x, y, v_y, a_y, ω] in case of 2-D, and [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω] in case of 3-D
    For CT model: a_x = a_y = 0 (handled by coordinated turn dynamics), a_z = 0 for 3D
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
          measurements = [x, y]

        3-D state: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
          H shape = (3, 10)
          measurements = [x, y, z]

        R is diagonal observation_noise_std**2.
        """
        if self.dim == 2:
            H = np.array([
                [1, 0, 0, 0, 0, 0, 0],  # measure x
                [0, 0, 0, 1, 0, 0, 0],  # measure y
            ])

        else:  # 3-D
            H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measure x
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # measure y
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # measure z
            ], dtype=float)

        R = np.diag(initialization_data.observation_noise_std ** 2)
        return H, R

    def f(self, x):
        # nonlinear transition for 2D or 3D CT
        dt = self.dt
        if self.dim == 2:
            # State: [x, v_x, a_x, y, v_y, a_y, ω]
            # CT core: [x, y, v_x, v_y, ω]
            x_ct = np.array([x[0], x[3], x[1], x[4], x[6]])
            fx = f_ct(x_ct, dt)
            x_new = x.copy()
            x_new[0] = fx[0]  # x
            x_new[3] = fx[1]  # y
            x_new[1] = fx[2]  # v_x
            x_new[4] = fx[3]  # v_y
            x_new[6] = fx[4]  # ω
            # a_x and a_y remain 0 for CT model
            x_new[2] = 0  # a_x
            x_new[5] = 0  # a_y
            return x_new
        else:
            # 3D: State: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
            # XY follows CT, Z follows CA
            x_ct = np.array([x[0], x[3], x[1], x[4], x[9]])
            fx_ct = f_ct(x_ct, dt)
            Fz = np.array([[1, dt, 0.5 * dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
            x_new = x.copy()
            x_new[0] = fx_ct[0]  # x
            x_new[3] = fx_ct[1]  # y
            x_new[1] = fx_ct[2]  # v_x
            x_new[4] = fx_ct[3]  # v_y
            x_new[9] = fx_ct[4]  # ω
            # a_x and a_y remain 0 for CT model
            x_new[2] = 0  # a_x
            x_new[5] = 0  # a_y
            # Z follows CA model
            x_new[6:9] = Fz @ x[6:9]
            return x_new

    def linearize_F(self, x):
        dt = self.dt
        if self.dim == 2:
            # State: [x, v_x, a_x, y, v_y, a_y, ω]
            x_ct = np.array([x[0], x[3], x[1], x[4], x[6]])
            F_ct_local = F_ct(x_ct, dt)
            F = np.eye(STATE_DIM_2D)
            # Map CT Jacobian to full state: [x, y, v_x, v_y, ω] -> [x, v_x, a_x, y, v_y, a_y, ω]
            F[np.ix_([0,3,1,4,6],[0,3,1,4,6])] = F_ct_local
            # a_x and a_y derivatives are 0
            F[2,:] = 0  # a_x doesn't depend on anything
            F[5,:] = 0  # a_y doesn't depend on anything
            return F
        else:
            # State: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
            x_ct = np.array([x[0], x[3], x[1], x[4], x[9]])
            F_ct_local = F_ct(x_ct, dt)
            Fz = np.array([[1, dt, 0.5 * dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
            F = np.eye(STATE_DIM_3D)
            # Map CT Jacobian to full state
            F[np.ix_([0,3,1,4,9],[0,3,1,4,9])] = F_ct_local
            # a_x and a_y derivatives are 0
            F[2,:] = 0  # a_x doesn't depend on anything
            F[5,:] = 0  # a_y doesn't depend on anything
            # Z follows CA model
            F[6:9,6:9] = Fz
            return F

    def get_current_Q(self, x):
        dt = self.dt
        if self.dim == 2:
            # State: [x, v_x, a_x, y, v_y, a_y, ω]
            q_acc = float(self.initialization_data.process_noise_std[0]) ** 2
            q_omega = float(self.initialization_data.omega_std) ** 2

            # Build the 5×5 Q for the CT core [x, y, vx, vy, ω]
            Q_ct_core = build_Q_ct(x[6], dt, q_acc, q_omega)

            # Embed it in a STATE_DIM_2D×STATE_DIM_2D matrix for [x, v_x, a_x, y, v_y, a_y, ω]
            Q_full = np.zeros((STATE_DIM_2D, STATE_DIM_2D))
            # Map CT core indices to full state: [x, y, vx, vy, ω] -> [x, v_x, a_x, y, v_y, a_y, ω]
            # State: [x, v_x, a_x, y, v_y, a_y, ω] -> CT core: [x, y, vx, vy, ω]
            idx = [0, 3, 1, 4, 6]  # [x, y, vx, vy, ω] in full state [x, v_x, a_x, y, v_y, a_y, ω]
            Q_full[np.ix_(idx, idx)] = Q_ct_core
            # a_x and a_y have no process noise (they're always 0)

            return Q_full

        else:
            # State: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
            q_acc_xy = float(self.initialization_data.process_noise_std[0]) ** 2
            q_acc_z = float(self.initialization_data.process_noise_std[2]) ** 2
            q_omega = float(self.initialization_data.omega_std) ** 2

            # CT noise for XY plane
            Q_ct_xy = build_Q_ct(x[9], dt, q_acc_xy, q_omega)

            # CA noise for Z axis
            Qz = q_acc_z * np.array([
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt]
            ])

            Q = np.zeros((STATE_DIM_3D, STATE_DIM_3D))
            # Map CT core indices to full state: [x, y, vx, vy, ω] -> [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
            Q[np.ix_([0, 3, 1, 4, 9], [0, 3, 1, 4, 9])] = Q_ct_xy
            # Z axis noise
            Q[6:9, 6:9] = Qz
            # a_x and a_y have no process noise (they're always 0)

            return Q

    def _get_x0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        estimated_v0 = (measurements[1] - measurements[0]) / self.dt
        if self.dim == 2:
            # State: [x, v_x, a_x, y, v_y, a_y, ω]
            x0 = np.array([measurements[0,0], estimated_v0[0], 0,  # x, v_x, a_x
                          measurements[0,1], estimated_v0[1], 0,  # y, v_y, a_y
                          0])  # ω
        else:
            # State: [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
            x0 = np.array([measurements[0,0], estimated_v0[0], 0,  # x, v_x, a_x
                          measurements[0,1], estimated_v0[1], 0,  # y, v_y, a_y
                          measurements[0,2], estimated_v0[2], 0,  # z, v_z, a_z
                          0])  # ω
        return x0

    def _get_P0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        if self.dim == 2:
            return np.eye(STATE_DIM_2D) * 1e1  # [x, v_x, a_x, y, v_y, a_y, ω]
        else:
            return np.eye(STATE_DIM_3D) * 1e1  # [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z, ω]
