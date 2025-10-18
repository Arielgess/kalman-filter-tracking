import numpy as np

from src.filters.base_kalman_filter import BaseKalmanFilter, InitializationData


class CVKalmanFilter(BaseKalmanFilter):
    """
    The state is [x, v_x, y, v_y] in case of 2-D, and [x, v_x, y, v_y, z, v_z] in case of 3-D
    """

    def initialize(self, x0=None, P0=None):
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

    def _get_x0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        estimated_v0 = (measurements[1] - measurements[0]) / self.dt
        if self.dim == 2:
            # this means that x0 should be of the form [x, v_x, y, v_y]
            x0 = np.array([measurements[0,0], estimated_v0[0], measurements[0,1], estimated_v0[1]])
        else:
            # then dim==3, so x0 should be of the form [x, v_x, y, v_y, z, v_z]
            x0 = np.array([measurements[0,0], estimated_v0[0], measurements[0,1], estimated_v0[1], measurements[0,2], estimated_v0[2]])
        return x0

    def _get_P0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        return np.eye(self.dim * 2) * 1e1

    def _project_Q(self, Q):
        return _sym_psd(Q)

    def _get_motion_model_matrices(self, initialization_data: InitializationData):
        dt = self.dt
        F1 = np.array([
            [1, dt],
            [0, 1]
        ], dtype=float)

        Z = np.zeros((2, 2))

        if self.dim == 2:
            F = np.block([
                [F1, Z],
                [Z, F1]
            ])

            H = np.array([
                [1, 0, 0, 0],  # measure x
                [0, 0, 1, 0],  # measure y
            ], dtype=float)
            """
            Q = np.block([
                [Q1, Z],
                [Z, Q1]
            ])
            """
            Q = np.zeros((4, 4), dtype=float)
            Q[1, 1] = initialization_data.process_noise_std[0]**2
            Q[3, 3] = initialization_data.process_noise_std[1]**2

        else:
            F = np.block([
                [F1, Z, Z],
                [Z, F1, Z],
                [Z, Z, F1]
            ])

            H = np.array([
                [1, 0, 0, 0, 0, 0],  # x
                [0, 0, 1, 0, 0, 0],  # y
                [0, 0, 0, 0, 1, 0],  # z
            ], dtype=float)

            Q = np.zeros((6, 6), dtype=float)
            Q[1, 1] = initialization_data.process_noise_std[0]**2
            Q[3, 3] = initialization_data.process_noise_std[1]**2
            Q[5, 5] = initialization_data.process_noise_std[2]**2

        R = np.diag(initialization_data.observation_noise_std ** 2)
        return F, H, Q, R
