from dataclasses import dataclass
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter


@dataclass
class InitializationData:
    observation_noise_std: np.array
    process_noise_std: Optional[np.array] = None
    white_accel_density: Optional[int] = None
    omega_std: Optional[float] = None


class BaseKalmanFilter:
    """
    This is a base class for the Kalman Filters that will used in this project.
    """

    def __init__(self, dim, dim_state, dim_measurement, dt, initialization_data: InitializationData):
        if dim not in [2, 3]:
            raise ValueError("Supporting only 2-D or 3-D")
        self.dim = int(dim)
        self.dt = float(dt)
        self.dim_state = int(dim_state)
        self.dim_measurement = int(dim_measurement)
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

    import numpy as np

    def _project_Q(self, Q):
        """
        Projects Q to the known form that the Q matrix should be.
        In every model, this method can be overriden in order to project Q differently, based on the modeled motion model.
        This method is used in the EM algorithm after every iteration, in order to keep Q in the best form fit for the model.
        :param Q:
        :return:
        """
        return Q

    def em(self, measurements: np.ndarray, n_iter: int = 10, estimate_R: bool = True, estimate_Q: bool = True,
           step_size: float = 0.1):
        """
        Expectation–Maximization (EM) algorithm to estimate Q and R
        for a linear Gaussian state-space model.
        Args:
            measurements : (T, dim_z) array of observations.
            n_iter       : number of EM iterations.

        Returns:
            dict with the final {'Q','R','x0','P0'}.
        """
        if step_size < 0 or step_size > 1:
            raise ValueError("Step size must be between 0 and 1")

        zs = measurements

        self.kf.x = self._get_x0_from_measurements(measurements)
        self.kf.P = self._get_P0_from_measurements(measurements)

        # we assume that self.kf.initialize was called before this method, so these matrices are already set
        F = self.kf.F.copy()
        H = self.kf.H.copy()
        Q = self.kf.Q.copy()
        R = self.kf.R.copy()

        for _ in range(n_iter):
            # ---------- E-step ----------
            means, covs, means_p, covs_p = self.kf.batch_filter(
                zs, Fs=[F] * len(zs), Qs=[Q] * len(zs), Hs=[H] * len(zs), Rs=[R] * len(zs)
            )
            xs, Ps, Js, Ppred = self.kf.rts_smoother(
                means, covs, Fs=[F] * len(zs), Qs=[Q] * len(zs)
            )

            # pairwise covariances  V_{t,t-1} = P_t J_{t-1}^T
            V_pair = np.zeros_like(Ps)
            for t in range(1, len(zs)):
                V_pair[t] = Ps[t] @ Js[t - 1].T

            T = len(zs)

            # ---------- M-step ----------
            # Update R
            if estimate_R:
                R_new = np.zeros((H.shape[0], H.shape[0]))
                for t in range(T):
                    y_t = zs[t].reshape(-1)
                    residual = y_t - H @ xs[t].reshape(-1)
                    R_new += np.outer(residual, residual) + H @ Ps[t] @ H.T
                R_new /= T

                R_new = 0.5 * (R_new + R_new.T)  # make it symmetric

                R = step_size * R_new + (1 - step_size) * R

            # Update Q
            if estimate_Q:
                Q_new = np.zeros_like(Q)
                for t in range(T - 1):
                    err = xs[t + 1] - F @ xs[t]
                    Q_new += (
                            np.outer(err, err)
                            + F @ Ps[t] @ F.T
                            + Ps[t + 1]
                            - V_pair[t + 1] @ F.T
                            - F @ V_pair[t + 1].T
                    )
                Q_new /= (T - 1)

                Q_new = self._project_Q(Q_new)
                Q = step_size * Q_new + (1 - step_size) * Q

            # Update initial mean/cov
            x0 = xs[0].copy()
            P0 = Ps[0].copy()

        # write back to filter
        self.kf.Q = Q
        self.kf.R = R
        self.kf.x = x0
        self.kf.P = P0

        return {"Q": Q.copy(), "R": R.copy(), "x0": x0.copy(), "P0": P0.copy()}

    def run_k_lag(self,
                  measurements: np.ndarray,
                  X0,
                  P0,
                  k: int = 1,
                  clean_signal: np.ndarray = None,
                  normalize_mse: bool = True):
        """
        Runs the KF in an Online method (feed another observation after every calculation) and then computes the MSE or the NMSE
        """
        self.kf.x = X0.copy()
        self.kf.P = P0.copy()
        self.kf._alpha_sq = 1.  # Set the fading memory control to be 1, just in case. This makes the KF behave normal
        self.kf.M[:] = 0

        T = measurements.shape[0]

        # Ensure all values are proper Python integers
        dim_measurement = int(self.dim_measurement)

        predictions = np.empty((T - k, dim_measurement))
        squared_residuals = []
        for t in range(T - k):
            # calculate x_{t+k|t}, P_{t+k|t}
            x_t_t = self.kf.x.copy()  # x_{t|t}
            P_t_t = self.kf.P.copy()  # P_{t|t}
            x_k, P_k = self._kstep_predict_state(x_t_t, P_t_t, k)

            yhat_k = (self.kf.H @ x_k).reshape(-1)  # calculate yhat_{t+k|t} by H @ x_{t+k|t}
            # calculate the residual - e = measurements[t++k] - yhat_{t+k|t}
            e = measurements[t + k].reshape(-1) - yhat_k
            predictions[t] = yhat_k
            squared_residuals.append(e @ e)
            self.predict()
            if t + max(k, 1) < T:  # the max is for the case that k=0
                self.update(measurements[t + 1])

        squared_residuals = np.asarray(squared_residuals)
        measurements_for_mse = measurements if clean_signal is None else clean_signal
        return self.calculate_mse(measurements_for_mse[k:], predictions, normalized=normalize_mse), predictions

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

    def evaluate_on_trajectory(
            self,
            noisy_trajectory: np.ndarray,
            clean_trajectory: np.ndarray = None,
            k: int = 1,
            normalize_mse: bool = True,
            return_predictions: bool = False
    ):
        """
        This method is used for running the kalman filter on the trajectory.
        By using this method (as opposed to the run_k_lag method), we are able to initialize x0 and P0
        from the measurements, inside the kalman filter. This gives the ability to use this method for out-of-the-box
        initialization.
        :param noisy_trajectory: an array containing the noisy measurements. shape: (T, self.dim)
        :param clean_trajectory: an array containing the clean measurements, if available. shape: (T, self.dim)
        :return: the MSE of the trajectory. if the clean trajectory was provided, the MSE will be calculated against it.
        """
        x0 = self._get_x0_from_measurements(noisy_trajectory)
        P0 = self._get_P0_from_measurements(noisy_trajectory)
        mse, predictions = self.run_k_lag(
            measurements=noisy_trajectory,
            X0=x0,
            P0=P0,
            k=k,
            clean_signal=clean_trajectory,
            normalize_mse=normalize_mse
        )
        if return_predictions:
            return mse, predictions
        return mse

    def _get_x0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _get_P0_from_measurements(self, measurements: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def calculate_mse(measurements, predictions, normalized: bool = True):
        measurement_second_moment = np.mean((measurements - measurements.mean(axis=0, keepdims=True)) ** 2)

        mse = float(np.mean((measurements - predictions)**2))

        if normalized:
            return mse / measurement_second_moment
        return mse
