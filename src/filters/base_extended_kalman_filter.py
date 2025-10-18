import numpy as np

from src.filters.base_kalman_filter import BaseKalmanFilter


class BaseExtendedKalmanFilter(BaseKalmanFilter):
    """
    Extended Kalman Filter base class.
    Inherit this class for any nonlinear motion model (e.g., Coordinated Turn).

    Subclasses must implement:
        f(self, x):              # nonlinear state transition function
        linearize_F(self, x):    # Jacobian ∂f/∂x
    Optionally:
        linearize_Q(self, x):    # recompute Q if state-dependent (e.g., CT)
    """

    def f(self, x):
        """
        Nonlinear transition function.
        Subclasses MUST override.
        """
        raise NotImplementedError("Subclass must implement f(x)")

    def linearize_F(self, x):
        """
        Jacobian of f with respect to x.
        Subclasses MUST override.
        """
        raise NotImplementedError("Subclass must implement linearize_F(x)")

    def get_current_Q(self, x):
        """
        Optional state-dependent process noise changing over time.
        Subclasses MAY override.
        Default: use self.kf.Q unchanged.
        """
        return self.kf.Q

    def predict(self, u=None):
        """
        EKF predict step.
        Performs nonlinear state propagation and covariance linearization.
        """
        x = self.kf.x  # in the code, x represents \hat{x}_{k-1|k-1}
        P = self.kf.P  # in the code, P represents P_{k-1|k-1}

        # This calculation follows the formulas introduced in page 3 of the summary
        x_pred = self.f(x)  # in the code, x_pred represents \hat{x}_{k|k-1}

        # Linearizing F and Q. No need to linearize H, it is already linear
        F_k = self.linearize_F(x)
        Q_current = self.get_current_Q(x)  # Q_current is actually Q_{k-1} in the equations

        # EKF covariance update
        P_pred = F_k @ P @ F_k.T + Q_current  # in the code, P_pred represents P_{k|k-1}

        # update internal KF state
        self.kf.x = x_pred
        self.kf.P = P_pred
        self.kf.F = F_k
        self.kf.Q = Q_current
        # the update step stays identical to linear KF

    def _kstep_predict_state(self,
                             x_t_t: np.ndarray,
                             P_t_t: np.ndarray,
                             k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Multi-step (k-step) prediction for Extended Kalman Filters.

        This performs recursive nonlinear propagation:
          x_{t+i+1|t} = f(x_{t+i|t})
          P_{t+i+1|t} = F_i P_{t+i|t} F_i^T + Q_i
        where F_i and Q_i are relinearized at each step.

        Parameters
        ----------
        x_t_t : np.ndarray
            Current estimated state (x_{t|t}).
        P_t_t : np.ndarray
            Current state covariance (P_{t|t}).
        k : int
            Number of prediction steps ahead.

        Returns
        -------
        x_k : np.ndarray
            Predicted mean k steps ahead (x_{t+k|t}).
        P_k : np.ndarray
            Predicted covariance k steps ahead (P_{t+k|t}).
        """
        x_pred = x_t_t.copy()
        P_pred = P_t_t.copy()

        for _ in range(k):
            # Nonlinear propagation
            F = self.linearize_F(x_pred)
            Q = self.get_current_Q(x_pred)
            x_pred = self.f(x_pred)
            P_pred = F @ P_pred @ F.T + Q

        return x_pred, P_pred
