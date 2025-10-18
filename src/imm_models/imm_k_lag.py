import numpy as np
from typing import List, Tuple
from filterpy.kalman import IMMEstimator


class IMMKlagEvaluator:
    """
    K-lag evaluation for IMM (Interacting Multiple Model) estimator.
    This class provides k-step ahead prediction capabilities for IMM.
    
    This implementation follows the proper IMM algorithm by:
    1. Using filterpy's built-in mixing probabilities (omega matrix)
    2. Reinitializing each KF with its mixed state (not the total IMM state)
    3. Using each filter's _kstep_predict_state method directly (works for KF and EKF)
    4. Combining predictions using model probabilities
    
    Key insight: The total IMM state should NOT be fed back to individual KFs.
    Instead, each KF should be reinitialized with a mixed state that combines
    information from all models weighted by transition probabilities.
    """
    
    def __init__(self, imm_estimator: IMMEstimator):
        self.imm = imm_estimator
        
    def run_k_lag(self, 
                  measurements: np.ndarray, 
                  k: int = 1,
                  clean_signal: np.ndarray = None,
                  normalize_mse: bool = True) -> Tuple[float, np.ndarray]:
        """
        Runs k-lag prediction for IMM estimator.
        
        Args:
            measurements: Noisy trajectory measurements (T, dim_measurement)
            k: Prediction lag (steps ahead)
            clean_signal: Clean trajectory for MSE calculation (optional)
            normalize_mse: Whether to normalize MSE
            
        Returns:
            Tuple of (MSE, predictions)
        """
        T = measurements.shape[0]
        dim_measurement = measurements.shape[1]
        
        # Initialize IMM with first measurement
        self.imm.update(measurements[0])
        
        predictions = np.empty((T - k, dim_measurement))
        squared_residuals = []
        
        for t in range(T - k):
            # Perform k-step prediction for IMM using proper algorithm
            # The IMM should maintain its own internal state management
            x_k, P_k = self._imm_kstep_predict(k)
            
            # Calculate prediction using measurement matrix
            # For IMM, we need to use the measurement matrix from one of the models
            # (assuming all models have the same H matrix)
            H = self.imm.filters[0].kf.H  # Use H from first model
            yhat_k = (H @ x_k).reshape(-1)
            
            # Calculate residual
            e = measurements[t + k].reshape(-1) - yhat_k
            predictions[t] = yhat_k
            squared_residuals.append(e @ e)
            
            # Update IMM for next iteration using filterpy's methods
            self.imm.predict()
            if t + max(k, 1) < T:
                self.imm.update(measurements[t + 1])
        
        squared_residuals = np.asarray(squared_residuals)
        measurements_for_mse = measurements if clean_signal is None else clean_signal
        
        return self._calculate_mse(measurements_for_mse, squared_residuals, normalized=normalize_mse), predictions
    
    def _imm_kstep_predict(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proper k-step prediction for IMM using filterpy's mixing implementation.
        
        This method leverages filterpy's built-in mixing step and state management:
        - Uses self.imm.omega (mixing probabilities) computed by filterpy's _compute_mixing_probabilities()
        - Uses self.imm.mu (model probabilities) maintained by filterpy
        - Replicates the mixing logic from filterpy's predict() method
        - Each KF operates on its mixed state, not the total IMM state
        """
        # Get current model probabilities and transition matrix
        omega = self.imm.omega.copy()
        mu = self.imm.mu.copy()
        
        # Get individual filter states
        individual_states = [kf.x.copy() for kf in self.imm.filters]
        individual_covariances = [kf.P.copy() for kf in self.imm.filters]
        
        # Perform mixing step
        mixed_states = []
        mixed_covariances = []

        # This code is taken from the IMM.py module from filterpy
        for i, (f, w) in enumerate(zip(self.imm.filters, omega.T)):
            # Mixed state: x_j = sum(omega[i,j] * x_i)
            x_mixed = np.zeros_like(individual_states[0])
            for kf, wj in zip(self.imm.filters, w):
                x_mixed += kf.x * wj
            mixed_states.append(x_mixed)
            
            # Mixed covariance: P_j = sum(omega[i,j] * (P_i + (x_i - x_j)(x_i - x_j)^T))
            P_mixed = np.zeros_like(individual_covariances[0])
            for kf, wj in zip(self.imm.filters, w):
                y = kf.x - x_mixed
                P_mixed += wj * (y @ y.T + kf.P)
            mixed_covariances.append(P_mixed)
        
        # Perform k-step prediction for each model from its mixed state
        x_k_combined = np.zeros_like(individual_states[0])
        P_k_combined = np.zeros_like(individual_covariances[0])
        
        #This follows the logic of the _compute_state_estimate method of the IMMEstimator
        # First pass: compute the combined state estimate
        for i, (kf, prob_i) in enumerate(zip(self.imm.filters, mu)):
            # Perform k-step prediction for each model from its mixed state
            x_k_i, _ = kf._kstep_predict_state(mixed_states[i], mixed_covariances[i], k)
            x_k_combined += prob_i * x_k_i
        
        # Second pass: compute the combined covariance
        for i, (kf, prob_i) in enumerate(zip(self.imm.filters, mu)):
            x_k_i, P_k_i = kf._kstep_predict_state(mixed_states[i], mixed_covariances[i], k)
            diff = x_k_i - x_k_combined
            P_k_combined += prob_i * (P_k_i + diff @ diff.T)
        
        return x_k_combined, P_k_combined
    
    
    @staticmethod
    def _calculate_mse(measurements, squared_residuals, normalized: bool = True):
        """Calculate MSE, optionally normalized."""
        measurement_second_moment = np.mean((measurements - measurements.mean(axis=0, keepdims=True)) ** 2)
        mse = float(np.mean(squared_residuals))
        
        if normalized:
            return mse / measurement_second_moment
        return mse
