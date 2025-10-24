"""
Kalman Filter Package

This package contains various implementations of Kalman filters for tracking applications.
"""

# Base classes
from .base_kalman_filter import BaseKalmanFilter, InitializationData
from .base_extended_kalman_filter import BaseExtendedKalmanFilter

# Specific filter implementations
from .cv_kalman_filter import CVKalmanFilter
from .ca_kalman_filter import CAKalmanFilter
from .ct_kalman_filter import CTKalmanFilter
from .singer_model_kalman_filter import SingerKF

__all__ = [
    # Base classes
    'BaseKalmanFilter',
    'BaseExtendedKalmanFilter',
    'InitializationData',
    
    # Filter implementations
    'CVKalmanFilter',
    'CAKalmanFilter', 
    'CTKalmanFilter',
    'SingerKF',
]
