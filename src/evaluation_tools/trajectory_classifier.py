from typing import List

import numpy as np

from src.filters.base_kalman_filter import BaseKalmanFilter


def classify_confusion_matrix(filters: List[BaseKalmanFilter], trajectories: List[tuple[np.ndarray, np.ndarray]]):
    if len(filters) != len(trajectories):
        raise ValueError('The length of `filters` and `trajectories` must be the same')
    confusion_matrix = np.zeros((len(filters), len(filters)))

    for i, trajectories in enumerate(trajectories):  # trajectories is a list that is related for some kf
        for noisy_traj, clean_traj in trajectories:
            mses = np.array([kf.evaluate_on_trajectory(
                noisy_trajectory=noisy_traj,
                clean_trajectory=clean_traj,
                k=1,
                normalize_mse=False,
            ) for kf in filters])
            classification = np.argmin(mses)
            confusion_matrix[i, classification] += 1

    return confusion_matrix
