"""
Singleton random number generator for consistent randomization across the codebase.

Usage:
    from src.dataset.random_gen import get_rng, set_seed
    
    # Set seed once at the beginning
    set_seed(42)
    
    # Get RNG instance anywhere
    rng = get_rng()
    value = rng.uniform(0, 1)
"""

import numpy as np
from typing import Optional


class RandomGeneratorSingleton:
    """Singleton class for managing a single random number generator instance"""
    
    _rng: Optional[np.random.Generator] = None
    _seed: Optional[int] = 42
    
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        
    def get_rng(self) -> np.random.Generator:
        """Get the random number generator instance"""
        return self._rng
    
    def get_seed(self) -> Optional[int]:
        """Get the current seed"""
        return self._seed

    def reinitialize(self):
        self._rng = np.random.default_rng(self._seed)


random_generator = RandomGeneratorSingleton()
