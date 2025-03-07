import numpy as np
from typing import Dict


class RMSProp:
    """
    Оптимизатор RMSProp.

    Аргументы:
        learning_rate: скорость обучения (default: 0.01),
        epsilon: малое число для стабильности (default: 1e-8),
        decay_rate: коэффициент затухания (default: 0.9).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        decay_rate: float = 0.9,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.cache: Dict[str, np.ndarray] = {}

    def update(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        for key in params.keys():
            if key not in self.cache:
                self.cache[key] = np.zeros_like(grads[key])
            self.cache[key] = self.decay_rate * self.cache[key] + (
                1 - self.decay_rate
            ) * (grads[key] ** 2)
            params[key] -= (
                self.learning_rate
                * grads[key]
                / (np.sqrt(self.cache[key]) + self.epsilon)
            )
