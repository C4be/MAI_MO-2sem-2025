import numpy as np
from typing import Dict


class MomentumSGD:
    """
    Стохастический градиентный спуск с импульсом.

    Аргументы:
        learning_rate: скорость обучения (default: 0.01),
        momentum: коэффициент импульса (default: 0.9).
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: Dict[str, np.ndarray] = {}

    def update(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(grads[key])
            self.velocity[key] = (
                self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            )
            params[key] += self.velocity[key]
