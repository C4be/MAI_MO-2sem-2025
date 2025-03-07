import numpy as np
from typing import Dict


class Adam:
    """
    Оптимизатор Adam.

    Аргументы:
        learning_rate: скорость обучения (default: 0.01),
        beta1: коэффициент первого момента (default: 0.9),
        beta2: коэффициент второго момента (default: 0.999),
        epsilon: малое число для стабильности (default: 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def update(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
