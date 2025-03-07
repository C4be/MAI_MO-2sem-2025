import numpy as np
from typing import Dict


class SGD:
    """
    Стандартный стохастический градиентный спуск (без импульса).

    Аргументы:
        learning_rate: скорость обучения (default: 0.01).
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        """
        Обновление параметров по правилу: param = param - learning_rate * grad.
        """
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]
