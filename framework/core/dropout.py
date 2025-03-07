import numpy as np
from .layer import Layer


class Dropout(Layer):
    """
    Слой Dropout для регуляризации.

    Аргументы:
        drop_rate: вероятность отбрасывания нейрона (например, 0.5).
    """

    def __init__(self, drop_rate: float = 0.5):
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.mask = (np.random.rand(*inputs.shape) > self.drop_rate) / (
            1.0 - self.drop_rate
        )
        return inputs * self.mask

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        return grad * self.mask
