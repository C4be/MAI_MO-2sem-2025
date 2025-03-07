from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Базовый класс для слоев нейронной сети.
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Вычисление выхода слоя.
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Обратное распространение ошибки: вычисление градиента и обновление параметров.
        """
        pass
