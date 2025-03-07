import numpy as np
import os
from typing import Tuple


class MNISTLoader:
    """
    Загрузчик данных MNIST из бинарных файлов.

    Аргументы:
        path: путь к каталогу с файлами MNIST.
    """

    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Пример загрузки изображений
        with open(os.path.join(self.path, "train-images.idx3-ubyte"), "rb") as f:
            f.read(16)  # Пропускаем заголовок
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
        with open(os.path.join(self.path, "train-labels.idx1-ubyte"), "rb") as f:
            f.read(8)  # Пропускаем заголовок
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        # Нормализация: масштабирование пикселей в [0, 1]
        images = images.astype("float32") / 255.0
        return images, labels
