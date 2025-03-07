import pandas as pd
import numpy as np
from typing import Callable, Any


class Dataset:
    """
    Класс для работы с наборами данных с поддержкой метода map.

    Аргументы:
        data: данные (pandas DataFrame или Series, numpy array),
        labels: метки.
    """

    def __init__(self, data: Any, labels: Any):
        self.data = data
        self.labels = labels

    def map(self, func: Callable[[Any], Any]) -> None:
        """
        Применение пользовательской функции к данным.
        Например, one-hot encoding для меток.
        """
        if isinstance(self.data, pd.Series):
            self.data = self.data.apply(func)
        else:
            self.data = np.vectorize(func)(self.data)
