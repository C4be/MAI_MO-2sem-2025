import numpy as np
import pandas as pd
from typing import Tuple, Generator


class DataLoader:
    """
    Класс для загрузки, нормализации данных и генерации батчей.

    Аргументы:
        data: DataFrame с данными,
        labels: Series с метками,
        batch_size: размер батча (default: 32),
        test_split: доля данных для теста (default: 0.2).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        batch_size: int = 32,
        test_split: float = 0.2,
    ):
        self.batch_size = batch_size
        self.test_split = test_split
        self.data = data
        self.labels = labels
        self._prepare_data()

    def _prepare_data(self) -> None:
        # Автоматическая нормализация данных
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        # Разделение на train/test
        split_idx = int(len(self.data) * (1 - self.test_split))
        self.train_data = self.data.iloc[:split_idx].reset_index(drop=True)
        self.train_labels = self.labels.iloc[:split_idx].reset_index(drop=True)
        self.test_data = self.data.iloc[split_idx:].reset_index(drop=True)
        self.test_labels = self.labels.iloc[split_idx:].reset_index(drop=True)

    def get_batches(
        self, train: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if train:
            data, labels = self.train_data, self.train_labels
        else:
            data, labels = self.test_data, self.test_labels

        for i in range(0, len(data), self.batch_size):
            batch_data = data.iloc[i : i + self.batch_size].values
            batch_labels = labels.iloc[i : i + self.batch_size].values
            yield batch_data, batch_labels
