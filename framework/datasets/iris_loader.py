import pandas as pd
import os
from typing import Tuple


class IrisLoader:
    """
    Загрузчик данных Iris из CSV-файла.

    Аргументы:
        path: путь к каталогу с файлом iris.csv.
    """

    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(os.path.join(self.path, "iris.csv"))
        labels = df.pop("species")
        return df, labels
