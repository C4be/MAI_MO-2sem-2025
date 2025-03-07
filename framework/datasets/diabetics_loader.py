import pandas as pd
import os
from typing import Tuple

class DiabeticsLoader:
    """
    Загрузчик данных Diabetics из CSV-файла.

    Аргументы:
        path: путь к каталогу с файлом diabetics.csv.
    """

    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(os.path.join(self.path, "diabetics.csv"))
        labels = df.pop("Outcome")  # или укажите корректное имя колонки с метками
        return df, labels
