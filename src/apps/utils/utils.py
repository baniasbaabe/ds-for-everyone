import os
from typing import Any, List

import pandas as pd


def get_file_path(directory: List) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *directory))


def get_file_type(file: str) -> str:
    try:
        extension = file.name.split(".")[-1]
    except AttributeError:
        extension = file.split(".")[-1]
    return extension


def read_file(path: Any, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        df = pd.read_csv(path)
    elif file_type == "xlsx":
        df = pd.read_xlsx(path)
    return df
