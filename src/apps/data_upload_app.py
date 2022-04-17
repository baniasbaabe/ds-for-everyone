import os
from typing import Any, List

import pandas as pd
import streamlit as st


def app():
    df = None

    st.title("Upload Data")

    uploaded_file = st.file_uploader(label="Upload your file", type=["csv", "xlsx"])

    if not uploaded_file:
        return

    file_type = get_file_type(uploaded_file)
    df = read_file(uploaded_file, file_type)

    path = get_file_path(["..", "data", "uploaded_file.csv"])
    df.to_csv(path, index = False)

def read_file(path: Any, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        df = pd.read_csv(path)
    elif file_type == "xlsx":
        df = pd.read_xlsx(path)
    return df

def get_file_type(file: str) -> str:
    try:
        extension = file.name.split(".")[-1]
    except AttributeError:
        extension = file.split(".")[-1]
    return extension

def get_file_path(directory: List) -> str:
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), *directory))