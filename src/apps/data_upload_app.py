import os
from typing import Any, List

import pandas as pd
import streamlit as st

from .utils.utils import get_file_path, get_file_type, read_file


def app():
    df = None

    st.title("Upload Data")

    uploaded_file = st.file_uploader(label="Upload your file", type=["csv", "xlsx"])

    if not uploaded_file:
        return

    file_type = get_file_type(uploaded_file)
    df = read_file(uploaded_file, file_type)

    path = get_file_path(["..", "..", "data", "uploaded_file.csv"])
    df.to_csv(path, index = False)