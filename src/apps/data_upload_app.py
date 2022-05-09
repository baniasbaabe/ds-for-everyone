import os
from turtle import up
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

    df = pd.read_csv(uploaded_file)

    st.session_state["uploaded_df"] = df