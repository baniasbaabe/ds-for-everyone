import os
from typing import List

import pandas as pd
import streamlit as st

from .utils.utils import get_file_path


def app():
    st.title("Configurations")

    if "uploaded_df" not in st.session_state:
        st.write("You have to upload your data in 'Upload data' section")
    else:
        data = st.session_state["uploaded_df"]

        all_columms = data.columns.tolist()

        col1, col2 = st.columns(2)

        global name, type

        name = col1.selectbox("Select Column", all_columms)

        current_type = data.loc[:, data.columns == name].dtypes[0]

        column_options = [
            "category",
            "object",
            "bool",
            "int64",
            "float64",
            "datetime64",
            "timedelta[ns]",
        ]
        current_index = column_options.index(current_type)

        select_dtype = col2.selectbox(
            "Select Column Type", options=column_options, index=current_index
        )

        if st.button("Change Column Type"):
            data[name] = data[name].astype(select_dtype)
            st.session_state["uploaded_df"] = data
