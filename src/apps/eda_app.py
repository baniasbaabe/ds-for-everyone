import os

import numpy as np
import pandas as pd
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

from .utils.utils import get_file_path


def app():
    st.title("EDA")

    if "uploaded_df" not in st.session_state:
        st.write("You have to upload your data in 'Upload data' section")
    else:
        data = st.session_state["uploaded_df"].copy()

        pr = data.profile_report()

        st_profile_report(pr)