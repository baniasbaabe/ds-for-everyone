import os

import pandas as pd
import streamlit as st

from apps import data_upload_app, configurations_app, eda_app, multipage

if __name__ == "__main__":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.title("Data Science for everyone")

    app = multipage.MultiPage()

    app.add_page("Upload Data", data_upload_app.app)

    app.add_page("Change Data Types", configurations_app.app)

    app.add_page("EDA", eda_app.app)

    app.run()