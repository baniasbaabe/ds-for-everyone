import os

import pandas as pd
import streamlit as st

from apps import configurations_app, data_upload_app, eda_app, ml_app, multipage

if __name__ == "__main__":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.title("Data Science for everyone")

    app = multipage.MultiPage()

    app.add_page("Upload Data", data_upload_app.app)

    app.add_page("Change Data Types", configurations_app.app)

    app.add_page("EDA", eda_app.app)

    app.add_page("Machine Learning", ml_app.app)

    app.run()
