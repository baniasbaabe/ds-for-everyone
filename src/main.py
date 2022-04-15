import streamlit as st
import pandas as pd

from apps import multipage, data_upload_app

if __name__ == "__main__":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.title("Dat for everyone")

    app = multipage.MultiPage()

    app.add_page("Upload Data", data_upload_app.app)

    app.run()
