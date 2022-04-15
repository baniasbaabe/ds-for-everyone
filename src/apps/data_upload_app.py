import pandas as pd
import streamlit as st


def app():  # sourcery skip: use-named-expression

    df = None

    st.title("Upload Data")


    uploaded_file = st.file_uploader(label = "Upload your file", type=["csv", "xlsx"])

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_xlsx(uploaded_file)

        df.to_csv("data/df.csv")

