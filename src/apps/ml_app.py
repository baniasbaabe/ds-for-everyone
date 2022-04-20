import os
import pandas as pd
import numpy as np
import streamlit as st
from .utils.utils import get_file_path

def app():
    if "uploaded_file.csv" not in os.listdir(get_file_path(["..", "..", "data"])):
        st.write("You have to upload your data in 'Upload data' section")
    else:
        data = pd.read_csv(get_file_path(["..", "..", "data", "uploaded_file.csv"]))

        st.write("Select the variable you want to be predicted (Y)")
        target = st.selectbox(
            "Select the variable you want to be predicted (Y)",
            tuple(list(data.columns)))
        
        st.write("You selected ", target)

        type_of_ml_problem = st.radio(
            "Select the type of Machine Learning Problem you want to have solved",
            ("Classification", "Regression")
        )

        if type_of_ml_problem == "Classification":
            pass
        else:
            pass
        
        st.write("Do you want to drop highly correlated numerical features?")
        agree_var_threshold = st.checkbox("Yes")
        if agree_var_threshold:
            cor_matrix = data.select_dtypes(include = np.number).corr().abs()
            min_variance_to_drop = st.slider("Variance", 0.0, 0.99, 0.01)
            len_before = len(data.columns)
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > min_variance_to_drop)]
            
            if st.button("Drop"):
                data = data.drop(to_drop, axis = 1)

                st.write(f"{len_before - len(data.columns)} columns dropped")
                st.write(data.columns)







