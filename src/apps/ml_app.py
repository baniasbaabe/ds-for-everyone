import os
import optuna
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from .utils.utils import get_file_path
from .preprocessing import feature_drop, preprocess_pipeline
from .optimizer import optimize_classification, optimize_regression

optimizer_collection = {
    "Classification": optimize_classification.objective,
    # "Regression": optimize_regression.objective,
}

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

        optimizer = optimizer_collection[type_of_ml_problem]
        
        if list(data.select_dtypes(include = np.number).columns):
            st.write("Do you want to drop highly correlated numerical features?")
            agree_min_corr = st.checkbox("Yes")
            if agree_min_corr:
                min_corr = st.slider("Minimum Correlation to drop feature", 0.0, 0.99, 0.01)
                corr_matrix = feature_drop.calc_corr_matrix(data)
                columns_to_drop = feature_drop.drop_high_corr_feature(corr_matrix, min_corr)

                if st.button("Drop features"):
                    len_before_drop = len(data.columns)
                    data_dropped = data.drop(columns_to_drop, axis = 1).copy()
                    len_after_drop = len(data_dropped.columns)

                    st.write(f"Columns dropped: {len_before_drop - len_after_drop}")
        
        if st.button("Run Machine Learning Algorithms"):
            y = data.loc[:, target].copy()
            X = data.loc[:, data.columns != target].copy()

            le = LabelEncoder()
            y = le.fit_transform(y)

            study = optuna.create_study(direction = "minimize")
            func = lambda trial: optimizer(trial, X, y, preprocess_pipeline.full_pipeline)
            study.optimize(func, n_trials = 3, show_progress_bar = False)

            print(study.best_params)