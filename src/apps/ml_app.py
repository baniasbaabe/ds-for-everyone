import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier.classification_report import classification_report
from yellowbrick.classifier.confusion_matrix import confusion_matrix
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.regressor.prediction_error import prediction_error
from yellowbrick.regressor.residuals import residuals_plot

from .optimizer import optimize_classification, optimize_regression
from .preprocessing import feature_drop, preprocess_pipeline
from .utils.utils import get_file_path

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

models_classification = {
    "XGBoost": xgb.XGBClassifier,
    "RandomForest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
}

models_regression = {
    "XGBoost": xgb.XGBRegressor,
    "RandomForest": RandomForestRegressor,
    "AdaBoost": AdaBoostRegressor,
}

optimizer_collection = {
    "Classification": optimize_classification.objective,
    "Regression": optimize_regression.objective,
}


def app():
    if "uploaded_file.csv" not in os.listdir(get_file_path(["..", "..", "data"])):
        st.write("You have to upload your data in 'Upload data' section")
    elif (
        pd.read_csv(get_file_path(["..", "..", "data", "uploaded_file.csv"])).shape[1]
        < 2
    ):
        st.write(
            "Your file doesn't contain 2 columns or more. Upload a new appropiate file."
        )
    else:
        data = pd.read_csv(get_file_path(["..", "..", "data", "uploaded_file.csv"]))

        st.write("Select the variable you want to be predicted (Y)")
        target = st.selectbox(
            "Select the variable you want to be predicted (Y)",
            tuple(list(data.columns)),
        )

        st.write("You selected ", target)

        type_of_ml_problem = st.radio(
            "Select the type of Machine Learning Problem you want to have solved",
            ("Classification", "Regression"),
        )

        optimizer = optimizer_collection[type_of_ml_problem]

        if list(data.select_dtypes(include=np.number).columns):
            st.write("Do you want to drop highly correlated numerical features?")
            agree_min_corr = st.checkbox("Yes")
            if agree_min_corr:
                min_corr = st.slider(
                    "Minimum Correlation to drop feature", 0.0, 0.99, 0.01
                )
                corr_matrix = feature_drop.calc_corr_matrix(data)
                columns_to_drop = feature_drop.drop_high_corr_feature(
                    corr_matrix, min_corr
                )

                if st.button("Drop features"):
                    len_before_drop = len(data.columns)
                    data_dropped = data.drop(columns_to_drop, axis=1).copy()
                    len_after_drop = len(data_dropped.columns)

                    st.write(f"Columns dropped: {len_before_drop - len_after_drop}")

        if st.button("Run Machine Learning Algorithms"):
            y = data.loc[:, target].copy()
            X = data.loc[:, data.columns != target].copy()
            le = LabelEncoder()
            y = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()
            num_cols = X.select_dtypes(include=np.number).columns.to_list()

            study = optuna.create_study(direction="minimize")
            func = lambda trial: optimizer(
                trial, X_train, y_train, preprocess_pipeline.full_pipeline
            )

            with st.spinner("Finding a optimal model, that needs some time..."):
                study.optimize(func, n_trials=4)
            st.write(study.best_params)
            if type_of_ml_problem == "Classification":
                model = models_classification[study.best_params["model_name"]]
            else:
                model = models_regression[study.best_params["model_name"]]

            model = model(
                **{
                    i: study.best_params[i]
                    for i in study.best_params
                    if i != "model_name"
                }
            )

            pipeline = preprocess_pipeline.full_pipeline(cat_cols, num_cols, model)

            pipeline.fit(X_train, y_train)

            if type_of_ml_problem == "Classification":
                with st.spinner("Create plot, that needs some time..."):
                    st.write("Classification Report")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    classification_report(
                        pipeline,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        ax=ax,
                        classes=le.classes_,
                    )
                    st.pyplot(fig)
                with st.spinner("Create plot, that needs some time..."):
                    st.write("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    confusion_matrix(
                        pipeline,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        ax=ax,
                        classes=le.classes_,
                    )
                    st.pyplot(fig)
                with st.spinner("Create plot, that needs some time..."):
                    st.write("ROC")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    is_binary = pd.Series(y).nunique() == 2
                    roc_auc(
                        pipeline,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        ax=ax,
                        classes=le.classes_,
                        binary=is_binary,
                    )
                    st.pyplot(fig)
            else:
                with st.spinner("Create plot, that needs some time..."):
                    st.write("Prediction Error")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    prediction_error(pipeline, X_train, y_train, X_test, y_test, ax=ax)
                    st.pyplot(fig)
                with st.spinner("Create plot, that needs some time..."):
                    st.write("Residuals Plot")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    residuals_plot(pipeline, X_train, y_train, X_test, y_test, ax=ax)
                    st.pyplot(fig)
            # st.write("Interpret Model with SHAP")
            # fig = plt.figure()
            # observations = pipeline["preprocessor"].transform(X_test)
            # shap_values = shap.KernelExplainer(pipeline["model"].predict_proba, observations).shap_values(observations)
            # shap.summary_plot(shap_values, observations, plot_type = "bar")
            # st.pyplot(fig)
