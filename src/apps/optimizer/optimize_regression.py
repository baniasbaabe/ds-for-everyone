import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def objective(trial, X, y, pipeline_func):

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()
    num_cols = X.select_dtypes(include=np.number).columns.to_list()

    classifier_name = trial.suggest_categorical(
        "model_name", ["RandomForest", "AdaBoost", "XGBoost"]
    )

    if classifier_name == "RandomForest":
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 3, 30, step=2),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "n_jobs": -1,
        }

        model = RandomForestRegressor(**param_grid)
    elif classifier_name == "AdaBoost":
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.8, step=0.01),
        }
        model = AdaBoostRegressor(**param_grid)

    elif classifier_name == "XGBoost":
        param_grid = {
            "verbosity": 0,
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
        }

        model = xgb.XGBRegressor(**param_grid)

    n_split = 3
    pipeline = pipeline_func(cat_cols, num_cols, model)

    cv = KFold(n_splits=n_split, shuffle=True)

    cv_scores = np.empty(n_split)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)

        y_preds = pipeline.predict(X_val)

        cv_scores[idx] = mean_squared_error(y_val, y_preds)
        # + std because one Fold could be good and all the others bad
        return np.mean(cv_scores) + np.std(cv_scores)
