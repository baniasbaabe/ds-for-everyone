import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split


def objective(trial, X, y, pipeline_func):

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()
    num_cols = X.select_dtypes(include=np.number).columns.to_list()

    classifier_name = trial.suggest_categorical(
        "classifier_name", ["RandomForest", "AdaBoost", "XGBoost"]
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

        model = RandomForestClassifier(**param_grid)
    elif classifier_name == "AdaBoost":
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.8, step=0.01),
        }
        model = AdaBoostClassifier(**param_grid)

    elif classifier_name == "XGBoost":
        param_grid = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
        }

        model = xgb.XGBClassifier(**param_grid)

    check_number_members = pd.concat(
        [X, pd.DataFrame(y, columns=["target"])], ignore_index=True
    )
    min_size_target_groups = check_number_members.groupby("target").size().min()

    n_split = min(min_size_target_groups, 3)
    pipeline = pipeline_func(cat_cols, num_cols, model)

    if n_split <= 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        y_preds = pipeline.predict_proba(X_test)
        return log_loss(y_test, y_preds, eps=1e-7)

    cv = StratifiedKFold(n_splits=n_split, shuffle=True)

    cv_scores = np.empty(n_split)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)

        y_preds = pipeline.predict_proba(X_val)

        cv_scores[idx] = log_loss(y_val, y_preds, eps=1e-7)
        # + std because one Fold could be good and all the others bad
        return np.mean(cv_scores) + np.std(cv_scores)
