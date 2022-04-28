import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss

def objective(trial, X, y, pipeline_func):

    cat_cols = X.select_dtypes(include = ["object", "category"]).columns.to_list()
    num_cols = X.select_dtypes(include = np.number).columns.to_list()

    classifier_name = trial.suggest_categorical(
        "classifier",
        ["KNN", "AdaBoost"]
    )

    if classifier_name == "KNN":
        param_grid = {
            "n_neighbors": trial.suggest_int(
                "n_neighbors",
                3, 7, step = 2
            ),
            "weights": trial.suggest_categorical(
                "weights",
                ["uniform", "distance"]
            )
        }

        model = KNeighborsClassifier(**param_grid)
    elif classifier_name == "AdaBoost":
        param_grid = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                50, 
                500, 
                step = 50
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.8,
                step = 0.01
            )
        }

        model = AdaBoostClassifier(**param_grid)

    n_split = 3

    cv = StratifiedKFold(n_splits = n_split, shuffle = True)

    cv_scores = np.empty(n_split)

    pipeline = pipeline_func(cat_cols, num_cols, model)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(len(train_idx))
        print(val_idx)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        print("Xtrain", X_train.shape)
        print("XVal", X_val.shape)

        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)

        y_preds = pipeline.predict_proba(X_val)

        cv_scores[idx] = log_loss(y_val, y_preds, eps = 1e-7)

        print(cv_scores)

       # + std because one Fold could be good and all the others bad
        return  np.mean(cv_scores) + np.std(cv_scores)