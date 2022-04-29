from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cat_pipeline():
    pipeline = Pipeline(
        [
            ("cat_impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", TargetEncoder()),
        ]
    )

    return pipeline


def num_pipeline():
    pipeline = Pipeline(
        [
            ("num_impute", KNNImputer(n_neighbors=5)),
        ]
    )

    return pipeline


def column_transformer_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        [("cat", cat_pipeline(), cat_cols), ("num", num_pipeline(), num_cols)]
    )

    return preprocessor


def full_pipeline(cat_cols, num_cols, model):
    pipeline = Pipeline(
        [
            ("preprocessor", column_transformer_pipeline(cat_cols, num_cols)),
            ("model", model),
        ]
    )

    return pipeline
