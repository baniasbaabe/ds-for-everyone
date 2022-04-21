import pandas as pd
import numpy as np

def calc_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = df.select_dtypes(include = np.number).corr().abs()
    return corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

def drop_high_corr_feature(corr_matrix: pd.DataFrame, min_corr: float) -> list:
    return [column for column in corr_matrix.columns if any(corr_matrix[column] > min_corr)]
