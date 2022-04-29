import numpy as np
import pandas as pd

from src.apps.preprocessing import feature_drop


def test_calc_corr_matrix():
    correlated_cols = np.array(
        [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16]]
    )

    df = pd.DataFrame(
        np.repeat(correlated_cols, 1000, axis=0), columns=["First", "Second"]
    )

    corr_matrix = feature_drop.calc_corr_matrix(df)
    assert 0.98 <= corr_matrix.iloc[0, 1]


def test_drop_high_corr_features():
    correlated_cols = np.array(
        [
            [1, 2, 3],
            [2, 4, 1],
            [3, 6, 1],
            [4, 8, 5],
            [5, 10, 0],
            [6, 12, 21],
            [7, 14, 44],
            [8, 16, 59],
        ]
    )

    min_corr = 0.95

    df = pd.DataFrame(
        np.repeat(correlated_cols, 1000, axis=0), columns=["First", "Second", "Third"]
    )

    corr_matrix = feature_drop.calc_corr_matrix(df)

    columns_to_drop = feature_drop.drop_high_corr_feature(corr_matrix, min_corr)

    assert len(columns_to_drop) == 1
