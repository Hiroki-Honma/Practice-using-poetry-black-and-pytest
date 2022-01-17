import sys

sys.path.append("../submit")

from compute import *
import numpy as np
import pandas as pd


def test_compute_correlations():
    df_test = pd.DataFrame(np.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]]))
    corr = compute_correlations(df=df_test, columns=df_test.columns)
    assert len(corr) == 3
    assert type(corr) == dict


def test_compute_ratio():
    df_test = pd.DataFrame(np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]))
    ratio = compute_ratio(
        df=df_test, numerator_column_name=0, denominator_column_name=1
    )
    assert len(ratio) == 2
    assert type(ratio) == list
