import numpy as np

# import pandas as pd
import itertools


def compute_correlations(df, columns):
    corrs = dict()
    for pair in itertools.combinations(columns, 2):
        corr = df[[pair[0], pair[1]]].corr().values[0, 1]
        corrs[f"{pair[0]} and {pair[1]}"] = corr
    return corrs


def compute_ratio(df, numerator_column_name, denominator_column_name):
    ratio = []
    for i, row in df[[numerator_column_name, denominator_column_name]].iterrows():
        try:
            ratio.append(row[numerator_column_name] / row[denominator_column_name])
        except:
            ratio.append(np.nan)
    return ratio
