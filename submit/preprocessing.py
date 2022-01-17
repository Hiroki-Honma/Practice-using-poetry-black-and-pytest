import numpy as np
import pandas as pd
import itertools


def read_data(data_path):
    data = pd.read_csv(data_path)
    return data


def extract_row_and_column(df, row_name, column_name):
    return df[df[column_name] == row_name]


def join_df(df1, df2, key_index=False):
    return pd.merge(df1, df2, left_index=key_index, right_index=key_index)


def clean_nan(df, columns):
    for column in columns:
        df.loc[df[column] < 0, column] = np.nan
    return df


def type_coversion_to_date(df, date_column_name):
    df[date_column_name] = pd.to_datetime(df[date_column_name].astype(str))
    return df


def create_month_column_from_date(df, date_column_name):
    df["MONTH"] = df[date_column_name].dt.month
    return df


def groupby_and_sum(df, group_column_name, sum_columns_list):
    df_ = df.groupby([group_column_name])[sum_columns_list].agg(sum)
    return df_


def groupby_and_mean(df, group_column_name, mean_columns_list):
    df_ = df.groupby([group_column_name])[mean_columns_list].agg(np.mean)
    return df_
