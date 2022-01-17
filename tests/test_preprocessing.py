import sys

sys.path.append("../submit")

from preprocessing import *
import numpy as np
import pandas as pd


def test_read_data():
    path = "test.csv"
    df_test = read_data(path)
    assert df_test.shape == (12, 5)


def test_extract_row_and_column():
    df = pd.read_csv("test.csv")
    df_test = extract_row_and_column(
        df=df, row_name=0.0, column_name="DLY-SNOW-PCTALL-GE030TI"
    )
    assert df_test.shape == (5, 5)


def test_join_df():
    df1 = pd.DataFrame(
        np.array([[-1, 2, 3, 4], [-1, 2, -3, 4], [1, 2, 3, 4]]),
        columns=["A", "B", "C", "D"],
    )
    df2 = pd.DataFrame(
        np.array([[-1, 2, 3, 4], [-1, 2, -3, 4], [1, 2, 3, 4]]),
        columns=["A", "E", "F", "G"],
    )
    df_test_false = join_df(df1=df1, df2=df2, key_index=False)
    df_test_true = join_df(df1=df1, df2=df2, key_index=True)
    assert df_test_false.shape == (5, 7)
    assert df_test_true.shape == (3, 8)


def test_clean_nan():
    df = pd.DataFrame(np.array([[-1, 2, 3, 4], [-1, 2, -3, 4], [1, 2, 3, 4]]))
    df_test = clean_nan(df=df, columns=df.columns)
    assert df_test[0].isnull().sum() == 2
    assert df_test[2].isnull().sum() == 1


def test_type_coversion_to_date():
    df = pd.DataFrame(
        np.array([[20120125, 2, 3, 4], [20020530, 1, 6, 4], [19940330, 2, 3, 4]]),
        columns=["DATE", "A", "B", "C"],
    )
    df_test = type_coversion_to_date(df=df, date_column_name="DATE")
    assert df_test["DATE"][0] == pd.to_datetime("20120125")
    assert df_test["DATE"][1] == pd.to_datetime("20020530")
    assert df_test["DATE"][2] == pd.to_datetime("19940330")
    assert df_test.shape == (3, 4)


def test_create_month_column_from_date():
    df_test = pd.DataFrame(
        np.array([["20120125", 2, 3, 4], ["20020530", 1, 6, 4], ["19940330", 2, 3, 4]]),
        columns=["DATE", "A", "B", "C"],
    )
    df_test["DATE"] = pd.to_datetime(df_test["DATE"])
    df_test = create_month_column_from_date(df=df_test, date_column_name="DATE")
    assert (df_test["MONTH"][0], df_test["MONTH"][1], df_test["MONTH"][2]) == (1, 5, 3)
    assert df_test.shape == (3, 5)


def test_groupby_and_sum():
    df = pd.DataFrame(
        np.array([[100, 2], [100, 6], [200, 11], [200, 12]]), columns=["A", "B"]
    )
    df_test = groupby_and_sum(df=df, group_column_name="A", sum_columns_list=["B"])
    assert df_test["B"].to_list() == [8, 23]
    assert df_test.shape == (2, 1)


def test_groupby_and_mean():
    df = pd.DataFrame(
        np.array([[100, 2], [100, 6], [200, 11], [200, 12]]), columns=["A", "B"]
    )
    df_test = groupby_and_mean(df=df, group_column_name="A", mean_columns_list=["B"])
    assert df_test["B"].to_list() == [4, 11.5]
    assert df_test.shape == (2, 1)
