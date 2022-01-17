import preprocessing as pre
import compute as cp
import print_ as pr

# Read the data
precip_data_path = "./data/climate_precip.csv"
temp_data_path = "./data/climate_temp.csv"
climate_precip = pre.read_data(precip_data_path)
climate_temp = pre.read_data(temp_data_path)

# Filter one station and inner join
precip_one_station = pre.extract_row_and_column(
    df=climate_precip, row_name="GHCND:USW00024215", column_name="STATION"
)
precip_and_temp_one_station = pre.join_df(df1=precip_one_station, df2=climate_temp)

# Clean NaN
precip_and_temp_one_station = pre.clean_nan(
    df=precip_and_temp_one_station,
    columns=["DLY-PRCP-PCTALL-GE001HI", "DLY-SNOW-PCTALL-GE030TI"],
)

# Tyope Conversion from int to datetime and extract month from date
precip_and_temp_one_station = pre.type_coversion_to_date(
    df=precip_and_temp_one_station, date_column_name="DATE"
)
precip_and_temp_one_station = pre.create_month_column_from_date(
    df=precip_and_temp_one_station, date_column_name="DATE"
)

# Groupby "MONTH" and compute sum for each columns "DLY-PRCP-PCTALL-GE001HI" and "DLY-SNOW-PCTALL-GE030TI"
monthly_precip_snow = pre.groupby_and_sum(
    df=precip_and_temp_one_station,
    group_column_name="MONTH",
    sum_columns_list=["DLY-PRCP-PCTALL-GE001HI", "DLY-SNOW-PCTALL-GE030TI"],
)

# Compute monthly rain/snow ratio
monthly_precip_snow["rain_snow_ratio"] = cp.compute_ratio(
    df=monthly_precip_snow,
    numerator_column_name="DLY-PRCP-PCTALL-GE001HI",
    denominator_column_name="DLY-SNOW-PCTALL-GE030TI",
)


# Groupby "MONTH" and compute mean of column "DLY-HTDD-NORMAL"
avg_clouds = pre.groupby_and_mean(
    df=precip_and_temp_one_station,
    group_column_name="MONTH",
    mean_columns_list=["DLY-HTDD-NORMAL"],
)

# Merge precipitations and cloud data and compute correlations
monthly_data = pre.join_df(df1=monthly_precip_snow, df2=avg_clouds, key_index=True)
corrs = cp.compute_correlations(df=monthly_data, columns=monthly_data.columns)

# Print correlations and create "monthly_data.csv"
pr.print_dict(corrs)
monthly_data.to_csv("monthly_data.csv")
