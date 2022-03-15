import pandas
import datetime


def load_csv_dataset(path_to_data):
    pandas_ds = pandas.read_csv(path_to_data)
    # only keep data attribute of interest, e.g., CGM or execution time (we assume that attribute in last column)
    date_format_str = '%d-%m-%Y %H:%M:%S'
    total_time = datetime.datetime.strptime(pandas_ds["_ts"].iloc[-1], date_format_str) - datetime.datetime.strptime(
        pandas_ds["_ts"].iloc[0], date_format_str)
    total_time = total_time.seconds
    # total time in hours
    total_time /= 360
    return pandas_ds.iloc[:, -1], total_time
