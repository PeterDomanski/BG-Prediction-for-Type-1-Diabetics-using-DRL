import pandas
import datetime
import numpy as np


def load_csv_dataset(path_to_data):
    pandas_ds = pandas.read_csv(path_to_data)
    # only keep data attribute of interest, e.g., CGM or execution time (we assume that attribute in last column)
    date_format_str = '%d-%m-%Y %H:%M:%S'
    # total_time = datetime.datetime.strptime(pandas_ds["_ts"].iloc[-1], date_format_str) - datetime.datetime.strptime(
    #     pandas_ds["_ts"].iloc[0], date_format_str)
    if "RasberryPi" in path_to_data:
        total_time = (35.09 / 60)
    else:
        # total_time = datetime.datetime.strptime(pandas_ds.iloc[0, -1], date_format_str) - datetime.datetime.strptime(
        #     pandas_ds.iloc[0, 0], date_format_str)
        total_time = datetime.datetime.strptime(pandas_ds["_ts"].iloc[-1], date_format_str) - \
                     datetime.datetime.strptime(pandas_ds["_ts"].iloc[0], date_format_str)
        total_time = np.round((total_time.days * 24) + (total_time.seconds / 3600))
    return pandas_ds.iloc[:, -1].astype(np.float32), total_time


def data_normalization(data_frame_train, data_frame_test, normalization_type):
    data_frame = pandas.concat([data_frame_train, data_frame_test])
    min_attribute_value = data_frame.min()
    max_attribute_value = data_frame.max()
    mean_attribute_value = data_frame.mean()
    std = data_frame.std()
    data_summary = {"min": min_attribute_value, "max": max_attribute_value, "mean": mean_attribute_value, "std": std,
                    "normalization_type": normalization_type}
    if normalization_type == "min_max":
        data_frame_train = (data_frame_train - min_attribute_value) / (max_attribute_value - min_attribute_value)
        data_frame_test = (data_frame_test - min_attribute_value) / (max_attribute_value - min_attribute_value)
    elif normalization_type == "mean":
        data_frame_train = (data_frame_train - mean_attribute_value) / (max_attribute_value - min_attribute_value)
        data_frame_test = (data_frame_test - mean_attribute_value) / (max_attribute_value - min_attribute_value)
    elif normalization_type == "z_score":
        data_frame_train = (data_frame_train - mean_attribute_value) / std
        data_frame_test = (data_frame_test - mean_attribute_value) / std
    return data_frame_train, data_frame_test, data_summary


def undo_data_normalization_sample_wise(sample, data_summary):
    if data_summary["normalization_type"] == "min_max":
        non_normalized_sample = (sample * (data_summary["max"] - data_summary["min"])) + data_summary["min"]
    elif data_summary["normalization_type"] == "mean":
        non_normalized_sample = (sample * (data_summary["max"] - data_summary["min"])) + data_summary["mean"]
    elif data_summary["normalization_type"] == "z_score":
        non_normalized_sample = (sample * data_summary["std"]) + data_summary["mean"]
    return non_normalized_sample
