import pandas


def load_csv_dataset(path_to_data):
    pandas_ds = pandas.read_csv(path_to_data)
    # only keep data attribute of interest, e.g., CGM or execution time (we assume that attribute in last column)
    return pandas_ds.iloc[:, -1]
