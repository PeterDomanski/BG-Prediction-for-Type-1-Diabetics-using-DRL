import random
import pandas
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default="./")
argparser.add_argument("--metrics", dest="metrics", default="mse,rmse")
argparser.add_argument("--indices", dest="indices", default="0,100")
argparser.add_argument("--strategy", dest="strategy", default="random")
argparser.add_argument("--setup", dest="setup", default="multi_step")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path
metrics = args.metrics.split(',')
sample_indices = [int(x) for x in args.indices.strip(' ').split(',')]
# consecutive, random
strategy = args.strategy
ts_forecasting_setup = args.setup


# -------------------------------------------- Function definitions ----------------------------------------------------

def load_csv_data(path):
    df = pandas.read_csv(path)
    return df


def calculate_metric_value(data_frame, metric_names, sampling_strategy, indices, setup):
    for name in metric_names:
        metric_data = data_frame[name].values
        if setup == "multi_step":
            metric_data = [x.strip('[ ]').split(',') for x in metric_data]
            metric_data = [z[0].split(" ") for z in metric_data]
            all_values = []
            for v in metric_data:
                row_values = []
                for d in v:
                    if d != "":
                        row_values.append(float(d))
                all_values.append(row_values)
            metric_data = all_values
            metric_data = np.mean(metric_data, -1)
        if sampling_strategy == "consecutive":
            metric_data = metric_data[indices[0]:indices[1]]
        elif sampling_strategy == "random":
            metric_data = random.sample(list(metric_data), indices[1] - indices[0])
        else:
            print("{} unsupported sampling strategy".format(sampling_strategy))
        metric_val = np.mean(metric_data, 0)
        print("{}={} ({} samples, {})".format(name, metric_val, indices[1] - indices[0], sampling_strategy))


# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data = load_csv_data(csv_path)
    calculate_metric_value(data, metrics, strategy, sample_indices, ts_forecasting_setup)
