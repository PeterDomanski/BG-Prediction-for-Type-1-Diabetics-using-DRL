import os
import pandas
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path",
                       default="/home/domanspr/rl_time_series_forecasting/logs/multi_step/log2022-05-02_16-44-26/data_summaries/performance_summary_35000_eval.csv")
argparser.add_argument("--setup", dest="setup", default="multi_step")
# vis_eval_samples, vis_avg_training, vis_avg_forecasting
# Note: pass path to dir for vis_avg_training otherwise pass explicit file (in csv_path argument)
argparser.add_argument("--vis_type", dest="vis_type", default="vis_avg_forecasting")
argparser.add_argument("--vis_steps", dest="vis_steps", default=80)
argparser.add_argument("--vis_std", dest="vis_std", default=True)
argparser.add_argument("--vis_forecasting_error", dest="vis_forecasting_error", default=True)
argparser.add_argument("--error_metric", dest="error_metric", default="mae")
argparser.add_argument("--y_lim", dest="y_lim", default=25)
argparser.add_argument("--dataset_path", dest="dataset_path",
                       default="/home/domanspr/Documents/Duke/projects/RL_for_time_series_forecasting/datasets/OhioT1DM/570-ws-testing.csv")
argparser.add_argument("--save_fig", dest="save_fig", default=False)
argparser.add_argument("--save_path", dest="save_path", default="/home/domanspr/Documents/Duke/projects/RL_for_time_series_forecasting")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path
ts_forecasting_setup = args.setup
save_fig = bool(args.save_fig)
save_path = args.save_path
vis_type = args.vis_type
vis_steps = args.vis_steps
vis_std = args.vis_std
vis_forecasting_error = args.vis_forecasting_error
error_metric_name = args.error_metric
y_lim = args.y_lim
path_to_ds = args.dataset_path


# -------------------------------------------- Function definitions ----------------------------------------------------

def load_csv_data(path):
    if os.path.isdir(path):
        csv_files = sorted(os.listdir(path), key=lambda f_name: int(f_name.split('_')[-2]))
        df = []
        step = []
        for i, f in enumerate(csv_files):
            if vis_steps == -1 or i < vis_steps:
                df.append(pandas.read_csv(os.path.join(path, f)))
                step.append(int(f.split('_')[-2]))
    else:
        df = pandas.read_csv(path)
        step = int(path.split('/')[-1].split('_')[-2])
    return df, step


def visualize_variance(data_frame, setup, step):
    var_data = data_frame['variance'].values
    num_x_values = len(data_frame)
    if setup == "multi_step":
        var_data = [x.strip('[ ]').split(',') for x in var_data]
        var_data = [z[0].split(" ") for z in var_data]
        all_values = []
        for v in var_data:
            row_values = []
            for d in v:
                if d != "":
                    row_values.append(float(d))
            all_values.append(row_values)
        var_data = np.array(all_values)
        max_var_val = np.max(var_data)
        fig, ax = plt.subplots(nrows=2, ncols=int(np.ceil(var_data.shape[-1] / 2)), figsize=(30, 15))
        fig.suptitle("Training step {}".format(step))
        for dim in range(var_data.shape[-1]):
            d = var_data[:, dim]
            col_index = dim % int(np.ceil(var_data.shape[-1] / 2))
            row_index = dim // int(np.ceil(var_data.shape[-1] / 2))
            x_values = np.linspace(start=0, stop=num_x_values, num=num_x_values)
            ax[row_index, col_index].plot(x_values, d)
            ax[row_index, col_index].set_ylim(ymin=0.0, ymax=max_var_val)
            ax[row_index, col_index].set_xlabel('Evaluation steps')
            ax[row_index, col_index].set_ylabel('Variance (forecasting step {})'.format(dim + 1))
        if save_fig:
            plt.savefig(save_path + "/uq_var_train_step_" + str(step) + ".pdf", dpi=300)
        else:
            plt.show()


def visualize_avg_var_training(data_frame, setup, step, show_forecasting_error, error_metric):
    if setup == "multi_step":
        all_var_data = []
        all_min_vals, all_max_vals = [], []
        if show_forecasting_error:
            all_error_data = []
            all_min_error_vals, all_max_error_vals = [], []
        # plot variance or standard deviation
        for df in data_frame:
            var_data = df['variance'].values
            var_data = [x.strip('[ ]').split(',') for x in var_data]
            var_data = [z[0].split(" ") for z in var_data]
            if show_forecasting_error:
                error_data = df[error_metric].values
                error_data = [x.strip('[ ]').split(',') for x in error_data]
                error_data = [z[0].split(" ") for z in error_data]

            all_var_values = []
            for v in var_data:
                row_values = []
                for d in v:
                    if d != "":
                        row_values.append(float(d))
                all_var_values.append(np.mean(row_values))

            if show_forecasting_error:
                all_error_values = []
                for e in error_data:
                    row_metric_values = []
                    for v in e:
                        if v != "":
                            row_metric_values.append(float(v))
                    all_error_values.append(np.mean(row_metric_values))

            all_min_vals.append(np.min(all_var_values))
            all_max_vals.append(np.max(all_var_values))
            var_data = np.array(np.mean(all_var_values))
            all_var_data.append(var_data)
            if show_forecasting_error:
                all_min_error_vals.append(np.min(all_error_values))
                all_max_error_vals.append(np.max(all_error_values))
                error_data = np.array(np.mean(all_error_values))
                all_error_data.append(error_data)
        all_var_data = np.array(all_var_data)
        all_min_vals = np.array(all_min_vals)
        all_max_vals = np.array(all_max_vals)
        if show_forecasting_error:
            all_error_data = np.array(all_error_data)
            # all_min_error_vals = np.array(all_min_error_vals)
            # all_max_error_vals = np.array(all_max_error_vals)
        fig, ax = plt.subplots(figsize=(30, 15))
        if vis_std:
            all_var_data = np.sqrt(all_var_data)
            all_min_vals = np.sqrt(all_min_vals)
            all_max_vals = np.sqrt(all_max_vals)
        max_var_val = np.max(all_max_vals)
        ax.plot(step, all_var_data, color='black', label="average")
        ax.plot(step, all_min_vals, color='green', label="minimum")
        ax.plot(step, all_max_vals, color='orange', label="maximum")
        ax.fill_between(step, all_min_vals, all_max_vals, color='blue', alpha=0.6)
        ax.set_ylim(ymin=0.0, ymax=max_var_val)
        ax.set_xlabel('Training steps')
        plt.legend()
        if vis_std:
            ax.set_ylabel('Standard deviation')
        else:
            ax.set_ylabel('Variance')
        if show_forecasting_error:
            ax2 = ax.twinx()
            ax2.set_ylabel('{} error'.format(error_metric), color='red')
            ax2.plot(step, all_error_data, color='red', label=error_metric, linewidth=3)
            # ax2.plot(step, all_min_error_vals, color='red')
            # ax2.plot(step, all_max_error_vals, color='red')
            # ax2.fill_between(step, all_min_error_vals, all_max_error_vals, color='red', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='red')
        # plt.legend()
        if save_fig:
            plt.savefig(save_path + "/uq_var_train_training.pdf", dpi=300)
        else:
            plt.show()


def visualize_var_forecasting(data_frame, setup, step, y_max):
    var_data = data_frame['variance'].values
    if setup == "multi_step":
        var_data = [x.strip('[ ]').split(',') for x in var_data]
        var_data = [z[0].split(" ") for z in var_data]
        all_values = []
        for v in var_data:
            row_values = []
            for d in v:
                if d != "":
                    row_values.append(float(d))
            all_values.append(row_values)
        var_data = np.array(all_values)
        min_var_data = np.min(var_data, axis=0)
        max_var_data = np.max(var_data, axis=0)
        window_id_min_error, window_id_max_error = {}, {}
        for i in range(len(min_var_data)):
            # if np.where(var_data == min_var_data[i])[0] not in window_id_min_error.values():
            window_id_min_error[str(i + 1)] = int(np.where(var_data == min_var_data[i])[0])
        for j in range(len(max_var_data)):
            # if np.where(var_data == max_var_data[j])[0] not in window_id_max_error.values():
            window_id_max_error[str(j + 1)] = int(np.where(var_data == max_var_data[j])[0])
        var_data = np.mean(var_data, axis=0)
        num_x_values = len(var_data)
        if vis_std:
            var_data = np.sqrt(var_data)
            min_var_data = np.sqrt(min_var_data)
            max_var_data = np.sqrt(max_var_data)
        fig, ax = plt.subplots(figsize=(30, 15))
        fig.suptitle("Evaluation step {}".format(step))
        x_values = np.linspace(start=0, stop=num_x_values, num=num_x_values)
        ax.plot(x_values, var_data, color='black', label='average')
        ax.plot(x_values, min_var_data, color='green', label='minimum')
        ax.plot(x_values, max_var_data, color='orange', label='maximum')
        ax.fill_between(x_values, min_var_data, max_var_data)
        ax.set_ylim(ymin=0.0, ymax=y_max)
        print("Maximum variance: {}".format(np.max(max_var_data)))
        ax.set_xlabel('Forecasting steps')
        if vis_std:
            ax.set_ylabel('Standard deviation')
        else:
            ax.set_ylabel('Variance')
        plt.legend()
    if save_fig:
        plt.savefig(save_path + "/uq_var_train_step_" + str(step) + ".pdf", dpi=300)
    else:
        plt.show()

    return window_id_min_error, window_id_max_error


def visualize_windows_on_dataset(ds_path, window_ids, mode, window_size=6, x_axis="sample_id"):
    fig, ax = plt.subplots()
    # load dataset from ds_path
    dataset = pandas.read_csv(ds_path)
    date_format_str = '%d-%m-%Y %H:%M:%S'
    if x_axis == "time":
        total_time = datetime.datetime.strptime(dataset["_ts"].iloc[-1], date_format_str) - datetime.datetime.strptime(
            dataset["_ts"].iloc[0], date_format_str)
        total_time = (total_time.days * 24) + (total_time.seconds / 3600)
        x_values = np.linspace(start=0, stop=total_time, num=len(dataset))
        ax.set_xlabel("Time")
    elif x_axis == "sample_id":
        x_values = np.linspace(start=0, stop=len(dataset), num=len(dataset))
        ax.set_xlabel("sample ID")
    ax.plot(x_values, dataset['_value'].values)
    # show windows with min and max error
    for v in set(window_ids.values()):
        # offset: 1 window
        x_low = window_size * v + window_size
        x_high = window_size * (v + 1) + window_size
        # each step is 5 min; time in hours
        data_values = dataset['_value'].iloc[x_low:x_high]
        if x_axis == "time":
            x_low_time = datetime.datetime.strptime(
                dataset['_ts'].iloc[x_low], date_format_str) - datetime.datetime.strptime(dataset["_ts"].iloc[0],
                                                                                          date_format_str)
            x_low_time = x_low_time.days * 24 + (x_low_time.seconds / 3600)
            x_high_time = datetime.datetime.strptime(
                dataset['_ts'].iloc[x_high], date_format_str) - datetime.datetime.strptime(dataset["_ts"].iloc[0],
                                                                                           date_format_str)
            x_high_time = x_high_time.days * 24 + (x_high_time.seconds / 3600)
        y_min = np.min(data_values)
        y_max = np.max(data_values)
        if mode == "min":
            if x_axis == "time":
                ax.add_patch(Rectangle((x_low_time, y_min), x_high_time - x_low_time, y_max - y_min, fill=False,
                                       color='magenta', ls="--"))
            elif x_axis == "sample_id":
                ax.add_patch(Rectangle((x_low, y_min), x_high - x_low, y_max - y_min, fill=False,
                                       color='magenta', ls="--", lw=2))
        elif mode == "max":
            if x_axis == "time":
                ax.add_patch(Rectangle((x_low_time, y_min), x_high_time - x_low_time, y_max - y_min, fill=False,
                                       color='red', ls="--"))
            elif x_axis == "sample_id":
                ax.add_patch(Rectangle((x_low, y_min), x_high - x_low, y_max - y_min, fill=False,
                                       color='red', ls="--", lw=2))
    plt.title("{} errors".format(mode))
    ax.set_ylabel("CGM value")
    plt.show()


# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data, s = load_csv_data(csv_path)
    if vis_type == "vis_eval_samples":
        visualize_variance(data, ts_forecasting_setup, s)
    elif vis_type == "vis_avg_training":
        visualize_avg_var_training(data, ts_forecasting_setup, s, vis_forecasting_error, error_metric_name)
    elif vis_type == "vis_avg_forecasting":
        min_error_window_ids, max_error_window_ids = visualize_var_forecasting(data, ts_forecasting_setup, s, y_lim)
        print("Windows with min error: {}".format(min_error_window_ids))
        print("Windows with max error: {}".format(max_error_window_ids))
        visualize_windows_on_dataset(path_to_ds, min_error_window_ids, mode="min")
        visualize_windows_on_dataset(path_to_ds, max_error_window_ids, mode="max")
