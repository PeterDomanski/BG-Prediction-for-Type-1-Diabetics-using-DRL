import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
args = argparser.parse_args()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
argparser.add_argument("--setup", dest="setup", default="multi_step")
argparser.add_argument("--save_fig", dest="save_fig", default=False)
argparser.add_argument("--save_path", dest="save_path", default=".")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path
ts_forecasting_setup = args.setup
save_fig = bool(args.save_fig)
save_path = args.save_path


# -------------------------------------------- Function definitions ----------------------------------------------------

def load_csv_data(path):
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


# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data, s = load_csv_data(csv_path)
    visualize_variance(data, ts_forecasting_setup, s)
