import gin
from absl import app, logging
from data import dataset
from rl import environment, rl_agent
import training
import tensorflow as tf
import datetime
import os


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.info(e)


def main(args):
    # parse config file
    gin.parse_config_file("config.gin")
    run()


@gin.configurable
def run(path_to_train_data="", path_to_eval_data="", normalization=False, normalization_type="min_max",
        setup="single_step", rl_algorithm="ddpg", env_implementation="tf", use_gpu=False, multi_task=False):
    # logging
    log_dir = "./logs/" + "log" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # load data set
    if multi_task:
        dataset_files = sorted(os.listdir(path_to_train_data), key=lambda index: int(index.split("-")[0]))
        patients_train_datasets = []
        patient_train_total_times = []
        for f in dataset_files:
            ts_train_data_sp, total_train_time_h = dataset.load_csv_dataset(os.path.join(path_to_train_data, f))
            patients_train_datasets.append(ts_train_data_sp)
            patient_train_total_times.append(total_train_time_h)
        ts_train_data = patients_train_datasets
        total_train_time_h = int((len(ts_train_data) * 5) / 60) + 1
    else:
        ts_train_data, total_train_time_h = dataset.load_csv_dataset(path_to_train_data)
    if path_to_eval_data != "":
        if multi_task:
            dataset_files = sorted(os.listdir(path_to_eval_data), key=lambda index: int(index.split("-")[0]))
            patients_eval_datasets = []
            patient_eval_total_times = []
            for f in dataset_files:
                ts_eval_data_sp, total_eval_time_h = dataset.load_csv_dataset(os.path.join(path_to_eval_data, f))
                patients_eval_datasets.append(ts_eval_data_sp)
                patient_eval_total_times.append(total_eval_time_h)
            ts_eval_data = patients_eval_datasets
            total_eval_time_h = int((len(ts_eval_data) * 5) / 60) + 1
        else:
            ts_eval_data, total_eval_time_h = dataset.load_csv_dataset(path_to_eval_data)
    else:
        ts_eval_data = ts_train_data
        total_eval_time_h = total_train_time_h
    if normalization:
        if multi_task:
            ts_train_data, ts_eval_data, data_summary = dataset.data_normalization_multi_patient(
                ts_train_data, ts_eval_data, normalization_type=normalization_type)
        else:
            ts_train_data, ts_eval_data, data_summary = dataset.data_normalization(
                ts_train_data, ts_eval_data, normalization_type=normalization_type)
    else:
        data_summary = {}
    # create environment
    if setup == "single_step":
        if env_implementation == "tf":
            train_env = environment.TsForecastingSingleStepTFEnv(ts_train_data, rl_algorithm, data_summary)
            train_env_eval = environment.TsForecastingSingleStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
        else:
            train_env = environment.TsForecastingSingleStepEnv(ts_train_data, rl_algorithm=rl_algorithm)
            train_env_eval = environment.TsForecastingSingleStepEnv(
                ts_train_data, evaluation=True, max_window_count=-1, rl_algorithm=rl_algorithm)
        if normalization:
            # max_attribute_val = train_env.max_attribute_val * data_summary["max"],
            max_attribute_val = dataset.undo_data_normalization_sample_wise(train_env.max_attribute_val, data_summary)
        else:
            max_attribute_val = train_env.max_attribute_val
        if path_to_eval_data != "":
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepTFEnv(ts_eval_data, rl_algorithm, data_summary)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(
                    ts_eval_data, evaluation=True, rl_algorithm=rl_algorithm, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepEnv(ts_eval_data, rl_algorithm=rl_algorithm)
        else:
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepTFEnv(ts_train_data, rl_algorithm, data_summary)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(
                    ts_train_data, evaluation=True, rl_algorithm=rl_algorithm, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepEnv(ts_train_data, rl_algorithm)
        forecasting_steps = 1
        num_iter = train_env.max_window_count
    elif setup == "multi_step":
        if env_implementation == "tf":
            train_env = environment.TsForecastingMultiStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, multi_task=multi_task)
            train_env_eval = environment.TsForecastingMultiStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, multi_task=multi_task,
                evaluation=True, max_window_count=-1)
        else:
            train_env = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm)
            train_env_eval = environment.TsForecastingMultiStepEnv(
                ts_train_data, rl_algorithm, evaluation=True, max_window_count=-1)
        if normalization:
            max_attribute_val = dataset.undo_data_normalization_sample_wise(train_env.max_attribute_val, data_summary)
        else:
            max_attribute_val = train_env.max_attribute_val
        if path_to_eval_data != "":
            if env_implementation == "tf":
                eval_env = environment.TsForecastingMultiStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1,
                    multi_task=multi_task)
                eval_env_train = environment.TsForecastingMultiStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, multi_task=multi_task)
            else:
                eval_env = environment.TsForecastingMultiStepEnv(ts_eval_data, rl_algorithm,
                                                                 evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingMultiStepEnv(ts_eval_data, rl_algorithm)
        else:
            if env_implementation == "tf":
                eval_env = environment.TsForecastingMultiStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1,
                    multi_task=multi_task)
                eval_env_train = environment.TsForecastingMultiStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, multi_task=multi_task)
            else:
                eval_env = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm, evaluation=True,
                                                                 max_window_count=-1)
                eval_env_train = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm)
        forecasting_steps = train_env.pred_horizon
        num_iter = train_env.max_window_count

    if env_implementation != "tf":
        # get TF environment
        tf_train_env = environment.get_tf_environment(train_env)
        tf_train_env_eval = environment.get_tf_environment(train_env_eval)
        tf_eval_env = environment.get_tf_environment(eval_env)
        tf_eval_env_train = environment.get_tf_environment(eval_env_train)
    else:
        tf_train_env = train_env
        tf_train_env_eval = train_env_eval
        tf_eval_env = eval_env
        tf_eval_env_train = eval_env_train
    # set up RL agent
    agent = rl_agent.get_rl_agent(tf_train_env, rl_algorithm, use_gpu)
    # save gin's operative config to a file before training
    config_txt_file = open(log_dir + "/gin_config.txt", "w+")
    config_txt_file.write("Configuration options available before training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()
    # train agent on environment
    training.rl_training_loop(log_dir, tf_train_env, tf_train_env_eval, tf_eval_env, tf_eval_env_train, agent,
                              ts_train_data, ts_eval_data, file_writer, setup, forecasting_steps, rl_algorithm,
                              total_train_time_h, total_eval_time_h, max_attribute_val, num_iter, data_summary,
                              env_implementation, multi_task)
    # save gin's operative config to a file after training
    config_txt_file = open(log_dir + "/gin_config.txt", "a")
    config_txt_file.write("\n")
    config_txt_file.write("Configuration options available after training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()


if __name__ == '__main__':
    app.run(main)
