import gin
from absl import app, logging
from data import dataset
from rl import environment, rl_agent
import training
import tensorflow as tf
import datetime


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
def run(path_to_train_data="", path_to_eval_data="", setup="single_step", rl_algorithm="ddpg", env_implementation="tf",
        use_gpu=False):
    # logging
    log_dir = "./logs/" + "log" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # load data set
    ts_train_data, total_time_h = dataset.load_csv_dataset(path_to_train_data)
    if path_to_eval_data != "":
        ts_eval_data, total_time_h = dataset.load_csv_dataset(path_to_eval_data)
    else:
        ts_eval_data = ts_train_data
    # create environment
    if setup == "single_step":
        if env_implementation == "tf":
            train_env = environment.TsForecastingSingleStepTFEnv(ts_train_data)
        else:
            train_env = environment.TsForecastingSingleStepEnv(ts_train_data, rl_algorithm=rl_algorithm)
        max_attribute_val = train_env.max_attribute_val
        if path_to_eval_data != "":
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(ts_eval_data, evaluation=True, max_window_count=-1)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(ts_eval_data, evaluation=True,
                                                                  rl_algorithm=rl_algorithm, max_window_count=-1)
        else:
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(ts_train_data, evaluation=True, max_window_count=-1)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(ts_train_data, evaluation=True,
                                                                  rl_algorithm=rl_algorithm, max_window_count=-1)
        forecasting_steps = 1
    elif setup == "multi_step":
        train_env = environment.TsForecastingMultiStepEnv(ts_train_data)
        if path_to_eval_data != "":
            eval_env = environment.TsForecastingMultiStepEnv(ts_eval_data, evaluation=True, max_window_count=-1)
        else:
            eval_env = environment.TsForecastingMultiStepEnv(ts_train_data, evaluation=True, max_window_count=-1)
        forecasting_steps = eval_env.forecasting_steps
    if env_implementation != "tf":
        # get TF environment
        tf_train_env = environment.get_tf_environment(train_env)
        tf_eval_env = environment.get_tf_environment(eval_env)
    else:
        tf_train_env = train_env
        tf_eval_env = eval_env
    # set up RL agent
    agent = rl_agent.get_rl_agent(tf_train_env, rl_algorithm, use_gpu)
    # save gin's operative config to a file before training
    config_txt_file = open(log_dir + "/gin_config.txt", "w+")
    config_txt_file.write("Configuration options available before training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()
    # train agent on environment
    training.rl_training_loop(log_dir, tf_train_env, tf_eval_env, agent, ts_eval_data, file_writer, setup,
                              forecasting_steps, rl_algorithm, total_time_h, max_attribute_val, env_implementation)
    # save gin's operative config to a file after training
    config_txt_file = open(log_dir + "/gin_config.txt", "a")
    config_txt_file.write("\n")
    config_txt_file.write("Configuration options available after training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()


if __name__ == '__main__':
    app.run(main)
