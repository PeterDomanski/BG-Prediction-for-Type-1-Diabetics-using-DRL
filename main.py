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
def run(path_to_train_data="", path_to_eval_data="", setup="single_step"):
    # logging
    log_dir = "./logs/" + "log" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # load data set
    ts_train_data = dataset.load_csv_dataset(path_to_train_data)
    if path_to_eval_data != "":
        ts_eval_data = dataset.load_csv_dataset(path_to_eval_data)
    # create environment
    if setup == "single_step":
        train_env = environment.TsForecastingSingleStepEnv(ts_train_data)
        if path_to_eval_data != "":
            eval_env = environment.TsForecastingSingleStepEnv(ts_eval_data, evaluation=True)
        else:
            eval_env = environment.TsForecastingSingleStepEnv(ts_train_data, evaluation=True)
    elif setup == "multi_step":
        train_env = environment.TsForecastingMultiStepEnv(ts_train_data)
    # get TF environment
    tf_train_env = environment.get_tf_environment(train_env)
    tf_eval_env = environment.get_tf_environment(eval_env)
    # set up RL agent
    agent = rl_agent.get_rl_agent(tf_train_env)
    # train agent on environment
    training.rl_training_loop(tf_train_env, tf_eval_env, agent, file_writer, setup)
    # evaluate the agent's performance
    print("Evaluation not implemented yet")


if __name__ == '__main__':
    app.run(main)
