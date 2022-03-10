import gin
from absl import app, logging
from data import dataset
from rl import environment, rl_agent
import training
import tensorflow as tf
import datetime


def main(args):
    # parse config file
    gin.parse_config_file("config.gin")
    run()


@gin.configurable
def run(path_to_data="", setup="single_step"):
    # logging
    log_dir = "./logs/" + "log" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # load data set
    ts_data = dataset.load_csv_dataset(path_to_data)
    # create environment
    if setup == "single_step":
        train_env = environment.TsForecastingSingleStepEnv(ts_data)
    elif setup == "multi_step":
        train_env = environment.TsForecastingMultiStepEnv(ts_data)
    # get TF environment
    tf_train_env = environment.get_tf_environment(train_env)
    # set up RL agent
    agent = rl_agent.get_rl_agent(tf_train_env)
    # train agent on environment
    training.rl_training_loop(tf_train_env, agent, file_writer)
    # evaluate the agent's performance
    print("Evaluation not implemented yet")


if __name__ == '__main__':
    app.run(main)
