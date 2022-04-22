import tensorflow as tf
from data import dataset
from absl import logging
import csv
import os


def compute_avg_return(env, policy, env_implementation, data_summary, num_iter=16, normalize=False, use_rnn_state=True):
    total_return = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_return = 0.0
        time_series_counter = 0

        while not time_step.is_last():
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            if use_rnn_state:
                if env_implementation == "tf":
                    time_step = env.step(action_step, rnn_state)
                elif env_implementation == "gym":
                    time_step = env.step(action_step)
            else:
                time_step = env.step(action_step)
            # if len(data_summary) == 0:
            #     episode_return += time_step.reward
            # else:
            #     episode_return += dataset.undo_data_normalization_sample_wise(time_step.reward, data_summary)
            episode_return += time_step.reward
            time_series_counter += 1

        if normalize:
            total_return += (episode_return / time_series_counter)
        else:
            total_return += episode_return

    avg_return = total_return / num_iter
    return tf.squeeze(avg_return)


def compute_metrics_single_step(env, policy, env_implementation, data_summary, step, log_dir,
                                metrics=['mae', 'mse', 'rmse'], num_iter=1, use_rnn_state=True, prefix="train"):
    if 'mae' in metrics:
        total_mae = 0.0
    if 'mse' in metrics:
        total_mse = 0.0
    if 'rmse' in metrics:
        total_rmse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        if 'mae' in metrics:
            episode_mae = 0.0
        if 'mse' in metrics:
            episode_mse = 0.0
        if 'rmse' in metrics:
            episode_rmse = 0.0
        step_counter = 0

        while not time_step.is_last():
            parameter_values = {}
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            if use_rnn_state:
                if env_implementation == "tf":
                    time_step = env.step(action_step, rnn_state)
                elif env_implementation == "gym":
                    time_step = env.step(action_step)
            else:
                time_step = env.step(action_step)
            # agent forecast
            if len(data_summary) == 0:
                agent_pred = tf.squeeze(action_step)
                ground_truth = time_step.reward
            else:
                agent_pred = dataset.undo_data_normalization_sample_wise(tf.squeeze(action_step), data_summary)
                ground_truth = dataset.undo_data_normalization_sample_wise(time_step.reward, data_summary)
            if 'mae' in metrics:
                mae_val = tf.math.abs(agent_pred - ground_truth)
                episode_mae += mae_val
                parameter_values['mae'] = tf.squeeze(mae_val).numpy()
            if 'mse' in metrics:
                mse_val = (agent_pred - ground_truth) ** 2
                episode_mse += mse_val
                parameter_values['mse'] = tf.squeeze(mse_val).numpy()
            if 'rmse' in metrics:
                rmse_val = (agent_pred - ground_truth) ** 2
                episode_rmse += rmse_val
                parameter_values['rmse'] = tf.squeeze(tf.math.sqrt(rmse_val)).numpy()

            if prefix == "eval":
                action_distribution = policy.distribution(time_step, rnn_state)[0]
                if len(data_summary) == 0:
                    action_mean = action_distribution.mean()
                else:
                    action_mean = dataset.undo_data_normalization_sample_wise(action_distribution.mean(), data_summary)
                parameter_values['mean'] = tf.squeeze(action_mean).numpy()
                action_variance = calculate_action_variance(action_distribution, action_mean, data_summary)
                parameter_values['variance'] = tf.squeeze(action_variance).numpy()
                write_to_csv_file(parameter_values, step, log_dir, prefix)

        if 'mae' in metrics:
            total_mae += episode_mae / step_counter
        if 'mse' in metrics:
            total_mse += episode_mse / step_counter
        if 'rmse' in metrics:
            total_rmse += tf.math.sqrt(episode_rmse / step_counter)

    if 'mae' in metrics:
        avg_mae = total_mae / num_iter
        logging.info("[{}] MAE (step {}): {}".format(prefix, step, avg_mae))
    if 'mse' in metrics:
        avg_mse = total_mse / num_iter
        logging.info("[{}] MSE (step {}): {}".format(prefix, step, avg_mse))
    if 'rmse' in metrics:
        avg_rmse = total_rmse / num_iter
        logging.info("[{}] RMSE (step {}): {}".format(prefix, step, avg_rmse))
    return tf.squeeze(avg_mae), tf.squeeze(avg_mse), tf.squeeze(avg_rmse)


def compute_metrics_multi_step(env, policy, env_implementation, data_summary, ts_data, pred_horizon, step, log_dir,
                               metrics=['mae', 'mse', 'rmse'],  num_iter=1, use_rnn_state=True, prefix="train"):
    if 'mae' in metrics:
        total_mae = 0.0
    if 'mse' in metrics:
        total_mse = 0.0
    if 'rmse' in metrics:
        total_rmse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        if 'mae' in metrics:
            episode_mae = 0.0
        if 'mse' in metrics:
            episode_mse = 0.0
        if 'rmse' in metrics:
            episode_rmse = 0.0
        step_counter = 0

        while not time_step.is_last():
            parameter_values = {}
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            # ground_truth = env._current_ground_truth
            ground_truth_pos = int(tf.squeeze(env._current_data_pos))
            if use_rnn_state:
                if env_implementation == "tf":
                    time_step = env.step(action_step, rnn_state)
                elif env_implementation == "gym":
                    time_step = env.step(action_step)
            else:
                time_step = env.step(action_step)
            # agent forecast
            if len(data_summary) == 0:
                agent_pred = tf.squeeze(action_step)
                # ground_truth_pos = int(tf.squeeze(time_step.reward))
                # ground_truth = ts_data[ground_truth_pos - pred_horizon:ground_truth_pos]
                ground_truth = ts_data[ground_truth_pos:ground_truth_pos + pred_horizon]
            else:
                agent_pred = dataset.undo_data_normalization_sample_wise(tf.squeeze(action_step), data_summary)
                # ground_truth_pos = int(tf.squeeze(time_step.reward))
                # ground_truth = ts_data[ground_truth_pos - pred_horizon:ground_truth_pos]
                ground_truth = ts_data[ground_truth_pos:ground_truth_pos + pred_horizon]
                ground_truth = dataset.undo_data_normalization_sample_wise(ground_truth, data_summary)
            if 'mae' in metrics:
                episode_mae_val = tf.math.abs(agent_pred - ground_truth)
                episode_mae += tf.math.reduce_mean(episode_mae_val)
                parameter_values['mae'] = tf.squeeze(episode_mae_val).numpy()
            if 'mse' in metrics:
                episode_mse_val = (agent_pred - ground_truth) ** 2
                episode_mse += tf.math.reduce_mean(episode_mse_val)
                parameter_values['mse'] = tf.squeeze(episode_mse_val).numpy()
            if 'rmse' in metrics:
                episode_rmse_val = (agent_pred - ground_truth) ** 2
                episode_rmse += tf.math.reduce_mean(episode_rmse_val)
                parameter_values['rmse'] = tf.squeeze(tf.math.sqrt(episode_rmse_val)).numpy()

            if prefix == "eval":
                action_distribution = policy.distribution(time_step, rnn_state)[0]
                if len(data_summary) == 0:
                    action_mean = action_distribution.mean()
                else:
                    action_mean = dataset.undo_data_normalization_sample_wise(action_distribution.mean(), data_summary)
                parameter_values['mean'] = tf.squeeze(action_mean).numpy()
                action_variance = calculate_action_variance(action_distribution, action_mean, data_summary)
                parameter_values['variance'] = tf.squeeze(action_variance).numpy()
                write_to_csv_file(parameter_values, step, log_dir, prefix)

        if 'mae' in metrics:
            total_mae += episode_mae / step_counter
        if 'mse' in metrics:
            total_mse += episode_mse / step_counter
        if 'rmse' in metrics:
            total_rmse += tf.math.sqrt(episode_rmse / step_counter)

    if 'mae' in metrics:
        avg_mae = total_mae / num_iter
        logging.info("[{}] MAE (step {}): {}".format(prefix, step, avg_mae))
    if 'mse' in metrics:
        avg_mse = total_mse / num_iter
        logging.info("[{}] MSE (step {}): {}".format(prefix, step, avg_mse))
    if 'rmse' in metrics:
        avg_rmse = total_rmse / num_iter
        logging.info("[{}] RMSE (step {}): {}".format(prefix, step, avg_rmse))
    return tf.squeeze(avg_mae), tf.squeeze(avg_mse), tf.squeeze(avg_rmse)


def calculate_action_variance(action_distribution, mean, data_summary, num_steps=100):
    variance = 0.0
    for _ in range(num_steps):
        if len(data_summary) == 0:
            s = action_distribution.sample()
        else:
            s = dataset.undo_data_normalization_sample_wise(action_distribution.sample(), data_summary)
        variance += (s - mean) ** 2

    variance = variance / num_steps
    return variance


def write_to_csv_file(parameter_values, step, log_dir, prefix):
    fieldnames = []
    for name in parameter_values.keys():
        fieldnames.append(name)

    if not os.path.isdir(log_dir + "/data_summaries"):
        os.makedirs(log_dir + "/data_summaries")

    with open(log_dir + "/data_summaries/performance_summary_" + str(step) + "_" + prefix + ".csv", 'a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(parameter_values)
