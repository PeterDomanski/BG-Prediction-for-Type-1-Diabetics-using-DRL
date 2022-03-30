import tensorflow as tf
from data import dataset


def compute_avg_return(env, policy, data_summary, num_iter=16, normalize=False, use_rnn_state=True):
    total_return = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_return = 0.0
        time_series_counter = 0

        while not time_step.is_last():
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            if use_rnn_state:
                time_step = env.step(action_step, rnn_state)
            else:
                time_step = env.step(action_step)
            if len(data_summary) == 0:
                episode_return += time_step.reward
            else:
                episode_return += dataset.undo_data_normalization_sample_wise(time_step.reward, data_summary)
            time_series_counter += 1

        if normalize:
            total_return += (episode_return / time_series_counter)
        else:
            total_return += episode_return

    avg_return = total_return / num_iter
    return tf.squeeze(avg_return)


def compute_metrics_single_step(env, policy, data_summary, metrics=['mae', 'mse', 'rmse'], num_iter=1, use_rnn_state=True):
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
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            if use_rnn_state:
                time_step = env.step(action_step, rnn_state)
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
                episode_mae += tf.math.abs(agent_pred - ground_truth)
            if 'mse' in metrics:
                episode_mse += (agent_pred - ground_truth) ** 2
            if 'rmse' in metrics:
                episode_rmse += (agent_pred - ground_truth) ** 2

        if 'mae' in metrics:
            total_mae += episode_mae / step_counter
        if 'mse' in metrics:
            total_mse += episode_mse / step_counter
        if 'rmse' in metrics:
            total_rmse += tf.math.sqrt(episode_rmse / step_counter)

    if 'mae' in metrics:
        avg_mae = total_mae / num_iter
    if 'mae' in metrics:
        avg_mse = total_mse / num_iter
    if 'mae' in metrics:
        avg_rmse = total_rmse / num_iter
    return tf.squeeze(avg_mae), tf.squeeze(avg_mse), tf.squeeze(avg_rmse)


def compute_mae_single_step(env, policy, num_iter=1):
    total_mae = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_mae = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth = time_step.reward
            episode_mae += tf.math.abs(agent_pred - ground_truth)

        total_mae += episode_mae / step_counter

    avg_mae = total_mae / num_iter
    return tf.squeeze(avg_mae)


def compute_mse_single_step(env, policy, num_iter=1):
    total_mse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_mse = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth = time_step.reward
            episode_mse += (agent_pred - ground_truth) ** 2

        total_mse += episode_mse / step_counter

    avg_mse = total_mse / num_iter
    return tf.squeeze(avg_mse)


def compute_rmse_single_step(env, policy, num_iter=1):
    total_rmse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_rmse = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth = time_step.reward
            episode_rmse += (agent_pred - ground_truth) ** 2

        total_rmse += tf.math.sqrt(episode_rmse / step_counter)

    avg_rmse = total_rmse / num_iter
    return tf.squeeze(avg_rmse)


def compute_mae_multi_step(env, policy, ts_eval_data, forecasting_steps, num_iter=1):
    total_mae = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_mae = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth_pos = int(tf.squeeze(time_step.reward))
            ground_truth = ts_eval_data[ground_truth_pos - forecasting_steps:ground_truth_pos]
            episode_mae += tf.math.reduce_mean(tf.math.abs(agent_pred - ground_truth))

        total_mae += episode_mae / step_counter

    avg_mae = total_mae / num_iter
    return tf.squeeze(avg_mae)


def compute_mse_multi_step(env, policy, ts_eval_data, forecasting_steps, num_iter=1):
    total_mse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_mse = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth_pos = int(tf.squeeze(time_step.reward))
            ground_truth = ts_eval_data[ground_truth_pos - forecasting_steps:ground_truth_pos]
            episode_mse += tf.math.reduce_mean((agent_pred - ground_truth) ** 2)

        total_mse += episode_mse / step_counter

    avg_mse = total_mse / num_iter
    return tf.squeeze(avg_mse)


def compute_rmse_multi_step(env, policy, ts_eval_data, forecasting_steps, num_iter=1):
    total_rmse = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_rmse = 0.0
        step_counter = 0

        while not time_step.is_last():
            step_counter += 1
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            # agent forecast
            agent_pred = tf.squeeze(action_step)
            ground_truth_pos = int(tf.squeeze(time_step.reward))
            ground_truth = ts_eval_data[ground_truth_pos - forecasting_steps:ground_truth_pos]
            episode_rmse += tf.math.reduce_mean((agent_pred - ground_truth) ** 2)

        total_rmse += tf.math.sqrt(episode_rmse / step_counter)

    avg_rmse = total_rmse / num_iter
    return tf.squeeze(avg_rmse)
