import tensorflow as tf


def compute_avg_return(env, policy, num_iter=16):
    total_return = 0.0
    for _ in range(num_iter):
        time_step = env.reset()
        rnn_state = policy.get_initial_state(batch_size=1)
        episode_return = 0.0
        time_series_counter = 0

        while not time_step.is_last():
            action_step, rnn_state, _ = policy.action(time_step, rnn_state)
            time_step = env.step(action_step)
            episode_return += time_step.reward
            time_series_counter += 1

        total_return += (episode_return / time_series_counter)

    avg_return = total_return / num_iter
    return tf.squeeze(avg_return)
