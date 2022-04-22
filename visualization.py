import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from data import dataset
from absl import logging


def plot_preds_vs_ground_truth_single_step(log_dir, env, agent, total_time_h, max_attribute_val, step,
                                           env_implementation, data_summary, prefix="eval", use_rnn_state=True):
    fig, ax = plt.subplots()
    preds, ground_truth = [], []
    time_step = env.reset()
    rnn_state = agent.policy.get_initial_state(batch_size=1)

    while not time_step.is_last():
        action_step, rnn_state, _ = agent.policy.action(time_step, rnn_state)
        if use_rnn_state:
            if env_implementation == "tf":
                time_step = env.step(action_step, rnn_state)
            elif env_implementation == "gym":
                time_step = env.step(action_step)
        else:
            time_step = env.step(action_step)
        if len(data_summary) == 0:
            preds.append(tf.squeeze(action_step))
            ground_truth.append(time_step.reward)
        else:
            preds.append(dataset.undo_data_normalization_sample_wise(tf.squeeze(action_step), data_summary))
            ground_truth.append(dataset.undo_data_normalization_sample_wise(time_step.reward, data_summary))
    x_values = np.linspace(start=0, stop=total_time_h, num=len(preds))
    ax.plot(x_values, ground_truth, color='green', label="ground_truth")
    ax.plot(x_values, preds, color='blue', label="rl_prediction")

    plt.legend(loc='upper right')
    plt.xlabel("Measurement time in hours")
    plt.ylabel("Blood glucose values")
    plt.ylim([0.0, max_attribute_val + (max_attribute_val / 4)])
    if not os.path.isdir(log_dir + "/visualization"):
        os.makedirs(log_dir + "/visualization")
    plt.savefig(log_dir + "/visualization/preds_vs_ground_truth_" + str(step) + "_" + prefix + ".pdf", dpi=300)
    plt.close()


def plot_preds_vs_ground_truth_multi_step(log_dir, env, agent, total_time_h, max_attribute_val, step,
                                          env_implementation, data_summary, ts_data, pred_horizon, prefix="eval",
                                          use_rnn_state=True):
    fig, ax = plt.subplots()
    preds, ground_truth = [], []
    time_step = env.reset()
    rnn_state = agent.policy.get_initial_state(batch_size=1)

    while not time_step.is_last():
        action_step, rnn_state, _ = agent.policy.action(time_step, rnn_state)
        # ground_truth_val = env._current_ground_truth
        ground_truth_pos = int(tf.squeeze(env._current_data_pos))
        if use_rnn_state:
            if env_implementation == "tf":
                time_step = env.step(action_step, rnn_state)
            elif env_implementation == "gym":
                time_step = env.step(action_step)
        else:
            time_step = env.step(action_step)
        if len(data_summary) == 0:
            preds.append(tf.squeeze(action_step))
            # ground_truth_pos = int(tf.squeeze(time_step.reward))
            # ground_truth_val = ts_data[ground_truth_pos - pred_horizon:ground_truth_pos]
            ground_truth_val = ts_data[ground_truth_pos:ground_truth_pos + pred_horizon]
            ground_truth.append(ground_truth_val)
        else:
            preds.append(dataset.undo_data_normalization_sample_wise(tf.squeeze(action_step), data_summary))
            # ground_truth_pos = int(tf.squeeze(time_step.reward))
            # ground_truth_val = ts_data[ground_truth_pos - pred_horizon:ground_truth_pos]
            ground_truth_val = ts_data[ground_truth_pos:ground_truth_pos + pred_horizon]
            ground_truth.append(dataset.undo_data_normalization_sample_wise(tf.squeeze(ground_truth_val), data_summary))
    preds = tf.concat(preds, -1)
    ground_truth = tf.concat(ground_truth, -1)
    logging.info("Num pred: {}, Num ground truth: {}, Len data set: {}".format(preds.shape[0],
                                                                               ground_truth.shape[0],
                                                                               len(ts_data)))
    x_values = np.linspace(start=0, stop=total_time_h, num=len(preds))
    ax.plot(x_values, ground_truth, color='green', label="ground_truth")
    ax.plot(x_values, preds, color='blue', label="rl_prediction")

    plt.legend(loc='upper right')
    plt.xlabel("Measurement time in hours")
    plt.ylabel("Blood glucose values")
    plt.ylim([0.0, max_attribute_val + (max_attribute_val / 4)])
    if not os.path.isdir(log_dir + "/visualization"):
        os.makedirs(log_dir + "/visualization")
    plt.savefig(log_dir + "/visualization/preds_vs_ground_truth_" + str(step) + "_" + prefix + ".pdf", dpi=300)
    plt.close()
