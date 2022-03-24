import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_preds_vs_ground_truth_single_step(log_dir, env, agent, total_time_h, max_attribute_val, step, prefix="eval",
                                           use_rnn_state=True):
    fig, ax = plt.subplots()
    preds, ground_truth = [], []
    time_step = env.reset()
    rnn_state = agent.policy.get_initial_state(batch_size=1)

    while not time_step.is_last():
        action_step, rnn_state, _ = agent.policy.action(time_step, rnn_state)
        if use_rnn_state:
            time_step = env.step(action_step, rnn_state)
        else:
            time_step = env.step(action_step)
        preds.append(tf.squeeze(action_step))
        ground_truth.append(time_step.reward)

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