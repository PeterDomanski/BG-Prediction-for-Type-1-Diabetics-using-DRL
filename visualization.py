import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_preds_vs_ground_truth_single_step(log_dir, env, agent, total_time_h, step):
    fig, ax = plt.subplots()
    preds, ground_truth = [], []
    time_step = env.reset()
    rnn_state = agent.policy.get_initial_state(batch_size=1)

    while not time_step.is_last():
        action_step, rnn_state, _ = agent.policy.action(time_step, rnn_state)
        time_step = env.step(action_step)
        preds.append(tf.squeeze(action_step))
        ground_truth.append(time_step.reward)

    x_values = np.linspace(start=0, stop=total_time_h, num=len(preds))
    ax.plot(x_values, ground_truth, color='green')
    ax.plot(x_values, preds, color='blue')

    plt.xlabel("Measurement time in hours")
    plt.ylabel("Blood glucose values")
    if not os.path.isdir(log_dir + "/visualization"):
        os.makedirs(log_dir + "/visualization")
    plt.savefig(log_dir + "/visualization/preds_vs_ground_truth_" + str(step) + ".pdf", dpi=300)
    plt.close()