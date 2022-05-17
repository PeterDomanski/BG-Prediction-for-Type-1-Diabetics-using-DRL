import logging
import gin
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy
from rl import tf_driver
import tensorflow as tf
import evaluation
import visualization
import numpy as np
import os


@gin.configurable
def rl_training_loop(log_dir, train_env, train_env_eval, eval_env, eval_env_train, agent, ts_train_data, ts_eval_data,
                     file_writer, setup, forecasting_steps, rl_algorithm, total_train_time_h, total_eval_time_h,
                     max_attribute_val, num_iter, data_summary, env_implementation, multi_task,
                     max_train_steps=1000, eval_interval=100, preheat_phase=False, restore_dir=""):
    # train_env_eval (train env with ground truth as reward) and
    # eval_env_train (eval env with standard reward definition) are for validation purposes
    on_policy_algorithms = ["reinforce", "ppo"]
    # replay buffer for data collection
    # TODO: reverb replay buffer
    if rl_algorithm in on_policy_algorithms:
        replay_buffer = get_replay_buffer(agent, batch_size=1, max_buffer_length=2000)
    else:
        replay_buffer = get_replay_buffer(agent, batch_size=1, max_buffer_length=10000)
    # create driver for data collection
    if rl_algorithm not in on_policy_algorithms:
        if env_implementation == "tf":
            collect_driver = tf_driver.TrainingDriver(agent, train_env, replay_buffer, rl_algorithm, batch_size=128)
        else:
            collect_driver = get_collect_driver(train_env,
                                                agent.collect_policy,
                                                [replay_buffer.add_batch],
                                                num_iter=64,
                                                driver_type="step")
    else:
        if env_implementation == "tf":
            # num iter has to be the length of an episode for on-policy algorithms
            # TODO: on-policy algorithm need episode driver
            collect_driver = tf_driver.TrainingDriver(agent, train_env, replay_buffer, rl_algorithm, batch_size=128,
                                                      num_iterations=num_iter)
        else:
            collect_driver = get_collect_driver(train_env,
                                                agent.collect_policy,
                                                [replay_buffer.add_batch],
                                                num_iter=16,
                                                driver_type="episode")

    if rl_algorithm not in on_policy_algorithms:
        if preheat_phase:
            # pre-training collection of experience with initial / random actor
            # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
            # TODO: random policy only possible if we do not use RNN state
            collect_driver = tf_driver.TrainingDriver(agent, train_env, replay_buffer, rl_algorithm, batch_size=128)
            logging.info("Collect a few steps using collect_policy and save to the replay buffer before training")
            for _ in range(1000):
                collect_driver.collect_step()

    best_avg_rmse = float("inf")
    ds_index = 0
    if restore_dir != "":
        logging.info("Restore model(s) from {}".format(restore_dir))
        if rl_algorithm == "sac":
            restore_network_parameters(agent._actor_network, os.path.join(restore_dir, "actor_network"))
            restore_network_parameters(agent._critic_network_1, os.path.join(restore_dir, "critic_network_1"))
            restore_network_parameters(agent._critic_network_2, os.path.join(restore_dir, "critic_network_2"))
            restore_network_parameters(agent._target_critic_network_1,
                                       os.path.join(restore_dir, "target_critic_network_1"))
            restore_network_parameters(agent._target_critic_network_2,
                                       os.path.join(restore_dir, "target_critic_network_2"))
        else:
            logging.info("Model restoring not implemented yet for {}".format(rl_algorithm))
    for i in range(max_train_steps + 1):
        logging.info("Start training iteration {}".format(i))
        if i % eval_interval == 0:
            # compute average return on train data
            avg_return_train = evaluation.compute_avg_return(train_env, agent.policy, env_implementation, data_summary)
            # compute average return on eval data
            avg_return_eval = evaluation.compute_avg_return(eval_env_train, agent.policy, env_implementation,
                                                            data_summary)
            with file_writer.as_default():
                tf.summary.scalar("Average Return (Training)", avg_return_train, i)
                tf.summary.scalar("Average Return (Evaluation)", avg_return_eval, i)
            if setup == "single_step":
                # evaluation on train data set
                avg_mae_train, avg_mse_train, avg_rmse_train = evaluation.compute_metrics_single_step(
                    train_env_eval, agent.policy, env_implementation, data_summary, i, log_dir, prefix="train")
                # visualization of (scalar) attribute of interest on train data set
                visualization.plot_preds_vs_ground_truth_single_step(
                    log_dir, train_env_eval, agent, total_train_time_h, max_attribute_val, i, env_implementation,
                    data_summary, prefix="train")
                # evaluation on eval data set
                avg_mae_eval, avg_mse_eval, avg_rmse_eval = evaluation.compute_metrics_single_step(
                    eval_env, agent.policy, env_implementation, data_summary, i, log_dir, prefix="eval")
                # visualization of (scalar) attribute of interest on eval data set
                visualization.plot_preds_vs_ground_truth_single_step(
                    log_dir, eval_env, agent, total_eval_time_h, max_attribute_val, i, env_implementation,
                    data_summary, prefix="eval")
            elif setup == "multi_step":
                if multi_task:
                    avg_mae_train, avg_mse_train, avg_rmse_train = evaluation.compute_metrics_multi_step(
                        train_env_eval, agent.policy, env_implementation, data_summary, ts_train_data[ds_index],
                        forecasting_steps, i, log_dir, prefix="train")
                    visualization.plot_preds_vs_ground_truth_multi_step(
                        log_dir, train_env_eval, agent, total_train_time_h, max_attribute_val, i, env_implementation,
                        data_summary, ts_train_data[ds_index], forecasting_steps, prefix="train")
                    avg_mae_eval, avg_mse_eval, avg_rmse_eval = evaluation.compute_metrics_multi_step(
                        eval_env, agent.policy, env_implementation, data_summary, ts_eval_data[ds_index],
                        forecasting_steps, i, log_dir, prefix="eval")
                    visualization.plot_preds_vs_ground_truth_multi_step(
                        log_dir, eval_env, agent, total_eval_time_h, max_attribute_val, i, env_implementation,
                        data_summary, ts_eval_data[ds_index], forecasting_steps, prefix="eval")
                else:
                    avg_mae_train, avg_mse_train, avg_rmse_train = evaluation.compute_metrics_multi_step(
                        train_env_eval, agent.policy, env_implementation, data_summary, ts_train_data,
                        forecasting_steps, i, log_dir, prefix="train")
                    visualization.plot_preds_vs_ground_truth_multi_step(
                        log_dir, train_env_eval, agent, total_train_time_h, max_attribute_val, i, env_implementation,
                        data_summary, ts_train_data, forecasting_steps, prefix="train")
                    avg_mae_eval, avg_mse_eval, avg_rmse_eval = evaluation.compute_metrics_multi_step(
                        eval_env, agent.policy, env_implementation, data_summary, ts_eval_data, forecasting_steps, i,
                        log_dir, prefix="eval")
                    visualization.plot_preds_vs_ground_truth_multi_step(
                        log_dir, eval_env, agent, total_eval_time_h, max_attribute_val, i, env_implementation,
                        data_summary, ts_eval_data, forecasting_steps, prefix="eval")
            else:
                logging.info("Setup {} not supported".format(setup))
            with file_writer.as_default():
                tf.summary.scalar("Average MAE (Training)", avg_mae_train, i)
                tf.summary.scalar("Average MSE (Training)", avg_mse_train, i)
                tf.summary.scalar("Average RMSE (Training)", avg_rmse_train, i)
                tf.summary.scalar("Average MAE (Evaluation)", avg_mae_eval, i)
                tf.summary.scalar("Average MSE (Evaluation)", avg_mse_eval, i)
                tf.summary.scalar("Average RMSE (Evaluation)", avg_rmse_eval, i)
            # keep track of actor network parameters
            with file_writer.as_default():
                if rl_algorithm == "ppo":
                    actor_net = agent._actor_net
                elif rl_algorithm == "reinforce" or \
                        rl_algorithm == "ddpg" or \
                        rl_algorithm == "sac" or \
                        rl_algorithm == "td3":
                    actor_net = agent._actor_network
                elif rl_algorithm == "dqn":
                    actor_net = agent._q_network
                for actor_var in actor_net.trainable_variables:
                    tf.summary.histogram(actor_var.name, actor_var, i)
                if rl_algorithm == "sac":
                    for critic_var in agent._critic_network_1.trainable_variables:
                        tf.summary.histogram(critic_var.name, critic_var, i)
            # save parameters of all relevant networks
            if avg_rmse_eval < best_avg_rmse:
                logging.info("Save model (step: {})".format(i))
                if rl_algorithm == "sac":
                    save_network_parameters(log_dir, actor_net, "actor_network")
                    save_network_parameters(log_dir, agent._critic_network_1, "critic_network_1")
                    save_network_parameters(log_dir, agent._critic_network_2, "critic_network_2")
                    save_network_parameters(log_dir, agent._target_critic_network_1, "target_critic_network_1")
                    save_network_parameters(log_dir, agent._target_critic_network_2, "target_critic_network_2")
                else:
                    logging.info("Model saving not implemented yet for {}".format(rl_algorithm))
                best_avg_rmse = avg_rmse_eval

        if env_implementation == "tf":
            train_loss = collect_driver.train_step()
        else:
            collect_driver.run()
            experience = replay_buffer.gather_all()
            train_loss = agent.train(experience)

        # keep track of actor loss
        if i % eval_interval == 0:
            with file_writer.as_default():
                tf.summary.scalar("Actor Loss", train_loss.loss, i)
            if multi_task:
                if env_implementation == "tf":
                    ds_index = np.random.randint(low=0, high=12)
                    train_env.set_patient_dataset(ds_index)
                    train_env_eval.set_patient_dataset(ds_index)
                    eval_env.set_patient_dataset(ds_index)
                    eval_env_train.set_patient_dataset(ds_index)
                    logging.info("Training / Evaluation on patient {}".format(ds_index))
                else:
                    logging.info("Patient generalization only implemented for tf environments")

        if env_implementation != "tf":
            replay_buffer.clear()


@gin.configurable
def get_replay_buffer(agent, batch_size=128, max_buffer_length=2000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                          batch_size=batch_size,
                                                          max_length=max_buffer_length)


@gin.configurable
def get_collect_driver(env, policy, observers, num_iter=128, driver_type="episode"):
    if driver_type == "step":
        return dynamic_step_driver.DynamicStepDriver(env,
                                                     policy,
                                                     observers=observers,
                                                     num_steps=num_iter)
    elif driver_type == "episode":
        return dynamic_episode_driver.DynamicEpisodeDriver(env,
                                                           policy,
                                                           observers=observers,
                                                           num_episodes=num_iter)


def save_network_parameters(log_dir, net, net_name=""):
    if not os.path.isdir(log_dir + "/w&b/" + net_name):
        os.makedirs(log_dir + "/w&b/" + net_name)
    for i, v in enumerate(net.trainable_variables):
        name = ""
        for n in v.name.split("/")[1:]:
            name += (n + "_")
        name = name[:-1]
        with open(log_dir + "/w&b/" + net_name + "/0" + str(i) + "_" + name + ".txt", "w") as weights_file:
            val = v.numpy()
            if len(val.shape) == 2:
                for row in val:
                    np.savetxt(weights_file, row)
            elif len(val.shape) == 1:
                np.savetxt(weights_file, val)
        # weights_file.close()


def restore_network_parameters(net, restore_dir, layers_to_train="last"):
    param_files = sorted(os.listdir(restore_dir), key=lambda index: int(index.split("_")[0]))
    network_parameters = []
    for i, file in enumerate(param_files):
        param_val = np.loadtxt(os.path.join(restore_dir, file)).reshape(net.trainable_variables[i].shape)
        network_parameters.append(param_val)
    net.set_weights(network_parameters)
    if layers_to_train == "last":
        logging.info("Train parameters of {} layer in network {}".format(layers_to_train, net.name))
        if "Actor" in net.name:
            net.layers[0].trainable = False
        elif "Critic" in net.name:
            for i, nl in enumerate(net.layers):
                if i < len(net.layers) - 1:
                    nl.trainable = False
                else:
                    nl.trainable = True
    elif layers_to_train == "dec_last":
        logging.info("Train parameters of {} layers in network {}".format(layers_to_train, net.name))
        if "Actor" in net.name:
            for nl in net.layers[0].layers:
                if "Encoding" in nl.name or "dynamic_unroll" in nl.name:
                    nl.trainable = False
        elif "Critic" in net.name:
            trainable = False
            for nl in net.layers:
                nl.trainable = trainable
                if "dynamic_unroll" in nl.name:
                    trainable = True
    elif layers_to_train == "lstm_dec_last":
        logging.info("Train parameters of {} layers in network {}".format(layers_to_train, net.name))
        if "Actor" in net.name:
            for net_layer in net.layers[0].layers:
                if "Encoding" in net_layer.name:
                    net_layer.trainable = False
        elif "Critic" in net.name:
            trainable = False
            for nl in net.layers:
                if "dynamic_unroll" in nl.name:
                    trainable = True
                nl.trainable = trainable
    else:
        logging.info("Train all parameters in network {}".format(net.name))

