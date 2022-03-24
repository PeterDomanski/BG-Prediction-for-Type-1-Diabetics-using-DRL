import logging
import gin
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy
from rl import tf_driver
import tensorflow as tf
import evaluation
import visualization


@gin.configurable
def rl_training_loop(log_dir, train_env, eval_env, agent, ts_eval_data, file_writer, setup, forecasting_steps,
                     rl_algorithm, total_time_h, max_attribute_val, env_implementation, max_train_steps=1000,
                     eval_interval=100, preheat_phase=False):
    on_policy_algorithms = ["reinforce", "ppo"]
    # replay buffer for data collection
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
            collect_driver = tf_driver.TrainingDriver(agent, train_env, replay_buffer, rl_algorithm, batch_size=128)
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
            logging.info("collect a few steps using collect_policy and save to the replay buffer before training")
            for _ in range(1000):
                collect_driver.collect_step()

    for i in range(max_train_steps + 1):
        logging.info("start training")
        if i % eval_interval == 0:
            avg_return = evaluation.compute_avg_return(train_env, agent.policy)
            with file_writer.as_default():
                tf.summary.scalar("Average Return", avg_return, i)
            if setup == "single_step":
                # TODO: evaluate on both training (difficult to get ground truth) and evaluation environment
                # avg_mae = evaluation.compute_mae_single_step(eval_env, agent.policy)
                # avg_mse = evaluation.compute_mse_single_step(eval_env, agent.policy)
                # avg_rmse = evaluation.compute_rmse_single_step(eval_env, agent.policy)
                avg_mae, avg_mse, avg_rmse = evaluation.compute_metrics_single_step(eval_env, agent.policy)
                # visualization of (scalar) attribute of interest
                visualization.plot_preds_vs_ground_truth_single_step(log_dir, eval_env, agent, total_time_h,
                                                                     max_attribute_val, i)
            elif setup == "mutli_step":
                avg_mae = evaluation.compute_mae_multi_step(eval_env, agent.policy, ts_eval_data, forecasting_steps)
                avg_mse = evaluation.compute_mse_multi_step(eval_env, agent.policy, ts_eval_data, forecasting_steps)
                avg_rmse = evaluation.compute_rmse_multi_step(eval_env, agent.policy, ts_eval_data, forecasting_steps)
            else:
                logging.info("Setup {} not supported".format(setup))
            with file_writer.as_default():
                tf.summary.scalar("Average MAE", avg_mae, i)
                tf.summary.scalar("Average MSE", avg_mse, i)
                tf.summary.scalar("Average RMSE", avg_rmse, i)
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