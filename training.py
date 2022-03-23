import logging
import gin
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
import tensorflow as tf
import evaluation
import visualization


@gin.configurable
def rl_training_loop(log_dir, train_env, eval_env, agent, ts_eval_data, file_writer, setup, forecasting_steps,
                     rl_algorithm, total_time_h, max_attribute_val, max_train_steps=1000, eval_interval=100):
    on_policy_algorithms = ["reinforce", "ppo"]
    # replay buffer for data collection
    replay_buffer = get_replay_buffer(agent, batch_size=1)
    # create driver for data collection
    if rl_algorithm not in on_policy_algorithms:
        collect_driver = get_collect_driver(train_env,
                                            agent.collect_policy,
                                            [replay_buffer.add_batch],
                                            num_iter=64,
                                            driver_type="step")
    else:
        collect_driver = get_collect_driver(train_env,
                                            agent.collect_policy,
                                            [replay_buffer.add_batch],
                                            num_iter=16,
                                            driver_type="episode")

    def collect_step(env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        # next_time_step = env.step(action_step.action, policy_state=action_step.state)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    # TODO: initial ('offline') data collection -> off-policy algorithms(they can also use step drivers)
    # if rl_algorithm not in ["reinforce", "ppo"]:
    #     init_collect_driver = get_collect_driver(
    #         train_env,
    #         random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()),
    #         [replay_buffer.add_batch],
    #         num_iter=64,
    #         driver_type="step"
    #     )
    #     for _ in range(100):
    #         init_collect_driver.run()
    # TODO: replay buffer as dataset
    # dataset = replay_buffer.as_dataset()
    # iterator = iter(dataset)

    for i in range(max_train_steps + 1):
        if i % eval_interval == 0:
            avg_return = evaluation.compute_avg_return(train_env, agent.policy)
            if setup == "single_step":
                # TODO: evaluate on both training (difficult to get ground truth) and evaluation environment
                avg_mae = evaluation.compute_mae_single_step(eval_env, agent.policy)
                avg_mse = evaluation.compute_mse_single_step(eval_env, agent.policy)
                avg_rmse = evaluation.compute_rmse_single_step(eval_env, agent.policy)
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
                tf.summary.scalar("Average Return", avg_return, i)
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
        collect_driver.run()
        # for _ in range(1):
        #     collect_step(train_env, agent.policy)
        # experience, _ = next(iterator)
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        # keep track of actor loss
        if i % eval_interval == 0:
            with file_writer.as_default():
                tf.summary.scalar("Actor Loss", train_loss.loss, i)

        if rl_algorithm in on_policy_algorithms:
            replay_buffer.clear()


@gin.configurable
def get_replay_buffer(agent, batch_size=128, max_buffer_length=8192):
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