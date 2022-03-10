import gin
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
import tensorflow as tf
import evaluation


def rl_training_loop(train_env, agent, file_writer, max_train_steps=1000, eval_interval=100):
    # replay buffer for data collection
    replay_buffer = get_replay_buffer(agent, batch_size=1)
    # create driver for data collection
    collect_driver = get_collect_driver(train_env,
                                        agent.collect_policy,
                                        [replay_buffer.add_batch],
                                        num_iter=8,
                                        driver_type="episode")
    for i in range(max_train_steps):
        if i % eval_interval == 0:
            avg_return = evaluation.compute_avg_return(train_env, agent.policy)
            with file_writer.as_default():
                tf.summary.scalar("Average Return", avg_return, i)
        collect_driver.run()
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        # keep track of actor and critic loss
        actor_loss = train_loss.loss
        with file_writer.as_default():
            tf.summary.scalar("Actor Loss", actor_loss, i)
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
