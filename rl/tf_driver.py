import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils import common


class TrainingDriver:
    def __init__(self, agent, environment, replay_buffer, replay_observer, num_iterations):
        self.train_losses = common.create_variable('train_losses', shape=(num_iterations,), dtype=tf.float32)
        self.returns = common.create_variable('returns', shape=(num_iterations,), dtype=tf.float32)
        self.driver = DynamicEpisodeDriver(environment, agent.collect_policy, replay_observer)
        self.train_env = environment
        self.replay_buffer = replay_buffer
        agent.train = common.function(agent.train)
        self.agent = agent
        self.step = common.create_variable("step", 0, dtype=tf.int64)

    @tf.function
    def train_step(self):

        self.driver.run()

        iterator = iter(self.replay_buffer.as_dataset(single_deterministic_pass=False,
                                                      sample_batch_size=self.train_env.batch_size,
                                                      num_steps=self.train_env.episode_length))
        experience, _ = next(iterator)
        loss = self.agent.train(experience=experience).loss
        avg_return = tf.divide(tf.reduce_sum(experience.reward), self.train_env.batch_size)
        normalized_avg_return = tf.divide(avg_return, tf.cast(self.train_env.episode_length, tf.float32))

        self.replay_buffer.clear()

        self.returns[self.step].assign(normalized_avg_return)
        self.train_losses[self.step].assign(loss)
        self.step.assign_add(1)
        return normalized_avg_return, loss

    def get_summary(self):
        return self.returns[:self.step], self.train_losses[:self.step]
