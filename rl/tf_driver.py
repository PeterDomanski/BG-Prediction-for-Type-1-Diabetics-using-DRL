import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class TrainingDriver:
    def __init__(self, agent, env, replay_buffer, batch_size=64, num_iterations=10, use_rnn_state=True):
        env.reset()
        self.num_iterations = num_iterations
        self.use_rnn_state = use_rnn_state
        self.env = env
        self.replay_buffer = replay_buffer
        self.dataset = self.replay_buffer.as_dataset(sample_batch_size=64, single_deterministic_pass=False)
        self.iterator = iter(self.dataset)
        agent.train = common.function(agent.train)
        self.agent = agent
        self.step = common.create_variable("step", 0, dtype=tf.int64)

    # tf.function
    def collect_step(self):
        time_step = self.env.current_time_step()
        action_step = self.agent.collect_policy.action(time_step)
        if self.use_rnn_state:
            next_time_step = self.env.step(action_step.action, policy_state=action_step.state)
        else:
            next_time_step = self.env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    # @tf.function
    def train_step(self):
        for _ in range(self.num_iterations):
            self.collect_step()
        experience, _ = next(self.iterator)
        # TODO: check why this is necessary
        experience = tf.nest.map_structure(lambda e: tf.expand_dims(e, 0), experience)
        train_loss = self.agent.train(experience=experience)
        # TODO: check if we need to clear replay buffer, especially for offline RL algorithms
        self.replay_buffer.clear()
        return train_loss

