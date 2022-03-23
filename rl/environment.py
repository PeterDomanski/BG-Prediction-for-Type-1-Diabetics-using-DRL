import gym
import numpy as np
import tensorflow as tf
from absl import logging
from tf_agents import specs
from tf_agents.utils import common
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from gym.spaces import Box
import gin


FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


@gin.configurable
class TsForecastingSingleStepTFEnv(tf_environment.TFEnvironment):
    def __init__(self, ts_data, initial_state_val=100.0, window_size=5, max_window_count=-1, min_attribute_val=35.0,
                 max_attribute_val=500.0, batch_size=1, evaluation=False, dtype=tf.float32, scope='TFEnviroment'):
        self._dtype = dtype
        self._scope = scope
        self._batch_size = batch_size
        self.ts_data = ts_data
        self.evaluation = evaluation
        self.window_size = window_size
        self.max_window_count = max_window_count
        self.min_attribute_val = min_attribute_val
        self.max_attribute_val = max_attribute_val
        self._initial_state = tf.cast([initial_state_val] * window_size, dtype=dtype)
        observation_spec = specs.BoundedTensorSpec((5, ), dtype,
                                                   # minimum=[min_attribute_val] * window_size,
                                                   minimum=min_attribute_val,
                                                   maximum=max_attribute_val,
                                                   # maximum=[max_attribute_val] * window_size,
                                                   name='observation')
        action_spec = specs.BoundedTensorSpec(shape=(1, ), dtype=dtype,
                                              minimum=min_attribute_val, maximum=max_attribute_val, name='action')
        reward_spec = specs.BoundedTensorSpec((), dtype, minimum=-465, maximum=0, name='reward')
        discount_spec = specs.BoundedTensorSpec((), dtype, minimum=0, maximum=1, name='discount')
        step_type_spec = specs.BoundedTensorSpec((), tf.int32, minimum=0, maximum=1, name='step_type')

        time_step_spec = ts.TimeStep(step_type=step_type_spec,
                                     reward=reward_spec,
                                     discount=discount_spec,
                                     observation=observation_spec)

        # time_step_spec = ts.time_step_spec(observation_spec)

        super(TsForecastingSingleStepTFEnv, self).__init__(time_step_spec, action_spec, batch_size=batch_size)
        self._window_counter = common.create_variable('window_counter', shape=(), dtype=tf.int64)
        self._current_ground_truth = common.create_variable('current_ground_truth', shape=(), dtype=dtype)
        self._current_data_pos = common.create_variable('current_data_pos', shape=(), dtype=dtype)
        self._state = common.create_variable('state', initial_state_val, shape=(window_size, ),
                                             dtype=dtype)
        self._reward = common.create_variable('reward', 0, dtype=dtype)
        self._steps = common.create_variable('steps', 0)
        self._resets = common.create_variable('resets', 0)

    def _current_time_step(self):
        def first():
            return tf.constant(FIRST, dtype=tf.int32)

        def mid():
            return tf.constant(MID, dtype=tf.int32)

        def last():
            return tf.constant(LAST, dtype=tf.int32)

        if self.max_window_count != -1:
            step_type = tf.case(
                [(tf.equal(self._steps, 0), first),
                 (tf.equal(self._steps, self.max_window_count - 1), last)],
                default=mid)
        else:
            step_type = tf.case(
                [(tf.equal(self._steps, 0), first),
                 (tf.equal(self._steps, len(self.ts_data) - 1), last)],
                default=mid)

        # no discounts
        # discount = tf.ones(shape=(self._batch_size, ))
        discount = tf.ones(shape=())

        # step_type = tf.tile((step_type,), (self.batch_size, ), name='step_type')
        return ts.TimeStep(tf.expand_dims(step_type, 0), tf.expand_dims(self._reward, 0), tf.expand_dims(discount, 0),
                           tf.expand_dims(self._state, 0))
        # return ts.TimeStep(step_type, self._reward, discount, self._state)

    def _reset(self):
        self._window_counter.assign(0)
        self._steps.assign(0)
        self._resets.assign_add(1)
        if self.evaluation:
            self._current_data_pos.assign(0)
        else:
            # random value for staring position
            pos = np.random.randint(low=0, high=len(self.ts_data) - (2 * self.window_size + 1))
            self._current_data_pos.assign(pos)
        pos = int(tf.squeeze(self._current_data_pos))
        self._state.assign(self.ts_data[pos:pos + self.window_size].values)
        self._current_data_pos.assign_add(self.window_size)
        self._current_ground_truth.assign(self.ts_data[int(tf.squeeze(self._current_data_pos))])
        self._current_data_pos.assign_add(1)
        self._window_counter.assign_add(1)
        return self.current_time_step()

    def step(self, action, policy_state=None):
        return self._step(action, policy_state=policy_state)

    # policy state is state of RNN (actor network)
    def _step(self, action, policy_state=None):
        if self.evaluation:
            self._reward.assign(self._current_ground_truth)
        else:
            self._reward.assign(tf.squeeze(-1 * tf.math.abs(action - self._current_ground_truth)))
        self._steps.assign_add(1)
        pos = int(tf.squeeze(self._current_data_pos))
        self._state.assign(self.ts_data[pos:pos + self.window_size].values)
        self._current_data_pos.assign_add(self.window_size)
        self._current_ground_truth.assign(self.ts_data[int(tf.squeeze(self._current_data_pos))])
        self._current_data_pos.assign_add(1)
        self._window_counter.assign_add(1)
        if self._current_data_pos + self.window_size < len(self.ts_data):
            if self.max_window_count != -1:
                if self._window_counter >= self.max_window_count:
                    return self._reset()
            # self._build_observation()
            return self.current_time_step()
        else:
            return self._reset()


@gin.configurable
class TsForecastingSingleStepEnv(gym.Env):
    def __init__(self, ts_data, rl_algorithm, window_size=5, max_window_count=-1, min_attribute_val=35.0,
                 max_attribute_val=500.0, reward_def="abs_diff", evaluation=False): #, include_prev_pred=True):
        self.reward_def = reward_def
        self.evaluation = evaluation
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = window_size
        # specify max number of windows to use; -1 to use all windows
        self.max_window_count = max_window_count
        self.window_counter = 0
        self.current_data_pos = 0
        self.current_ground_truth = None
        # self.include_prev_ped = include_prev_pred
        # self.state = None
        self.max_attribute_val = max_attribute_val
        self.min_attribute_val = min_attribute_val
        # define observation space
        # if include_prev_pred:
        #     self.observation_space = Box(np.array([min_attribute_val for _ in range(self.window_length + 1)]),
        #                                  np.array([max_attribute_val for _ in range(self.window_length + 1)]))
        # else:
        self.observation_space = Box(np.array([min_attribute_val for _ in range(self.window_length)]),
                                     np.array([max_attribute_val for _ in range(self.window_length)]))
        # define action space
        if rl_algorithm == "dqn":
            self.action_space = Box(np.array(min_attribute_val), np.array(max_attribute_val))
        else:
            self.action_space = Box(np.array([min_attribute_val]), np.array([max_attribute_val]))

    def step(self, action):
        if self.evaluation:
            reward = self.current_ground_truth
        else:
            if self.reward_def == "abs_diff":
                # normalize in [0, 1]
                reward = np.squeeze(np.abs(action - self.current_ground_truth))
                # normalization
                # reward /= (self.max_attribute_val - self.min_attribute_val)
                # large diff -> small reward, small diff -> large reward
                # reward = -1 * (reward - 1)
                reward *= -1
            elif self.reward_def == "squared_diff":
                reward = tf.squeeze(tf.math.squared_difference(action, self.current_ground_truth))
                # normalization
                # reward /= (self.max_attribute_val - self.min_attribute_val) ** 2
                # large diff -> small reward, small diff -> large reward
                # reward = -1 * (reward - 1)
                reward *= -1
            elif self.reward_def == "linear":
                # calculate reward -> reward scale: [0, 1]
                reward = np.squeeze(np.abs(action - self.current_ground_truth))
                # reward = tf.squeeze(tf.math.squared_difference(action, self.current_ground_truth))
                # reward = ((-1 / (self.max_attribute_val - self.min_attribute_val)) * reward) + 1
                # reward = ((-1 / (self.max_attribute_val - self.min_attribute_val) ** 2) * reward) + 1
            elif self.reward_def == "exponential":
                # calculate reward -> reward scale: [0, 1]
                reward = np.squeeze(np.exp(-np.abs(action - self.current_ground_truth)))
                # reward = tf.squeeze(tf.math.exp(-tf.math.squared_difference(action, self.current_ground_truth)))
            else:
                logging.info("Reward definition {} is not supported".format(self.reward_def))

        # get next observation -> fixed size window
        state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # if self.include_prev_ped:
        #     state = list(state)
        #     state.append(np.squeeze(action))
        #     state = np.array(state)
        # set current data position and ground truth for next step
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        self.window_counter += 1
        # done is True at the end of the time series data -> restart at random position of the series
        if self.current_data_pos + self.window_length < self.num_data_points:
            if self.max_window_count != -1:
                if self.window_counter < self.max_window_count:
                    done = False
                else:
                    done = True
            else:
                done = False
        else:
            done = True
        return state, reward, done, ()

    def reset(self):
        self.window_counter = 0
        if self.evaluation:
            self.current_data_pos = 0
        else:
            self.current_data_pos = np.random.randint(low=0, high=self.num_data_points - (2 * self.window_length + 1))
        state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # if self.include_prev_ped:
        #     state = list(state)
        #     state.append(0.0)
        #     state = np.array(state)
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        self.window_counter += 1
        return state

    def render(self, mode='human'):
        pass


@gin.configurable
class TsForecastingMultiStepEnv(gym.Env):
    def __init__(self, ts_data, window_size=12, max_window_count=-1, forecasting_steps=5, min_attribute_val=35.0,
                 max_attribute_val=500.0, reward_def="abs_diff", evaluation=False, use_rnn_state=True):
        self.reward_def = reward_def
        self.evaluation = evaluation
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = window_size
        self.max_window_count = max_window_count
        self.window_counter = 0
        self.forecasting_steps = forecasting_steps
        self.current_data_pos = 0
        self.current_ground_truth = None
        self.state = None
        self.use_rnn_state = use_rnn_state
        self.min_attribute_val = min_attribute_val
        self.max_attribute_val = max_attribute_val
        # define observation space
        self.observation_space = Box(np.array([min_attribute_val for _ in range(self.window_length)]),
                                     np.array([max_attribute_val for _ in range(self.window_length)]))
        # define action space
        self.action_space = Box(np.array([min_attribute_val for _ in range(self.forecasting_steps)]),
                                np.array([max_attribute_val for _ in range(self.forecasting_steps)]))

    # def _step(self, action, rnn_state=None):
    #     self.step(action, rnn_state=rnn_state)

    def step(self, action): #, rnn_state):
        if self.evaluation:
            reward = self.current_data_pos
        else:
            if self.reward_def == "abs_diff":
                # normalize in [0, 1]
                reward = tf.reduce_mean(np.abs(action - self.current_ground_truth))
                # normalization
                reward /= (self.max_attribute_val - self.min_attribute_val)
                # large diff -> small reward, small diff -> large reward
                reward = -1 * (reward - 1)
            elif self.reward_def == "linear":
                # calculate reward -> reward scale: [0, 1]
                reward = tf.math.reduce_mean(np.abs(action - self.current_ground_truth))
                reward = ((-1 / (self.max_attribute_val - self.min_attribute_val)) * reward) + 1
            elif self.reward_def == "exponential":
                reward = np.exp(-tf.math.reduce_mean(tf.math.abs(action - self.current_ground_truth)))
            else:
                logging.info("Reward definition {} is not supported".format(self.reward_def))
            # get next observation -> fixed size window
        self.state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # set current data position and ground truth for next step
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos:self.current_data_pos+self.forecasting_steps]
        self.current_data_pos += self.forecasting_steps
        self.window_counter += 1
        # done is True at the end of the time series data -> restart at random position of the series
        if self.current_data_pos + self.window_length + self.forecasting_steps < self.num_data_points:
            if self.max_window_count != -1:
                if self.window_counter < self.max_window_count:
                    done = False
                else:
                    done = True
            else:
                done = False
        else:
            done = True
        return self.state, reward, done, ()

    # def _reset(self):
    #     self.reset()

    def reset(self):
        if self.evaluation:
            self.current_data_pos = 0
        else:
            self.current_data_pos = np.random.randint(
                low=0,
                high=self.num_data_points - 2 * (self.window_length + self.forecasting_steps))
        self.state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos:self.current_data_pos+self.forecasting_steps]
        self.current_data_pos += self.forecasting_steps
        self.window_counter += 1
        return self.state

    def render(self, mode='human'):
        pass


@gin.configurable
class TsForecastingMultiStepTFEnv(tf_environment.TFEnvironment):
    def __init__(self):
        pass

    def _current_time_step(self):
        pass

    def _reset(self):
        pass

    # policy state is state of RNN (actor network)
    def _step(self, action, policy_state=None):
        pass


# only necessary if environment is Python / Gym environment
def get_tf_environment(env):
    env = GymWrapper(env)
    env = TFPyEnvironment(env)
    return env
