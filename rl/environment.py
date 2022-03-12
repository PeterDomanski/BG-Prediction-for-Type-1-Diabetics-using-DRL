import gym
import numpy as np
import tensorflow as tf
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from gym.spaces import Box
import gin


@gin.configurable
class TsForecastingSingleStepEnv(gym.Env):
    def __init__(self, ts_data, window_size=5, min_attribute_val=35.0, max_attribute_val=500.0,
                 reward_def="abs_diff", evaluation=False):
        self.reward_def = reward_def
        self.evaluation = evaluation
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = window_size
        self.current_data_pos = 0
        self.current_ground_truth = None
        self.state = None
        self.max_attribute_val = max_attribute_val
        self.min_attribute_val = min_attribute_val
        # define observation space
        self.observation_space = Box(np.array([min_attribute_val for _ in range(self.window_length)]),
                                     np.array([max_attribute_val for _ in range(self.window_length)]))
        # define action space
        self.action_space = Box(np.array([min_attribute_val]), np.array([max_attribute_val]))

    def step(self, action):
        if self.evaluation:
            reward = self.current_ground_truth
        else:
            if self.reward_def == "abs_diff":
                # normalize in [0, 1]
                reward = np.abs(action - self.current_ground_truth)
                # normalization
                reward /= (self.max_attribute_val - self.min_attribute_val)
                # large diff -> small reward, small diff -> large reward
                reward = -1 * (reward - 1)
            elif self.reward_def == "linear":
                # calculate reward -> reward scale: [0, 1]
                reward = np.abs(action - self.current_ground_truth)
                reward = (-1 / self.max_attribute_val - self.min_attribute_val) * reward + 1
            elif self.reward_def == "exponential":
                # calculate reward -> reward scale: [0, 1]
                reward = np.squeeze(np.exp(-np.abs(action - self.current_ground_truth)))
            else:
                print("Reward definition {} is not supported".format(self.reward_def))

        # get next observation -> fixed size window
        self.state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # set current data position and ground truth for next step
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        # done is True at the end of the time series data -> restart at random position of the series
        if self.current_data_pos + self.window_length < self.num_data_points:
            done = False
        else:
            done = True
        return self.state, reward, done, ()

    def reset(self):
        if self.evaluation:
            self.current_data_pos = 0
        else:
            self.current_data_pos = np.random.randint(low=0, high=self.num_data_points - (2 * self.window_length + 1))
        self.state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        return self.state

    def render(self, mode='human'):
        pass


@gin.configurable
class TsForecastingMultiStepEnv(gym.Env):
    def __init__(self, ts_data, window_size=12, forecasting_steps=5, min_attribute_val=35.0, max_attribute_val=500.0,
                 reward_def="abs_diff", evaluation=False):
        self.reward_def = reward_def
        self.evaluation = evaluation
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = window_size
        self.forecasting_steps = forecasting_steps
        self.current_data_pos = 0
        self.current_ground_truth = None
        self.state = None
        self.min_attribute_val = min_attribute_val
        self.max_attribute_val = max_attribute_val
        # define observation space
        self.observation_space = Box(np.array([min_attribute_val for _ in range(self.window_length)]),
                                     np.array([max_attribute_val for _ in range(self.window_length)]))
        # define action space
        self.action_space = Box(np.array([min_attribute_val for _ in range(self.forecasting_steps)]),
                                np.array([max_attribute_val for _ in range(self.forecasting_steps)]))

    def step(self, action):
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
                reward = (-1 / self.max_attribute_val - self.min_attribute_val) * reward + 1
            elif self.reward_def == "exponential":
                reward = np.exp(-tf.math.reduce_mean(tf.math.abs(action - self.current_ground_truth)))
            else:
                print("Reward definition {} is not supported".format(self.reward_def))
            # get next observation -> fixed size window
        self.state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # set current data position and ground truth for next step
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos:self.current_data_pos+self.forecasting_steps]
        self.current_data_pos += self.forecasting_steps
        # done is True at the end of the time series data -> restart at random position of the series
        if self.current_data_pos + self.window_length + self.forecasting_steps < self.num_data_points:
            done = False
        else:
            done = True
        return self.state, reward, done, ()

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
        return self.state

    def render(self, mode='human'):
        pass


def get_tf_environment(env):
    env = GymWrapper(env)
    env = TFPyEnvironment(env)
    return env
