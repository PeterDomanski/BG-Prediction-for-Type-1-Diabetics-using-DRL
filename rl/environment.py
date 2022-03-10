import gym
import numpy as np
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from gym.spaces import Box, Tuple


class TsForecastingSingleStepEnv(gym.Env):
    def __init__(self, ts_data):
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = 5
        self.current_data_pos = 0
        self.current_ground_truth = None
        # define observation space
        # self.observation_space = Box(np.expand_dims(np.array([0.0 for _ in range(self.window_length)]), 0),
        #                              np.expand_dims(np.array([2.0 for _ in range(self.window_length)]), 0))
        self.observation_space = Box(np.array([0.0 for _ in range(self.window_length)]),
                                     np.array([2.0 for _ in range(self.window_length)]))
        # define action space
        # self.action_space = Tuple([Box(np.array([0.0]), np.array([2.0]))])
        self.action_space = Box(np.array([0.0]), np.array([2.0]))

    def step(self, action):
        # get next observation -> fixed size window
        state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        # set current data position and ground truth for next step
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        # calculate reward -> reward scale: [0, 1]
        reward = np.squeeze(np.exp(-np.abs(action - self.current_ground_truth)))
        # done is True at the end of the time series data -> restart at random position of the series
        if self.current_data_pos < (self.num_data_points - self.window_length):
            done = False
        else:
            done = True

        return state, reward, done, ()

    def reset(self):
        self.current_data_pos = np.random.randint(low=0, high=len(self.ts_data) - self.window_length)
        state = self.ts_data[self.current_data_pos:self.current_data_pos + self.window_length].values
        self.current_data_pos += self.window_length
        self.current_ground_truth = self.ts_data[self.current_data_pos]
        self.current_data_pos += 1
        return state

    def render(self, mode='human'):
        pass


class TsForecastingMultiStepEnv(gym.Env):
    def __init__(self, ts_data):
        self.ts_data = ts_data
        self.num_data_points = len(ts_data)
        self.window_length = 25
        self.current_data_pos = 0
        self.forecasting_steps = 5
        self.current_ground_truth = [None for _ in range(self.forecasting_steps)]

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


def get_tf_environment(env):
    env = GymWrapper(env)
    env = TFPyEnvironment(env)
    return env
