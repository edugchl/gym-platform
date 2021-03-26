import time

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class WorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def _encode_cycle(self, t, max_val):
        # encode the datetime and capture the cyclical pattern
        # https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
        sin_feat = np.sin(2 * np.pi * t/max_val)
        cos_feat = np.cos(2 * np.pi * t/max_val)
        return sin_feat, cos_feat

    def _get_obs(self):
        year, month, day, hour, min, day_of_week = map(int, time.strftime("%Y %m %d %H %M %w").split())
        month_feat = self._encode_cycle(month, 12)
        day_feat = self._encode_cycle(day, 31)
        hour_feat = self._encode_cycle(hour, 23)
        min_feat = self._encode_cycle(month, 59)
        dow_feat = self._encode_cycle(day_of_week, 6)
        datetime_state = np.array([month_feat, day_feat, hour_feat, min_feat, dow_feat]).flatten()
        return datetime_state

    def reset(self):
        obs = self._get_obs()
        return obs

    def step(self, action):
        pass

    def render(self, mode):
        pass

    def close(self):
        pass