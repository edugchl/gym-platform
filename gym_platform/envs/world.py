from datetime import datetime

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from utils import time_elapsed


class WorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def _encode_cycle(self, t, max_val):
        # encode the datetime and capture the cyclical pattern
        # https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
        sin_feat = np.sin(2 * np.pi * t/max_val)
        cos_feat = np.cos(2 * np.pi * t/max_val)
        return sin_feat, cos_feat

    def timestamp_to_features(self, ts):
        month, day, hour, minute, day_of_week = ts.month, ts.day, ts.hour, ts.minute, ts.weekday()
        month_feat = self._encode_cycle(month, 12)
        day_feat = self._encode_cycle(day, 31)
        hour_feat = self._encode_cycle(hour, 23)
        min_feat = self._encode_cycle(minute, 59)
        dow_feat = self._encode_cycle(day_of_week, 6)
        features = np.array([month_feat, day_feat, hour_feat, min_feat, dow_feat]).flatten()
        return features

    @property
    def current_time(self):
        return datetime.now()

    @property
    def hours_elapsed(self):
        elapsed = time_elapsed(self.start_time, self.current_time)
        return elapsed

    def reset(self):
        self.start_time = self.current_time
        return self.timestamp_to_features(self.start_time)

    def step(self):
        now = self.current_time
        obs, reward, terminal, _ = self.timestamp_to_features(now), None, False, None
        return obs, reward, terminal, _

    def render(self, mode):
        pass

    def close(self):
        pass