from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_platform.envs.utils import time_elapsed


class WorldEnv(gym.Env):
    """
    Args:
        - ratio: using it to change the speed of the time, it is an exchange rate between the seconds in the artificial \
                world and the real world. For instance, if ratio=3600, it means that 1 second in real world equals to 1 hour \
                (3600 seconds) in the artificial world.
    """
    def __init__(self, ratio: int):
        super().__init__()
        self.ratio = ratio

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
        """Time in the aritifical world."""
        seconds_elapsed_real = time_elapsed(self.start_time, self.local_time, unit='seconds', digit=10)
        seconds_elapsed_aritifical = seconds_elapsed_real * self.ratio
        current_time_aritifical = self.start_time + timedelta(seconds=seconds_elapsed_aritifical)
        return current_time_aritifical

    @property
    def local_time(self):
        return datetime.now()

    @property
    def hours_elapsed(self):
        elapsed = time_elapsed(self.start_time, self.current_time)
        return elapsed

    def reset(self):
        self.start_time = datetime.now()
        return self.timestamp_to_features(self.start_time)

    def step(self):
        now = self.current_time
        obs, reward, terminal, _ = self.timestamp_to_features(now), None, False, None
        return obs, reward, terminal, _

    def render(self, mode):
        pass

    def close(self):
        pass