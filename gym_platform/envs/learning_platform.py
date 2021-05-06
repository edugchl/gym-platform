import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_platform.envs.utils import time_elapsed
from gym_platform.envs.user import User
from gym_platform.envs.clock import Clock


class LearningPlatform(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        user: User,  
        clock: Clock, 
        end_days: int,
        include_time_features: bool = True,
    ):
        self.user = user
        self.clock = clock
        self.end_days = end_days
        self.include_time_features = include_time_features
        self.start = self.user.start

        self.action_space = self.user.action_space
        self.low = user.low
        self.high = user.high
        if self.include_time_features:
            self.low = np.concatenate([self.user.low, self.clock.low])
            self.high = np.concatenate([self.user.high, self.clock.high])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float64)

    def reset(self):
        obs = self.user.reset()

        if self.include_time_features:
            time_obs = self.clock.time_features(dt=self.start)
            obs = np.concatenate((obs, time_obs), axis=0)

        return obs

    def step(self, action: int, now: datetime):
         obs, reward, user_terminal, _ = self.user.step(action, now)

        if self.include_time_features:
            time_obs = self.clock.time_features(dt=now)
            obs = np.concatenate((obs, time_obs), axis=0)

        terminal = False
        if time_elapsed(t1=self.start, t2=now, unit='days') > self.end_days:
            terminal = True
    
        return obs, reward, terminal, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
