from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import LabelBinarizer

from gym_platform.envs.utils import time_elapsed


class User(gym.Env):
    def __init__(
        self, 
        job_type: str, 
        num_dependents: int, 
        dt: datetime, 
        alpha: float = -3e-06,
        e: float = 4.0, 
        freeness_threshold: float = 0.5,
        burden_threshold: float = 0.8,
        ):
        self.job_type = job_type
        self.num_dependents = num_dependents
        self.start = dt
        self.alpha = alpha # control the burden cycle 
        self.e = e # control how fast users feel stressed
        self.freeness_threshold = freeness_threshold
        self.burden_threshold = burden_threshold
        self.min_freeness = 0.0
        self.max_freeness = 1.0
        self.min_burden = 0.0
        self.max_burden = 1.0

        self.action_space = spaces.Discrete(2)
        low = np.array([self.min_freeness, self.min_burden])
        high = np.array([self.max_freeness, self.max_burden])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    @property
    def freeness(self, dt: datetime):
        # TODO: test it
        job_freeness = self.job_freeness(job=self.job_type, dt=dt)
        dependent_freeness = self.dependent_freeness(num=self.num_dependents)
        freeness = self.max_freeness * job_freeness * dependent_freeness
        return freeness

    @property
    def job_freeness(self, job, dt):
        hour,  weekday = dt.hour, dt.weekday()

        if job == 'WHITE COLLAR':
            weekday_score = np.random.uniform(0.5,0.8) if weekday in range(1,6) else 0.9
            hour_score = np.random.uniform(0.7,0.9) if hour in range(18,23) else 0.3
        
        score = weekday_score * hour_score
        return score

    @property
    def dependent_freeness(self, num):
        score = np.clip(-np.log(num/10+0.1), 0, 1)
        return score

    @property
    def burden(self, hrs_since_notification):
        # TODO: test it
        y = self.alpha * hrs_since_notification**(self.e) + 1
        notification_burden = np.clip(y, 0, 1)
        return notification_burden

    @property
    def obs(self, dt):
        freeness = self.freeness(dt)
        hrs_since_notification = time_elapsed(self.last_notification, dt, 'hours')
        burden = self.burden(hrs_since_notification)
        return np.array([freeness, burden])

    def get_reward(self, obs, action):
        freeness, burden = obs[0], obs[1]

        # OK to receive 
        if (freeness>=self.freeness_threshold) and (burden<=self.burden_threshold):
            reward = 1.0 if action == 1 else -1e05
        # definitely not OK to receive
        elif (freeness<=self.freeness_threshold) and (burden>=self.burden_threshold): 
            reward = -1.0 if action == 1 else 1e05
        # not quite sure, better not
        else:
            reward = -1e04 if action == 1 else 0.0
        
        return reward

    def reset(self):
        elapsed = timedelta(
            days=np.random.randint(0, 3), 
            hours=np.random.randint(0, 23), 
            seconds=np.random.randint(0, 86400))
        self.last_notification = self.start - elapsed
        obs = self.obs(self.start)
        return obs
        
    def step(self, action: int, now: datetime):
        # TODO: test it
        obs = self.obs
        if action == 1:
            self.last_notification = now
        reward = self.get_reward(obs, action)
        obs_next = self.obs(now)
        terminal = False
        return obs_next, reward, terminal, {}