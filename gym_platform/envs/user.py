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
        start: datetime, 
        alpha: float = -3e-06,
        e: float = 4.0, 
        freeness_threshold: float = 0.5,
        burden_threshold: float = 0.8,
        ):
        self.job_type = job_type
        self.num_dependents = num_dependents
        self.start = start
        self.alpha = alpha # control the burden cycle 
        self.e = e # control how fast users feel stressed
        self.freeness_threshold = freeness_threshold
        self.burden_threshold = burden_threshold
        self.min_freeness = 0.0
        self.max_freeness = 1.0
        self.min_burden = 0.0
        self.max_burden = 1.0

        self.action_space = spaces.Discrete(2)
        self.low = np.array([self.min_freeness, self.min_burden])
        self.high = np.array([self.max_freeness, self.max_burden])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def freeness(self, dt: datetime):
        job_freeness = self.job_freeness(job=self.job_type, dt=dt)
        dependent_freeness = self.dependent_freeness(num=self.num_dependents)
        freeness = self.max_freeness * job_freeness * dependent_freeness
        return freeness

    def job_freeness(self, job, dt):
        hour,  weekday = dt.hour, dt.weekday()

        if job == 'WHITE COLLAR':
            sleep_hours = 8
            work_hours = 11

            schedule = np.ones((7,24))
            working_hours = list(range(8,19))
            working_days = list(range(1,6))

            sleep_schedule = np.zeros((7,sleep_hours))
            working_schedule = np.random.uniform(low=0.1,high=0.3,size=(5,work_hours))
            weekday_free_schedule = np.clip(np.random.normal(loc=0.7, scale=0.05, size=(5,24-sleep_hours-work_hours)),0, 1)
            weekend_free_schedule = np.clip(np.random.normal(loc=0.9, scale=0.1, size=24-sleep_hours),0, 1)

            # assume sleeping at 12 a.m.
            schedule[:,:sleep_hours] = sleep_schedule
            schedule[[0,6],sleep_hours:] = weekend_free_schedule
            schedule[1:6,sleep_hours:sleep_hours+work_hours] = working_schedule
            schedule[1:6,sleep_hours+work_hours:] = weekday_free_schedule
        
        score = schedule[weekday][hour]
        return score

    def dependent_freeness(self, num):
        score = -np.log10( (num/12) + 0.1)
        # score = np.clip(y, 0, 1)
        return score

    def burden(self, hrs_since_notification):
        y = self.alpha * hrs_since_notification**(self.e) + 1
        notification_burden = np.clip(y, 0, 1)
        return notification_burden

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