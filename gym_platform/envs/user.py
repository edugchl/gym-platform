from collections import defaultdict

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import LabelBinarizer

from gym_platform.envs.utils import time_elapsed


class UserEnv(gym.Env):
    """
    Attributes:
        - state: the state of a user; what we observe and understand
        - obs: observation space, transformed state; input to agent
    """
    JOB_LIST = ['WHITE COLLAR','BLUE COLLAR', 'ON SHIFT']
    job_encoder = LabelBinarizer()
    job_encoder.fit(JOB_LIST)

    def __init__(self, world, job_type, num_dependents):
        super().__init__()
        self.world = world
        self.state = defaultdict(dict)
        self.state['user']['num_dependents'] = num_dependents
        self.state['user']['job_type'] = job_type
        # Encoded the job type and remove last dim to avoid collinearity 
        # self.state['user']['job_encoded'] = job_encoder.transform(job_type)[:,:-1]

        # obs: freeness, notification_burden
        low = np.array([0, 0])
        high = np.array([1, 1])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def compute_freeness(self, num_dependents, hour_of_day, day_of_week):
        """Model free time based on job and dependents.
        """
        # A simple way to compute to get the program up and run
        weekday_freeness = np.random.uniform(0.5,0.8) if day_of_week in range(1,6) else 0.9
        hour_freeness = np.random.uniform(0.7,0.9) if hour_of_day in range(19,23) else 0.3
        dependent_discount = np.clip(-np.log(num_dependents/10+0.1), 0, 1)
        freeness = weekday_freeness * hour_freeness * dependent_discount
        return freeness

    def compute_notification_burden(self, hrs_since_notification):
        # A simple way to compute to get the program up and run
        alpha = -3e-06  # control the cycle 
        e = 4   # control how fast user feels stressed
        notification_burden = np.clip(alpha*hrs_since_notification**(e) + 1, 0, 1)
        return notification_burden
    
    def _get_obs(self):
        """Transform the state into observation. 
        The benefit of computing freeness and notification burden is that 
        we can encode our knowledge about the human behaviour into the function, 
        so that our agent does not need to learn from an extremely large state space.

        The observation will be fed to agent to compute actions and estimate rewards. 
        """
        # get the varaibles from state
        num_dependents = self.state['user']['num_dependents']
        hrs_since_notification = self.state['notification']['hrs_since_notification']
        timestamp = self.state['world']['time']
        hour_of_day, day_of_week = timestamp.hour, timestamp.weekday()
        # transform into observation
        freeness = self.compute_freeness(num_dependents, hour_of_day, day_of_week)
        notification_burden = self.compute_notification_burden(hrs_since_notification)
        # TODO: add job features
        return np.array([freeness, notification_burden])

    def reset(self):
        """Reset the user state to initial values, and return obs."""
        now = self.world.current_time
        self.state['notification']['last_notification'] = now
        self.state['notification']['hrs_since_notification'] = 0
        self.state['world']['time'] = now
        return self._get_obs()

    def _act(self, action):
        assert (action==0) or (action==1)
        # get the state
        last_notification = self.state['notification']['last_notification']
        now = self.world.current_time
        # compute
        if action == 0:
            hrs_since_notification = time_elapsed(last_notification, now)
        else:
            last_notification = now
            hrs_since_notification = 0.0
        # update the state
        self.state['notification']['last_notification'] = last_notification
        self.state['notification']['hrs_since_notification'] = hrs_since_notification
        self.state['world']['time'] = now

    def __direction_and_gate(self, a, b):
        if a == 1 and b == 1:
            return 1
        else:
            return -1

    def __direction_nand_gate(self, a, b):
        if a == 1 and b == 1:
            return -1
        else:
            return 1    

    def _compute_reward(self, obs, action):
        """Try to model the reward given by user, \
        yet ideally this should be the feedback provided by users.
        
        Reward should be a function of the state, not observation. 
        But for simplicity, assume the environment is fully observable by agent
        and so state is the same as observation.

        Reward is computed as direction * magnitude, in order to 
        punish (more) false positive and (less) false negative, and
        reward (more) true positive and (less) true negative.
        """
        freeness, notification_burden = obs[0], obs[1]
        
        # take action
        if action == 1:
            free = freeness - 0.6
            anti_burden = 1 - notification_burden - 0.7
            direction = self.__direction_and_gate(free>=0, anti_burden>=0)
            # if ok to receive notification, get +5.0
            if direction == 1:
                magnitude = 5.0 
            # if not ok to receive notification, get -3.0
            else:
                magnitude = 3.0
        # not take action
        else:
            free = freeness - 0.5
            anti_burden = 1 - notification_burden - 0.5
            direction = self.__direction_nand_gate(free>=0, anti_burden>=0)
            # if not ok to receive notification, get +3.0
            if direction == 1:
                magnitude = 0.01
            # if ok to receive notification, get -1.0
            else:
                magnitude = 0.5
        
        reward = direction * magnitude
        return reward

    def step(self, action):
        # update current state
        self._act(action)
        # retrieve current obs for computing rewards  
        obs = self._get_obs()
        # reward
        reward = self._compute_reward(obs, action)
        # assume no terminal state in the single-user environment
        terminal = False
        return self._get_obs(), reward, terminal, {}

    def render(self, mode):
        pass

    def close(self):
        pass
