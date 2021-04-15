from datetime import datetime

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import LabelBinarizer


class UserEnv(gym.Env):
    """
    Attributes:
        - state: the state of a user; what we observe and understand
        - obs: observation space, transformed state; input to agent
    """
    JOB_LIST = ['WHITE COLLAR','BLUE COLLAR', 'ON SHIFT']
    job_encoder = LabelBinarizer()
    job_encoder.fit(JOB_LIST)

    def __init__(self, job_type, num_dependents):
        super().__init__()
        self.job_type = job_type
        # Encoded the job type and remove last dim to avoid collinearity 
        self.job_encoded = job_encoder.transform(job_type)[:,:-1]
        self.num_dependents = num_dependents
        self.state = None

    def _compute_freeness(self):
        # model free time based on job and dependents
        # A simple way to compute to get the program up and run
        weekday_freeness = np.random.uniform(0.5,0.8) if day_of_week in range(1,6) else 0.9
        hour_freeness = np.random.uniform(0.7,0.9) if hour in range(19,23) else 0.3
        dependent_discount = -np.log(self.num_dependents+0.1)+1.04139268516     # not work if num_dependents>10
        freeness = weekday_freeness * hour_freeness * dependent_discount
        return freeness

    def _compute_notification_burden(self):
        # A simple way to compute to get the program up and run
        notification_burden = (log(-self.hrs_since_notification+30.001)+2)/3.47713573096
        return notification_burden
    
    def _get_obs(self):
        """Transform the state into observation. \
        The benefit of computing freeness and notification burden is that \
        we can encode our knowledge about the human behaviour into the function, \
        so that our agent does not need to learn from an extremely large state space.
        """
        freeness = self._compute_freeness()
        notification_burden = self._compute_notification_burden()
        return np.array([freeness, notification_burden])

    def reset(self):
        """Reset the user state to initial values, and return obs."""
        now = datetime.now()
        self.last_notification = now
        self.hrs_since_notification = 0
        self.state = np.array([hrs_since_notification])
        return self._get_obs()

    @staticmethod
    def time_elapsed(t1, t2, unit='hours', digit=2):
        elapsed = t2 - t1
        if unit == 'hours':
            return round(elapsed.total_seconds()/3600, digit)
        elif unit == 'minutes':
            return round(elapsed.total_seconds()/60, digit)
        elif unit == 'seconds':
            return round(elapsed.total_seconds(), digit)
        else:
            raise NotImplementedError('Required unit is not implemented.')

    def _act(self, action):
        assert (action==0) or (action==1)
        now = datetime.now()
        
        if action == 0:
            self.hrs_since_notification = time_elapsed(self.last_notification, now)
        else:
            self.last_notification = now
            self.hrs_since_notification = 0.0

        self.state = np.array([self.hrs_since_notification])

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
        
        if action == 1:
            free = freeness - 0.6
            anti_burden = 1 - notification_burden - 0.7
            direction = self.__direction_and_gate(free>=0, anti_burden>=0)
            magnitude = 5.0 if direction == 1 else 3.0
        else:
            free = freeness - 0.5
            anti_burden = 1 - notification_burden - 0.5
            direction = self.__direction_nand_gate(free>=0, anti_burden>=0)
            magnitude = 3.0 if direction == 1 else 1.0
        reward = direction * magnitude
        return reward

    def _is_terminal(self):
        # assume no terminal state in the single-user environment
        return False

    def step(self, action):
        # update current state
        self._act(action)
        # retrieve current obs for computing rewards  
        obs = self._get_obs()
        # reward
        reward = self._compute_reward(obs, action)
        terminal = self._is_terminal()
        return self._get_obs(), reward, terminal, {}

    def render(self, mode):
        pass

    def close(self):
        pass
