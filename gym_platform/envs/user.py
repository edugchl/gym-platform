import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import LabelBinarizer


class UserEnv(gym.Env):
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
        if day_of_week is in range(1,6):
            freeness = 0.5
        else:
            freeness = 0.9

        dependent_discount = -np.log(self.num_dependents+30)+2
        freeness *= dependent_discount
        return freeness

    def _compute_notification_burden(self):
        # TODO: model notification burden
        # A simple way to compute to get the program up and run
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
        return self._get_obs()

    def step(self, action):
        # TODO: model reward as a function of free time and notification burden
        pass

    def render(self, mode):
        pass

    def close(self):
        pass
