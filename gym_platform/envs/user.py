import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import LabelBinarizer


class UserEnv(gym.Env):
    JOB_LIST = ['WHITE COLLAR','BLUE COLLAR', 'ON SHIFT']
    job_encoder = LabelBinarizer()
    job_encoder.fit(JOB_LIST)

    def __init__(self, job_type, num_dependents, notification_burden):
        super().__init__()
        self.job_type = job_type
        # Encoded the job type and remove last dim to avoid collinearity 
        self.job_encoded = job_encoder.transform(job_type)[:,:-1]
        self.num_dependents = num_dependents
        self.initial_burden = notification_burden
    
    def _get_obs(self):
        return np.append(self.job_encoded, self.num_dependents, self.notification_burden)

    def reset(self):
        """Reset the user state to initial values."""
        self.notification_burden = self.initial_burden
        return self._get_obs()

    def step(self, action):
        pass

    def render(self, mode):
        pass

    def close(self):
        pass
