import gym
from gym import error, spaces, utils
from gym.utils import seeding


class WorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def reset(self):
        obs = self._get_obs()
        return obs

    def step(self, action):
        pass

    def render(self, mode):
        pass

    def close(self):
        pass