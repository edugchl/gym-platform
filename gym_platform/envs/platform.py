import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PlatformEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, user_env, world_env):
        super().__init__()
        self.user_env = user_env
        self.world_env = world_env

    def reset(self):
        world_obs = self.world_env.reset()
        user_obs = self.user_env.reset()
        obs = np.concatenate((user_obs, world_obs), axis=0)
        return obs

    def _is_terminal(self):
        # update every 724 hours, i.e. 30 days
        if self.world_env.hours_elapsed >= 30*24:
            return True
        else:
            return False

    def _step(self, action):
        user_obs, user_reward, user_terminal, _ = self.user_env.step(action)
        world_obs, world_reward, world_terminal, _ = self.world_env.step()
        obs = np.concatenate((user_obs, world_obs), axis=0)
        terminal = self._is_terminal()
        return obs, user_reward, terminal, _ 

    def step(self, action):
        obs, reward, terminal, _ = self._step(action)
        return obs, reward, terminal, _

    def render(self, mode='human'):
        # TODO: render the env
        pass

    def close(self):
        pass
