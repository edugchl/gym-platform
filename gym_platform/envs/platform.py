import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PlatformEnv(gym.Env):
    """
    Args:
        - til_doomsday: no. of hours until doomsday. i.e. terminal=True
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, user_env, world_env, til_doomsday):
        super().__init__()
        self.user_env = user_env
        self.world_env = world_env
        self.til_doomsday = til_doomsday

        low = np.concatenate([np.zeros(2), np.ones(10)*-1])
        high = np.ones(12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        world_obs = self.world_env.reset()
        user_obs = self.user_env.reset()
        obs = np.concatenate((user_obs, world_obs), axis=0)
        return obs

    def is_terminal(self):
        # update every 724 hours, i.e. 30 days
        if self.world_env.hours_elapsed >= self.til_doomsday:
            return True
        else:
            return False

    def step(self, action):
        user_obs, user_reward, user_terminal, _ = self.user_env.step(action)
        world_obs, world_reward, world_terminal, _ = self.world_env.step()
        obs = np.concatenate((user_obs, world_obs), axis=0)
        terminal = self.is_terminal()
        return obs, user_reward, terminal, {}

    def render(self, mode='human'):
        # TODO: render the env
        pass

    def close(self):
        pass
