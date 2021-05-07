"""
TODO:
    - Catching KeyboardInterrupt
    - rebuild 
"""

import argparse
import time
from dateutil.relativedelta import relativedelta

import numpy as np
from stable_baselines3 import A2C
import wandb

from gym_platform import LearningPlatform, User, Clock
from gym_platform.envs.utils import flatten_dict


hyperparameter_defaults = {
    # general
    'wandb': False, 
    'episodes': 100, 
    # environment
    'clock_ratio': 36000,
    'job': 'WHITE COLLAR', 
    'dependents': 0, 
    'end_days': 365, 
    'include_time_features': False, 
    # agent
    'agent_timesteps': 100, 
}


def train(params=None):

    clock = Clock(params['clock_ratio'])
    start = clock.real_start
    user = User(params['job'], params['dependents'], clock.artifical_start)
    platform = LearningPlatform(user, clock, params['end_days'], params['include_time_features'])
    obs = platform.reset()
    terminal = False
    print(f'initial obs: {obs}')
    # model = A2C('MlpPolicy', platform, verbose=1)
    # model.learn(total_timesteps=params['agent_timesteps'])

    for eps in range(params['episodes']):
        
        while not terminal:
            action = 0
            # action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminal, _ = platform.step(action, )

            # TODO: remove
            # simulate time passes
            # time.sleep(3600/params['clock_ratio'])

            if terminal:
                obs = platform.reset()


if __name__ == '__main__':
    train(params=hyperparameter_defaults)
