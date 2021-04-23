import time
from dateutil.relativedelta import relativedelta

import numpy as np
from stable_baselines3 import A2C
import wandb

from gym_platform import PlatformEnv, UserEnv, WorldEnv


wandb.init(project='rl-notification-system')
world_env = WorldEnv(ratio=3600)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env, 7*24)

model = A2C('MlpPolicy', platform, verbose=1)
model.learn(total_timesteps=7*24)

terminal = False
obs = platform.reset()
while True:
    time.sleep(1)
    # action = np.random.choice(2)
    action, _state = model.predict(obs, deterministic=True)
    obs, r, terminal, _ = platform.step(action)
    wandb.log({'action': action, 'reward':r})
    print(f'state: {platform.user_env.state}\n, obs: {obs}\n, action:{action}, reward:{r} \n')