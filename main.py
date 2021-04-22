import time
from dateutil.relativedelta import relativedelta

import numpy as np
from stable_baselines3 import A2C
import wandb

from gym_platform import PlatformEnv, UserEnv, WorldEnv


wandb.init(project='rl-notification-system')
world_env = WorldEnv(ratio=3600)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env, 30*24)

model = A2C('MlpPolicy', platform, verbose=0)
model.learn(total_timesteps=1000)

terminal = False
obs = platform.reset()
while not terminal:
    time.sleep(1)
    # action = np.random.choice(2)
    action, _state = model.predict(obs, deterministic=True)
    new_obs, r, terminal, _ = platform.step(action)
    wandb.log({'reward':r})
    print(f'state: {platform.user_env.state}\n, obs: {new_obs}\n, action:{action}, reward:{r} \n')