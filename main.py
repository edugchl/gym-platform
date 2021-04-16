import time
from dateutil.relativedelta import relativedelta

import numpy as np
from gym_platform import PlatformEnv, UserEnv, WorldEnv


world_env = WorldEnv(ratio=3600)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env)

terminal = False
obs = platform.reset()
while not terminal:
    time.sleep(1)
    action = np.random.choice(2)
    new_obs, r, terminal, _ = platform.step(action)
    print('state:',platform.user_env.state, '\n', 'obs',new_obs, r, '\n')