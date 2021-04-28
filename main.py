import time
from dateutil.relativedelta import relativedelta

import numpy as np
from stable_baselines3 import A2C
import wandb

from gym_platform import PlatformEnv, UserEnv, WorldEnv
from gym_platform.envs.utils import flatten_dict


EPISODES = 100

wandb.init(project='rl-notification-system')
world_env = WorldEnv(ratio=36000)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env, 20*24)

model = A2C('MlpPolicy', platform, verbose=1)
model.learn(total_timesteps=20*24)

obs = platform.reset()

for eps in range(EPISODES):
    trajectory, states = [], []
    _debug_freeness, _debug_burden = [], []
    cumulative_rewards, step = 0, 0
    terminal = False
    while not terminal:
        time.sleep(0.1)
        # action = np.random.choice(2)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminal, _ = platform.step(action)
        step += 1; cumulative_rewards+=reward
        trajectory.append([eps, step] + list(obs) + [action, reward])
        states.append([eps, step] + list(flatten_dict(user_env.state).values()))
        print(f'state: {platform.user_env.state}\n, obs: {obs}\n, action:{action}, reward:{reward} \n')
        if terminal:
            obs = platform.reset()
            wandb.log({'cumulative rewards': cumulative_rewards})
            trajectory_table = wandb.Table(data=trajectory, columns=['episode', 'step']+platform.observation_name+['action', 'reward'])
            states_table = wandb.Table(data=states, columns=['episode', 'step']+list(flatten_dict(user_env.state).keys()))

            line_plot = wandb.plot.line(trajectory_table, x='step', y='reward', title='Line Plot')
            histogram = wandb.plot.histogram(trajectory_table, value='reward', title='Histogram')
            scatter = wandb.plot.scatter(trajectory_table, x='step', y='reward', title='Scatter Plot')

            wandb.log({'line_1': line_plot, 
                        'histogram_1': histogram, 
                        'scatter_1': scatter,
                        'observation_table': trajectory_table, 
                        'state_table': states_table, 
                    })

                


