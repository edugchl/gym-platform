import time
from dateutil.relativedelta import relativedelta

import numpy as np
from stable_baselines3 import A2C
import wandb

from gym_platform import PlatformEnv, UserEnv, WorldEnv


EPISODES = 20

wandb.init(project='rl-notification-system')
world_env = WorldEnv(ratio=36000)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env, 2*24)

model = A2C('MlpPolicy', platform, verbose=1)
model.learn(total_timesteps=1*24)

cumulative_rewards = []
obs = platform.reset()

for eps in range(EPISODES):
    trajectory = []
    step = 0
    terminal = False
    while not terminal:
        time.sleep(0.1)
        # action = np.random.choice(2)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminal, _ = platform.step(action)
        step += 1
        trajectory.append([eps, step, platform.world_env.current_time.strftime("%m-%d-%Y_%H-%M-%S"), action, reward])
        print(f'state: {platform.user_env.state}\n, obs: {obs}\n, action:{action}, reward:{reward} \n')
        if terminal:
            obs = platform.reset()
            cumulative_rewards.append(reward)
            table = wandb.Table(data=trajectory, columns=['episode', 'step', 'datetime', 'action', 'reward'])

            # Use the table to populate various custom charts
            line_plot = wandb.plot.line(table, x='step', y='reward', title='Line Plot')
            line_plot_dt = wandb.plot.line(table, x='datetime', y='reward', title='Line Plot')
            histogram = wandb.plot.histogram(table, value='reward', title='Histogram')
            scatter = wandb.plot.scatter(table, x='step', y='reward', title='Scatter Plot')
            
            # Log custom tables, which will show up in customizable charts in the UI
            wandb.log({'line_1': line_plot, 
                        'line_dt': line_plot_dt,
                        'histogram_1': histogram, 
                        'scatter_1': scatter})

                


