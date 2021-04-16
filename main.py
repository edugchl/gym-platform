from gym_platform import PlatformEnv, UserEnv, WorldEnv

world_env = WorldEnv()
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env)

terminal = False
obs = platform.reset()
while not terminal:
    action = 1
    new_obs, r, terminal, _ = platform.step(action)
    print(new_obs, r)