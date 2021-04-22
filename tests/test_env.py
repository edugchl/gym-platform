from gym_platform import PlatformEnv, UserEnv, WorldEnv
from stable_baselines3.common.env_checker import check_env


world_env = WorldEnv(ratio=3600)
user_env = UserEnv(world_env, 'WHITE COLLAR', 0)
platform = PlatformEnv(user_env, world_env)

check_env(env)
