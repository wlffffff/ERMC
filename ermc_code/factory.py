"""The distribution exposes only the S3R3 task."""
import gym
import gym_minigrid  # Registers the environment with the pinned legacy Gym API.
from ermc.scheduler import Settings, SchedulerWrapper

ENVIRONMENT_ID = 'MiniGrid-KeyCorridorS3R3-v0'


def make_environment(seed=0, settings=None):
    env = SchedulerWrapper(gym.make(ENVIRONMENT_ID), settings or Settings())
    env.seed(seed)
    return env
