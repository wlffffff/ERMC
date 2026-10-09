"""ERMC algorithm modules for KeyCorridorS3R3, without a training framework.

Use make_environment() and CurriculumController with a host RL learner.
The host owns policy optimization, base-reward preprocessing and evaluation.
Metadata cells must be embedded separately from the three image channels.
"""
from ermc.scheduler import Settings, SchedulerWrapper as ERMCWrapper
from ermc.factory import make_environment
from ermc.controller import CurriculumController
from ermc.encoding import SemanticEncoding
from ermc.heads import ResidualHead
from ermc.rewards import combine_rewards

__all__ = ['Settings', 'ERMCWrapper', 'make_environment', 'CurriculumController',
           'SemanticEncoding', 'ResidualHead', 'combine_rewards']
