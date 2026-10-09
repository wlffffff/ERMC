"""Observable key predicates layered on the unchanged ERMC RM wrapper."""
from dataclasses import dataclass, asdict
import numpy as np
from ermc.reward_machine import ERMCRewardMachine
DIRECT, DISCOVER_KEY, READY_KEY = range(3)
CONTEXT_DIM = 10

@dataclass
class Settings:
    mode: str = 'learned'
    horizon: int = 32
    gamma: float = 0.99
    auxiliary_bonus: float = 0.005

    def __post_init__(self):
        if self.mode not in ('baseline', 'random', 'learned'):
            raise ValueError('Invalid teacher mode')
        if self.horizon <= 0 or self.auxiliary_bonus < 0:
            raise ValueError('Invalid teacher horizon or bonus')

class U0TeacherWrapper(ERMCRewardMachine):

    def __init__(self, env, settings=None):
        super().__init__(env)
        self.cfg = settings or Settings()
        self.rng = np.random.RandomState(0)
        self.weights = None
        self.goal = DIRECT
        self.active = None
        self.frozen = False

    def seed(self, seed=None):
        self.rng.seed(seed)
        return self.env.seed(seed)

    def set_teacher(self, weights):
        self.weights = weights

    def freeze(self, enabled=True):
        self.frozen = enabled

    def _encode(self, obs):
        return self._stamp(super()._encode(obs))

    def censor(self):
        count = len(self.decisions) + int(self.active is not None)
        self.decisions = []
        self.active = None
        return count
