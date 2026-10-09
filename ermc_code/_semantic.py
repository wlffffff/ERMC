"""Only u0 assistance; unchanged external reward and original RM reward."""
from dataclasses import dataclass
import numpy as np
from ermc._base import U0TeacherWrapper, Settings as BaseSettings
from ermc.reward_machine import ERMCRewardMachine
from ermc.goals import GOALS, OBJECT_CODES, N_GOALS, FIXED_IDS, GOAL_FEATURES, name, STATE_COUNT

@dataclass
class Settings(BaseSettings):
    pool: str = 'semantic'
    auxiliary_cap: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.pool not in ('fixed', 'semantic') or self.auxiliary_cap < 0:
            raise ValueError('Invalid pool or reward cap')

def teacher_numpy(context, weights):
    inputs = np.concatenate((np.repeat(context[None], N_GOALS, axis=0), GOAL_FEATURES), axis=-1)
    h = np.tanh(inputs @ weights['score_hidden.weight'].T + weights['score_hidden.bias'])
    return (h @ weights['score.weight'].T + weights['score.bias']).ravel()

class SemanticWrapper(U0TeacherWrapper):

    def __init__(self, env, settings=None):
        super().__init__(env, settings or Settings())

    def sense(self):
        image = self.unwrapped.gen_obs()['image'].copy()
        cx, cy = (image.shape[0] // 2, image.shape[1] - 1)
        inventory = image[cx, cy].copy()
        image[cx, cy] = 0
        front = image[cx, cy - 1].copy()
        visible = np.asarray([np.any(image[..., 0] == code) for code in OBJECT_CODES[1:]])
        empty = self.unwrapped.carrying is None
        ready = np.asarray([front[0] == code and (empty or code == 4) for code in OBJECT_CODES[1:]])
        return dict(visible=visible, ready=ready, empty=empty, front=front, inventory=inventory)

    def facts(self, sensed):
        facts = set()
        for i, (o, p) in enumerate(GOALS[1:], 1):
            if p in (1, 2) and sensed['visible' if p == 1 else 'ready'][o - 1]:
                facts.add(i)
        return facts

    def context(self, sensed):
        return np.concatenate((self.seen, sensed['visible'], sensed['ready'], [i in self.achieved for i in range(N_GOALS)], [sensed['empty'], self.elapsed / self.unwrapped.max_steps, self.last_hit, self.last_timeout], np.eye(STATE_COUNT)[min(self.rm_state, STATE_COUNT - 1)])).astype(np.float32)

    def _reset_rm(self):
        return ERMCRewardMachine.reset(self)

    def reset(self):
        self.goal = 0
        obs = self._reset_rm()
        self.elapsed = 0
        sensed = self.sense()
        self.seen = sensed['visible'].copy()
        self.achieved = self.facts(sensed)
        self.decisions = []
        self.last_hit = self.last_timeout = 0.0
        self.first_key = self.first_door = None
        self.first_visible_key = 0 if sensed['visible'][0] else None
        self.first_ready_key = 0 if sensed['ready'][0] else None
        self.ext_return = self.aux_return = self.rm_return = self.anti_return = 0.0
        self.proposals = np.zeros(N_GOALS, dtype=int)
        self.exposures = np.zeros(N_GOALS, dtype=int)
        self.goal_hits = np.zeros(N_GOALS, dtype=int)
        self.aux_paid = np.zeros(N_GOALS, dtype=int)
        self.direct_probability_sum = 0.0
        self.choose(sensed)
        return self._stamp(obs)

    def effects(self, before, after, action):
        found = set()
        for i, (o, p) in enumerate(GOALS):
            code = OBJECT_CODES[o]
            if p == 3 and action == 3 and before['empty'] and (before['front'][0] == code):
                if not after['empty'] and after['inventory'][0] == code:
                    found.add(i)
            if o == 2 and p == 4 and (action == 5) and (before['front'][0] == 4):
                if after['front'][0] == 4 and after['front'][1] == before['front'][1] and (before['front'][2] != 0) and (after['front'][2] == 0):
                    found.add(i)
        return found

    def option_ends(self, active, hit, timeout, progress, done):
        return hit or timeout or progress or done

    def option_hit(self, active, hit):
        return hit
