"""Phase-level deadline feedback, separate from teacher reward and RM deadlines."""
from dataclasses import dataclass
import copy
import math
from ermc._qualified import FactorWrapper, Settings as FactorSettings

@dataclass
class Settings(FactorSettings):
    end_on_hit: bool = False
    schedule_mode: str = 'adaptive'
    schedule_alpha: float = 0.1
    schedule_target: float = 0.7
    schedule_min: int = 8
    schedule_max: int = 32
    schedule_initial_p: float = 0.7
    schedule_scope: str = 'all'
    schedule_stage_max: tuple = None

    def __post_init__(self):
        super().__post_init__()
        if self.schedule_scope not in ('all', 'u0'):
            raise ValueError('Unknown scheduler scope')
        if self.schedule_mode not in ('fixed32', 'fixed28', 'adaptive'):
            raise ValueError('Unknown schedule mode')
        if not (0 < self.schedule_alpha <= 1 and 0 <= self.schedule_target <= 1):
            raise ValueError('Invalid feedback parameters')
        if not 1 <= self.schedule_min <= self.schedule_max:
            raise ValueError('Invalid horizon range')
        if self.schedule_stage_max is not None:
            self.schedule_stage_max = tuple(self.schedule_stage_max)
            if len(self.schedule_stage_max) != 3 or any((type(v) is not int or not self.schedule_min <= v <= self.schedule_max for v in self.schedule_stage_max)):
                raise ValueError('Expected three integer stage caps within the horizon range')

class Feedback:

    def __init__(self, cfg):
        self.cfg = cfg
        self.p = [cfg.schedule_initial_p] * 3
        self.d = [0.0] * 3
        self.trials = [0] * 3
        self.hits = [0] * 3

    def update(self, event):
        if event['action'] == 0 or (self.cfg.schedule_scope == 'u0' and event['stage'] != 0):
            return False
        stage, hit, alpha = (event['stage'], int(event['hit']), self.cfg.schedule_alpha)
        self.trials[stage] += 1
        self.hits[stage] += hit
        self.p[stage] = (1 - alpha) * self.p[stage] + alpha * hit
        self.d[stage] = min(1.0, max(0.0, (1 - alpha) * self.d[stage] + alpha * (self.p[stage] - self.cfg.schedule_target)))
        return True

    def state(self):
        return copy.deepcopy(dict(p=self.p, d=self.d, trials=self.trials, hits=self.hits))

def deadline(cfg, state, stage, action):
    if action == 0 or (cfg.schedule_scope == 'u0' and stage != 0):
        return 32
    if cfg.schedule_mode == 'fixed32':
        return 32
    if cfg.schedule_mode == 'fixed28':
        return 28
    upper = cfg.schedule_stage_max[stage] if cfg.schedule_stage_max is not None else cfg.schedule_max
    raw = cfg.schedule_min + (1 - state['d'][stage]) * (upper - cfg.schedule_min)
    return max(cfg.schedule_min, min(upper, math.floor(raw)))

class SchedulerWrapper(FactorWrapper):

    def __init__(self, env, settings=None):
        cfg = settings or Settings()
        self.schedule = Feedback(cfg).state()
        self.schedule_events = []
        super().__init__(env, cfg)

    def set_scheduler(self, state):
        self.schedule = copy.deepcopy(state)

    def set_teacher(self, weights):
        if weights is not None and '_scheduler' in weights:
            self.set_scheduler(weights['_scheduler'])
        return super().set_teacher(weights)

    def choose(self, sensed):
        super().choose(sensed)
        a = self.active
        a.update(deadline=deadline(self.cfg, self.schedule, a['stage'], a['action']), scheduler_p_start=self.schedule['p'][a['stage']], scheduler_d_start=self.schedule['d'][a['stage']], scheduler_trials_start=self.schedule['trials'][a['stage']])

    def option_timeout(self, active):
        return active['elapsed'] >= active['deadline']

    def option_ends(self, active, hit, timeout, progress, done):
        if self.cfg.schedule_scope == 'u0' and active['stage'] != 0:
            active['any_hit'] = active.get('any_hit', False) or bool(hit)
            return timeout or progress or done
        return super().option_ends(active, hit, timeout, progress, done)

    def option_finished(self, active, boundary, done, progress):
        self.schedule_events.append(dict(stage=active['stage'], action=active['action'], elapsed=active['elapsed'], deadline=active['deadline'], hit=active['hit'], timeout=bool(self.option_timeout(active) and (not active['hit'])), boundary=bool(boundary), done=bool(done), progress=bool(progress), rm_changed=bool(self.rm_state != active['stage']), p_start=active['scheduler_p_start'], d_start=active['scheduler_d_start']))

    def step(self, action):
        self.schedule_events = []
        obs, reward, done, info = super().step(action)
        if self.schedule_events:
            info['scheduler_events'] = self.schedule_events
        return (obs, reward, done, info)
