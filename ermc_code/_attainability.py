"""Task-schema type mask and observed, per-stage reachability prior."""
from dataclasses import dataclass
import numpy as np
from ermc._semantic import SemanticWrapper, Settings as OriginalSettings, teacher_numpy
from ermc.goals import GOALS, OBJECTS, N_GOALS, STATE_COUNT

@dataclass
class Settings(OriginalSettings):
    refinement: bool = True
    credit: str = 'legacy'
    reachability_floor: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.credit not in ('legacy', 'direct') or not 0 < self.reachability_floor <= 1:
            raise ValueError('Invalid credit estimator or reachability floor')

class RefinedWrapper(SemanticWrapper):

    def __init__(self, env, settings=None):
        super().__init__(env, settings or Settings())
        self.supported_types = {'key', 'door', str(getattr(self.unwrapped, 'obj_type', 'ball'))}
        self.reach_trials = np.zeros((STATE_COUNT, N_GOALS), dtype=np.float64)
        self.reach_hits = np.zeros((STATE_COUNT, N_GOALS), dtype=np.float64)
        self.stage_exposed = np.zeros(N_GOALS, dtype=bool)

    def set_teacher(self, weights):
        self.weights = dict(weights)
        reach = self.weights.pop('_reachability', None)
        if reach is not None:
            self.reach_trials = np.asarray(reach['trials']).copy()
            self.reach_hits = np.asarray(reach['hits']).copy()

    def reachability_state(self):
        return dict(trials=self.reach_trials.copy(), hits=self.reach_hits.copy())

    def reset(self):
        self.stage_exposed = np.zeros(N_GOALS, dtype=bool)
        return super().reset()
