from dataclasses import dataclass
import numpy as np
from ermc._phases import StageWrapper, Settings as StageSettings
from ermc.goals import GOALS, N_GOALS, STATE_COUNT, ACTIVE_STAGES, CONTEXT_DIM as SEMANTIC_CONTEXT_DIM
from ermc._semantic import teacher_numpy

CONTEXT_DIM = SEMANTIC_CONTEXT_DIM + 24


@dataclass
class Settings(StageSettings):
    color_goals: bool = True
    direct_mode: str = 'selectable'
    end_on_hit: bool = False
    credit: str = 'direct'
    fix_u2_failed_done: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.direct_mode not in ('selectable', 'none'):
            raise ValueError('Invalid DIRECT mode')
        if self.credit == 'direct' and self.direct_mode == 'none':
            raise ValueError('Contrast requires real DIRECT reference data')


class FactorWrapper(StageWrapper):
    def _subgoal_reached(self, next_state, reward, done):
        if self.cfg.fix_u2_failed_done and self.rm_state == 2:
            # KeyCorridor success is a positive external reward, not termination.
            return reward > 0
        return super()._subgoal_reached(next_state, reward, done)

    def __init__(self, env, settings=None):
        self.bound_color = 0
        self.color_seen = np.zeros(6, dtype=bool)
        super().__init__(env, settings or Settings())

    def reset(self):
        self.bound_color = 0
        self.color_seen = np.zeros(6, dtype=bool)
        return super().reset()

    def sense(self):
        sensed = super().sense()
        image = self.unwrapped.gen_obs()['image'].copy()
        image[image.shape[0]//2, image.shape[1]-1] = 0
        doors = image[..., 0] == 4
        for color in range(6):
            self.color_seen[color] |= bool(np.any(doors & (image[..., 1] == color)))
        if (self.cfg.color_goals and self.rm_state == 1 and not self.bound_color
                and not sensed['empty'] and sensed['inventory'][0] == 5):
            self.bound_color = int(sensed['inventory'][1])+1
        sensed['qualified_color'] = self.bound_color if self.cfg.color_goals and self.rm_state == 1 else 0
        if sensed['qualified_color']:
            color = sensed['qualified_color']-1
            sensed['visible'][1] = bool(np.any(doors & (image[..., 1] == color)))
            sensed['ready'][1] = bool(sensed['front'][0] == 4 and sensed['front'][1] == color)
        return sensed

    def context(self, sensed):
        original = super().context(sensed)
        extra = np.zeros(24, dtype=np.float32)
        if self.cfg.color_goals:
            bound = sensed['qualified_color']
            held = int(sensed['inventory'][1])+1 if not sensed['empty'] and sensed['inventory'][0] == 5 else 0
            front = int(sensed['front'][1])+1 if sensed['front'][0] == 4 else 0
            extra[:7] = np.eye(7)[bound]
            extra[7:14] = np.eye(7)[held]
            extra[14:21] = np.eye(7)[front]
            # Leave the final three reserved fields zero in every arm: adding
            # lock-state observations here would confound the color ablation.
        return np.concatenate((original, extra))

    def eligible(self, sensed):
        mask = super().eligible(sensed)
        if self.cfg.color_goals and self.rm_state == 1:
            for i, (obj, predicate) in enumerate(GOALS):
                if obj == 2 and predicate != 1:
                    mask[i] &= bool(self.bound_color and self.color_seen[self.bound_color-1])
            # Without a known key color, do not silently issue an ANY door goal.
            if not self.bound_color:
                mask[1:] = False
        if self.cfg.direct_mode == 'none' and mask[1:].any():
            mask[0] = False
        return mask

    def effects(self, before, after, action):
        result = super().effects(before, after, action)
        if before.get('qualified_color'):
            opened = GOALS.index((2, 4))
            if int(before['front'][1])+1 != before['qualified_color']:
                result.discard(opened)
        return result

    def prior(self, mask):
        stage = min(self.rm_state, STATE_COUNT-1)
        weight = np.maximum(.1, (self.reach_hits[stage]+2)/(self.reach_trials[stage]+4))
        # DIRECT has the same neutral pseudo-count weight as a new auxiliary
        # candidate. No reserved mass and no count-dependent 50% rule.
        weight[0] = .5
        weight *= mask
        return weight/weight.sum()

    def choose(self, sensed):
        context, mask = self.context(sensed), self.eligible(sensed)
        prior = self.prior(mask)
        score = np.zeros(N_GOALS)
        if self.cfg.mode == 'learned' and self.weights is not None:
            score = teacher_numpy(context, self.weights)
        score = np.where(mask, score, -np.inf)
        probs = prior*np.exp(score-np.max(score))
        probs /= probs.sum()
        self.goal = int(self.rng.choice(N_GOALS, p=probs))
        stage = min(self.rm_state, ACTIVE_STAGES-1)
        self.active = dict(context=context.tolist(), mask=mask.tolist(), prior=prior.tolist(),
            action=self.goal, logp=float(np.log(probs[self.goal])), elapsed=0, reward=0.,
            direct_probability=float(probs[0]), valid_count=int(mask.sum()), stage=stage,
            goal_color=int(sensed['qualified_color']) if GOALS[self.goal][0] == 2 else 0)
        self.proposals[self.goal] += 1
        self.exposures += mask
        self.direct_probability_sum += probs[0]
        self.stage_proposals[stage, self.goal] += 1
        self.stage_direct_sum[stage] += probs[0]

    def _stamp(self, obs):
        result = super()._stamp(obs)
        result['image'][0, 3, 3] = (self.bound_color if self.cfg.color_goals
            and self.rm_state == 1 and GOALS[self.goal][0] == 2 else 0)
        return result

    def option_ends(self, active, hit, timeout, progress, done):
        ended = super().option_ends(active, hit, timeout, progress, done)
        return ended or (self.cfg.end_on_hit and hit)
