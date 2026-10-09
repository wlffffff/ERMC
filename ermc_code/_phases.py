from dataclasses import dataclass
import numpy as np
from ermc._holding import ControlledWrapper, Settings as PreviousSettings
from ermc.goals import GOALS, OBJECTS, OBJECT_CODES, N_GOALS, name, ACTIVE_STAGES
from ermc.reward_machine import ERMCRewardMachine


@dataclass
class Settings(PreviousSettings):
    stage_scope: str = 'all'

    def __post_init__(self):
        super().__post_init__()
        if self.stage_scope not in ('u0', 'all'):
            raise ValueError('stage_scope must be u0 or all')


class StageWrapper(ControlledWrapper):
    key_start_stage = 0
    door_reached_stage = 2
    def __init__(self, env, settings=None):
        super().__init__(env, settings or Settings())
        self.target_type = str(getattr(self.unwrapped, 'obj_type', 'ball'))

    def enabled(self, stage):
        return self.cfg.mode != 'baseline' and (stage == 0 or self.cfg.stage_scope == 'all' and stage in (1, 2))

    def _stamp(self, obs):
        result = dict(obs)
        result['image'] = obs['image'].copy()
        obj, predicate = GOALS[self.goal if self.enabled(self.rm_state) else 0]
        result['image'][0, 1, 3] = obj
        result['image'][0, 2, 3] = predicate
        return result

    def eligible(self, sensed):
        mask = np.zeros(N_GOALS, dtype=bool)
        mask[0] = True
        if not self.enabled(self.rm_state):
            return mask
        # Keep u0 identical to the preceding filtered library. Later stages use
        # the public task schema: unlock door, then reach/acquire the task object.
        types = self.supported_types if self.rm_state == 0 else (
            {'door'} if self.rm_state == 1 else {'door', self.target_type})
        for i, (obj, predicate) in enumerate(GOALS[1:], 1):
            mask[i] = (OBJECTS[obj] in types and i not in self.achieved
                       and (predicate == 1 or self.seen[obj-1]))
        return mask

    def choose(self, sensed):
        super().choose(sensed)
        stage = min(self.rm_state, 2)
        self.active['stage'] = stage
        self.stage_proposals[stage, self.goal] += 1
        self.stage_direct_sum[stage] += self.active['direct_probability']

    def reset(self):
        self.stage_entered = np.zeros(ACTIVE_STAGES, dtype=bool)
        self.stage_entered[0] = True
        self.stage_steps = np.zeros(ACTIVE_STAGES, dtype=int)
        self.stage_progress = np.zeros(ACTIVE_STAGES, dtype=int)
        self.stage_aux_steps = np.zeros(ACTIVE_STAGES, dtype=int)
        self.stage_aux_return = np.zeros(ACTIVE_STAGES)
        self.stage_proposals = np.zeros((ACTIVE_STAGES, N_GOALS), dtype=int)
        self.stage_hits = np.zeros((ACTIVE_STAGES, N_GOALS), dtype=int)
        self.stage_direct_sum = np.zeros(ACTIVE_STAGES)
        self.false_terminal_events = 0
        return super().reset()

    def begin_stage(self, sensed):
        self.goal, self.active = 0, None
        self.decisions = []
        self.stage_exposed = np.zeros(N_GOALS, dtype=bool)
        self.achieved = self.facts(sensed)
        # An already-held object is not a new acquisition in the new stage.
        for i, (obj, predicate) in enumerate(GOALS):
            if predicate == 3 and not sensed['empty'] and sensed['inventory'][0] == OBJECT_CODES[obj]:
                self.achieved.add(i)
        self.last_hit = self.last_timeout = 0.
        if self.rm_state in range(ACTIVE_STAGES):
            self.stage_entered[self.rm_state] = True
        if self.enabled(self.rm_state):
            self.choose(sensed)

    def option_timeout(self, active):
        return active['elapsed'] >= self.cfg.horizon

    def option_finished(self, active, boundary, done, progress):
        pass

    def _rm_step(self, action):
        return ERMCRewardMachine.step(self, action)

    def step(self, action):
        stage, before = self.rm_state, self.sense()
        selected = self.goal if self.enabled(stage) else 0
        if stage in range(ACTIVE_STAGES):
            self.stage_steps[stage] += 1
            self.stage_aux_steps[stage] += int(selected != 0)
            self.stage_exposed |= self.eligible(before)
        obs, reward, done, info = self._rm_step(action)
        self.elapsed += 1
        self.ext_return += reward
        self.rm_return += info['ermc_intrinsic_reward']
        self.anti_return += info['anti_stuck_penalty_mean']
        sensed = self.sense()
        facts = self.facts(sensed) | self.effects(before, sensed, action)
        new_facts = facts - self.achieved
        self.achieved |= facts
        self.seen |= sensed['visible']
        if self.first_visible_key is None and sensed['visible'][0]:
            self.first_visible_key = self.elapsed
        if self.first_ready_key is None and sensed['ready'][0]:
            self.first_ready_key = self.elapsed
        # Only a successful task event completes the final phase.
        progress = bool((stage in range(ACTIVE_STAGES-1) and self.rm_state > stage)
                        or (stage == ACTIVE_STAGES-1 and reward > 0))
        boundary = progress or done or self.rm_state != stage
        if stage in range(ACTIVE_STAGES):
            self.stage_progress[stage] += int(progress)
        if self.first_key is None and stage == self.key_start_stage and progress:
            self.first_key = self.elapsed
        if self.first_door is None and self.rm_state >= self.door_reached_stage:
            self.first_door = self.elapsed
        if stage == ACTIVE_STAGES-1 and done and reward <= 0 and info['ermc_option_reached']:
            self.false_terminal_events += 1

        auxiliary = 0.
        reselect = False
        if self.enabled(stage) and self.active is not None:
            a = self.active
            a['reward'] += self.cfg.gamma**a['elapsed'] * float(progress)
            a['elapsed'] += 1
            hit = selected != 0 and selected in new_facts
            self.goal_hits[selected] += int(hit)
            self.stage_hits[stage, selected] += int(hit)
            if hit and not boundary:
                auxiliary = min(self.cfg.auxiliary_bonus, max(0., self.cfg.auxiliary_cap-self.aux_return))
                self.aux_paid[selected] += int(auxiliary > 1e-12)
            timeout = self.option_timeout(a)
            if self.option_ends(a, hit, timeout, boundary, done):
                ended_hit = self.option_hit(a, hit)
                a.update(discount=0. if boundary else self.cfg.gamma**a['elapsed'], hit=bool(ended_hit))
                self.option_finished(a, boundary, done, progress)
                self.decisions.append(a)
                self.last_hit, self.last_timeout = float(ended_hit), float(timeout and not ended_hit)
                self.active = None
                if boundary:
                    info['teacher_trajectory'] = self.decisions
                    self.decisions = []
                else:
                    reselect = True

        self.aux_return += auxiliary
        if stage in range(ACTIVE_STAGES):
            self.stage_aux_return[stage] += auxiliary
            if boundary and not self.frozen:
                achieved = np.asarray([i in self.achieved for i in range(N_GOALS)])
                self.reach_trials[stage] += self.stage_exposed
                self.reach_hits[stage] += self.stage_exposed & achieved
        if done:
            self.goal, self.active = 0, None
        elif boundary:
            self.begin_stage(sensed)
        elif reselect:
            self.choose(sensed)
        assert self.aux_return <= self.cfg.auxiliary_cap + 1e-9
        info.update(auxiliary_reward=auxiliary, selected_goal=selected,
                    teacher_phase=stage, teacher_phase_progress=float(progress))
        if done:
            info['episode_metrics'] = self.episode_record(reward)
        return self._stamp(obs), reward, done, info

    def episode_record(self, reward):
        def delay(end, start):
            return None if end is None or start is None else max(0, end-start)
        episode = dict(return_ext=self.ext_return, success=float(reward > 0), length=self.elapsed,
            phase_u1=float(self.first_key is not None), phase_u2=float(self.first_door is not None),
            first_key=self.first_key, first_door=self.first_door,
            first_visible_key=self.first_visible_key, first_ready_key=self.first_ready_key,
            visible_to_ready=delay(self.first_ready_key, self.first_visible_key),
            ready_to_key=delay(self.first_key, self.first_ready_key),
            key_wait_capped=self.first_key if self.first_key is not None else self.elapsed,
            key_censored=float(self.first_key is None), auxiliary_return=self.aux_return,
            rm_intrinsic_return=self.rm_return, anti_stuck_return=self.anti_return,
            direct_probability=self.direct_probability_sum/max(1, self.proposals.sum()),
            u0_steps=int(self.stage_steps[0]), auxiliary_active_steps=int(self.stage_aux_steps[0]),
            auxiliary_active_fraction=float(self.stage_aux_steps[0]/max(1,self.stage_steps[0])),
            all_stage_auxiliary_steps=int(self.stage_aux_steps.sum()),
            rm_false_terminal_events=self.false_terminal_events)
        for i in range(N_GOALS):
            for field, values in [('exposure', self.exposures), ('proposals', self.proposals),
                                  ('hits', self.goal_hits), ('paid', self.aux_paid)]:
                episode[name(i)+'_'+field] = int(values[i])
        for stage in range(ACTIVE_STAGES):
            prefix = 'stage_u'+str(stage)+'_'
            episode.update({prefix+'entered':int(self.stage_entered[stage]),
                prefix+'steps':int(self.stage_steps[stage]), prefix+'progress':int(self.stage_progress[stage]),
                prefix+'auxiliary_steps':int(self.stage_aux_steps[stage]),
                prefix+'auxiliary_return':float(self.stage_aux_return[stage]),
                prefix+'direct_probability':float(self.stage_direct_sum[stage]/self.stage_proposals[stage].sum())
                    if self.stage_proposals[stage].sum() else None})
            for goal in range(N_GOALS):
                episode[prefix+name(goal)+'_proposals'] = int(self.stage_proposals[stage,goal])
                episode[prefix+name(goal)+'_hits'] = int(self.stage_hits[stage,goal])
        return episode
