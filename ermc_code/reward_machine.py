import os

import gym
import numpy as np
from gym import spaces


RM_KEY_NOT_PICKED = 0
RM_KEY_PICKED = 1
RM_DOOR_OPEN = 2
RM_GOAL_REACHED = 3

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_DROP = 4
ACTION_TOGGLE = 5


def _objects(env):
    base = env.unwrapped
    found = {}
    for x in range(base.grid.width):
        for y in range(base.grid.height):
            obj = base.grid.get(x, y)
            if obj and obj.type in ("key", "door", "ball", "box"):
                found.setdefault(obj.type, []).append(((x, y), obj))
    return found


class ERMCRewardMachine(gym.Wrapper):
    """KeyCorridor event reward machine and weak no-progress regularizer.

    The policy sees the normal partial MiniGrid image plus one option/RM-state channel.
    Privileged symbolic state is used only for reward and logging.
    """

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space.spaces["image"]
        h, w, c = old.shape
        spaces_dict = dict(env.observation_space.spaces)
        spaces_dict["image"] = spaces.Box(0, 255, (h, w, c + 1), dtype=np.uint8)
        self.observation_space = spaces.Dict(spaces_dict)

        self.intrinsic_event_reward = float(os.environ.get("ERMC_EVENT_REWARD", "1.0"))
        self.anti_stuck_cost = float(os.environ.get("ERMC_ANTI_STUCK_COST", "0.005"))
        self.anti_stuck_grace = int(os.environ.get("ERMC_ANTI_STUCK_GRACE", "30"))
        self.deadlines = {
            RM_KEY_NOT_PICKED: int(os.environ.get("ERMC_DEADLINE_U0", "7")),
            RM_KEY_PICKED: int(os.environ.get("ERMC_DEADLINE_U1", "9")),
            RM_DOOR_OPEN: int(os.environ.get("ERMC_DEADLINE_U2", "9")),
        }
        self.rm_state = RM_KEY_NOT_PICKED
        self.goal_elapsed = 0
        self.stuck_streak = 0
        self.matching_door_pos = None
        self.target_pos = None
        self.prev_agent_pos = None

    def _encode(self, obs):
        goal_channel = np.full((*obs["image"].shape[:2], 1), self.rm_state, dtype=np.uint8)
        out = dict(obs)
        out["image"] = np.concatenate([obs["image"], goal_channel], axis=-1)
        return out

    def reset(self):
        obs = self.env.reset()
        self.rm_state = RM_KEY_NOT_PICKED
        self.goal_elapsed = 0
        self.stuck_streak = 0
        found = _objects(self.env)
        locked_doors = [(pos, obj) for pos, obj in found.get("door", []) if getattr(obj, "is_locked", False)]
        targets = found.get("ball", []) + found.get("box", [])
        self.matching_door_pos = tuple(locked_doors[0][0]) if locked_doors else None
        self.target_pos = tuple(targets[0][0]) if targets else None
        self.prev_agent_pos = tuple(self.env.unwrapped.agent_pos)
        return self._encode(obs)

    def _matching_door(self):
        if self.matching_door_pos is None:
            return None
        return self.env.unwrapped.grid.get(*self.matching_door_pos)

    def _infer_rm_state(self, reward, done):
        carrying = self.env.unwrapped.carrying
        door = self._matching_door()
        has_key = carrying is not None and carrying.type == "key"
        has_target = carrying is not None and carrying.type in ("ball", "box")

        next_state = self.rm_state
        if next_state < RM_KEY_PICKED and has_key:
            next_state = RM_KEY_PICKED
        if next_state < RM_DOOR_OPEN and door is not None and getattr(door, "is_open", False):
            next_state = RM_DOOR_OPEN
        if reward > 0:
            next_state = RM_GOAL_REACHED
        return next_state

    def _subgoal_reached(self, next_state, reward, done):
        if self.rm_state == RM_KEY_NOT_PICKED:
            return next_state >= RM_KEY_PICKED
        if self.rm_state == RM_KEY_PICKED:
            return next_state >= RM_DOOR_OPEN
        if self.rm_state == RM_DOOR_OPEN:
            return reward > 0
        return False

    def step(self, action):
        prev_pos = tuple(self.env.unwrapped.agent_pos)
        obs, ext_reward, done, info = self.env.step(action)
        new_pos = tuple(self.env.unwrapped.agent_pos)

        old_state = self.rm_state
        next_state = self._infer_rm_state(ext_reward, done)
        reached = self._subgoal_reached(next_state, ext_reward, done)

        interaction_action = int(action) in (ACTION_PICKUP, ACTION_DROP, ACTION_TOGGLE)
        same_position = float(new_pos == prev_pos)
        no_progress = bool(same_position and not interaction_action and not done)
        self.stuck_streak = self.stuck_streak + 1 if no_progress else 0
        repeated_stuck = self.stuck_streak >= self.anti_stuck_grace
        anti_stuck_penalty = -self.anti_stuck_cost if repeated_stuck else 0.0

        self.goal_elapsed += 1
        deadline = self.deadlines.get(old_state, self.deadlines[RM_DOOR_OPEN])
        deadline_hit = float(self.goal_elapsed >= deadline and not reached and not done)
        refresh_reason = 0
        if reached:
            refresh_reason = 1
            self.rm_state = next_state
            self.goal_elapsed = 0
        elif done:
            refresh_reason = 3
            self.rm_state = RM_GOAL_REACHED if ext_reward > 0 else RM_KEY_NOT_PICKED
            self.goal_elapsed = 0
        else:
            self.rm_state = next_state

        intrinsic = self.intrinsic_event_reward * float(reached) + anti_stuck_penalty

        info.update(
            ermc_intrinsic_reward=intrinsic,
            ermc_stage=float(self.rm_state),
            ermc_success=float(ext_reward > 0),
            ermc_option_reached=float(reached),
            ermc_option_timeout=deadline_hit,
            ermc_rm_state=float(self.rm_state),
            rm_state_u0_rate=float(self.rm_state == RM_KEY_NOT_PICKED),
            rm_state_u1_rate=float(self.rm_state == RM_KEY_PICKED),
            rm_state_u2_rate=float(self.rm_state == RM_DOOR_OPEN),
            rm_state_uF_rate=float(self.rm_state == RM_GOAL_REACHED),
            goal_success_rate=float(reached),
            goal_success_u0_rate=float(reached and old_state == RM_KEY_NOT_PICKED),
            goal_success_u1_rate=float(reached and old_state == RM_KEY_PICKED),
            goal_success_u2_rate=float(reached and old_state == RM_DOOR_OPEN),
            goal_refresh_idle_rate=float(refresh_reason == 0),
            goal_refresh_success_rate=float(refresh_reason == 1),
            goal_refresh_deadline_rate=deadline_hit,
            goal_refresh_episode_end_rate=float(refresh_reason == 3),
            deadline_hit_rate=deadline_hit,
            mean_goal_elapsed=float(self.goal_elapsed),
            mean_goal_deadline=float(deadline),
            mean_rm_state=float(self.rm_state),
            anti_stuck_penalty_mean=float(anti_stuck_penalty),
            same_position_rate=same_position,
            same_position_nonterminal_rate=float(same_position and not done),
            turn_same_position_rate=float(same_position and int(action) in (ACTION_LEFT, ACTION_RIGHT)),
            no_progress_rate=float(no_progress),
            stuck_streak_mean=float(self.stuck_streak),
            stuck_streak_ge_grace_rate=float(repeated_stuck),
            action_left_rate=float(int(action) == ACTION_LEFT),
            action_right_rate=float(int(action) == ACTION_RIGHT),
            action_forward_rate=float(int(action) == ACTION_FORWARD),
            action_pickup_rate=float(int(action) == ACTION_PICKUP),
            action_toggle_rate=float(int(action) == ACTION_TOGGLE),
        )
        self.prev_agent_pos = new_pos
        return self._encode(obs), ext_reward, done, info
