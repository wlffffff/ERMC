"""Algorithm checks and short real-environment integration, not experiments."""
import unittest
import gym
import numpy as np
import torch
from torch import nn
from gym_minigrid.minigrid import Grid, Key, Door, Ball
from ermc import Settings, ERMCWrapper, make_environment, CurriculumController
from ermc import SemanticEncoding, ResidualHead, combine_rewards
from ermc.goals import GOALS, N_GOALS
from ermc.scheduler import Feedback, deadline
from ermc.teacher import TeacherLearner


class EventEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (7, 7, 3), dtype=np.uint8)})
        self.action_space = gym.spaces.Discrete(7)
        self.max_steps, self.obj_type = 270, 'ball'
        self.agent_pos = (3, 3)
        self.grid = Grid(7, 7)
        self.door = Door('yellow', is_locked=True)
        self.grid.set(5, 3, self.door)
        self.key, self.ball = Key('yellow'), Ball('red')
        self.carrying, self.front, self.visible = None, None, []
        self.event = ''

    def gen_obs(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        for i, obj in enumerate(self.visible):
            image[i, 2] = obj.encode()
        if self.front is not None:
            image[3, 5] = self.front.encode()
        if self.carrying is not None:
            image[3, 6] = self.carrying.encode()
        return {'image': image}

    def reset(self):
        return self.gen_obs()

    def step(self, action):
        event, self.event = self.event, ''
        if event == 'see_key':
            self.visible = [self.key]
        if event == 'key':
            self.carrying, self.front = self.key, None
            self.visible = [obj for obj in self.visible if obj.type != 'key']
        if event == 'open':
            self.door.is_open, self.door.is_locked = True, False
        if event == 'ball':
            self.carrying, self.front = self.ball, None
        return self.gen_obs(), float(event == 'ball'), event in ('ball', 'timeout'), {}


class Tests(unittest.TestCase):
    def setUp(self):
        self.env = ERMCWrapper(EventEnvironment(), Settings())
        self.env.reset()

    def event(self, event, action=0):
        self.env.unwrapped.event = event
        return self.env.step(action)

    def test_defaults(self):
        cfg = self.env.cfg
        self.assertTrue(cfg.color_goals)
        self.assertTrue(cfg.fix_u2_failed_done)
        self.assertFalse(cfg.end_on_hit)
        self.assertEqual(cfg.credit, 'direct')
        self.assertEqual(cfg.schedule_mode, 'adaptive')

    def test_hold_and_no_repeat_bonus(self):
        goal = GOALS.index((1, 1))
        self.env.goal = self.env.active['action'] = goal
        active = self.env.active
        _, _, _, info = self.event('see_key')
        self.assertEqual(info['auxiliary_reward'], .005)
        self.assertIs(self.env.active, active)
        self.assertEqual(self.event('')[3]['auxiliary_reward'], 0.)
        for _ in range(40):
            self.event('')
        self.assertLessEqual(self.env.aux_return, .01)

    def test_phase_reward_and_failure(self):
        info = self.event('key', 3)[3]
        self.assertEqual(self.env.rm_state, 1)
        self.assertEqual(info['ermc_intrinsic_reward'], 1.)
        self.assertEqual(info['auxiliary_reward'], 0.)
        self.event('open', 5)
        self.assertEqual(self.env.rm_state, 2)
        _, reward, done, info = self.event('timeout')
        self.assertTrue(done)
        self.assertEqual(reward, 0.)
        self.assertEqual(info['ermc_option_reached'], 0.)
        self.assertEqual(info['ermc_intrinsic_reward'], 0.)
        self.assertEqual(info['teacher_trajectory'][-1]['reward'], 0.)
        self.assertEqual(info['episode_metrics']['success'], 0.)

    def test_final_success(self):
        self.event('key', 3)
        self.event('open', 5)
        _, _, done, info = self.event('ball', 3)
        self.assertTrue(done)
        self.assertEqual(self.env.rm_state, 3)
        self.assertEqual(info['teacher_phase_progress'], 1.)
        self.assertEqual(info['episode_metrics']['success'], 1.)

    def test_color_binding(self):
        self.event('key', 3)
        self.assertEqual(self.env.bound_color, self.env.unwrapped.key.encode()[1]+1)
        self.env.unwrapped.front = Door('blue', is_locked=False)
        self.assertFalse(self.env.sense()['ready'][1])
        self.env.unwrapped.front = self.env.unwrapped.door
        self.assertTrue(self.env.sense()['ready'][1])

    def test_no_fixed_direct_quota(self):
        mask = self.env.eligible(self.env.sense())
        prior = self.env.prior(mask)
        self.assertAlmostEqual(prior.sum(), 1.)
        self.assertAlmostEqual(prior[0], 1/mask.sum())
        self.assertNotEqual(prior[0], .5)
        self.env.achieved = set(range(1, N_GOALS))
        self.assertEqual(self.env.prior(self.env.eligible(self.env.sense()))[0], 1.)

    def test_feedback_and_deadline_lock(self):
        cfg = self.env.cfg
        feedback = Feedback(cfg)
        self.assertFalse(feedback.update(dict(stage=0, action=0, hit=True)))
        locked = self.env.active['deadline']
        for _ in range(100):
            feedback.update(dict(stage=0, action=1, hit=True))
        self.assertLess(deadline(cfg, feedback.state(), 0, 1), 32)
        self.env.set_scheduler(feedback.state())
        self.assertEqual(self.env.active['deadline'], locked)

    def test_residual_initialization_and_routing(self):
        layer = nn.Linear(5, 2)
        head = ResidualHead(layer)
        data = torch.randn(4, 5)
        head.phase = torch.arange(4)
        torch.testing.assert_close(head(data), layer(data))
        head(data)[1].sum().backward()
        grads = head.residual.weight.grad.reshape(4, 2, 5)
        self.assertGreater(float(grads[1].abs().sum()), 0)
        self.assertEqual(float(grads[[0, 2, 3]].abs().sum()), 0)

    def test_separate_goal_encoding(self):
        obs = torch.tensor(self.env._encode(self.env.unwrapped.gen_obs())['image'])
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        encoder = SemanticEncoding()
        self.assertEqual(encoder(obs).shape, (1, 16))
        torch.testing.assert_close(encoder(obs), encoder(obs.float()/255, normalized=True))

    def test_reward_budget(self):
        base = np.ones(20)
        raw = np.full(20, .005)
        result, metrics = combine_rewards(base, raw)
        self.assertLessEqual(metrics['auxiliary/paid_sum'], .00100001)
        self.assertTrue(np.all(result >= base))
        self.assertAlmostEqual(metrics['auxiliary/scale'], .01)

    def test_teacher_update_and_frozen_counts(self):
        controller = CurriculumController(seed=3)
        controller.publish([self.env])
        info = self.event('key', 3)[3]
        controller.observe([info])
        metrics = controller.update_teacher()
        self.assertTrue(np.isfinite(metrics['teacher_loss']))
        self.assertEqual(metrics['teacher_contrast_supported_fraction'], 0.)
        weights = controller.evaluation_weights([self.env])
        evaluation = ERMCWrapper(EventEnvironment(), Settings())
        evaluation.set_teacher(weights)
        evaluation.freeze(True)
        evaluation.reset()
        before = evaluation.reachability_state()
        evaluation.unwrapped.event = 'key'
        evaluation.step(3)
        np.testing.assert_array_equal(before['trials'], evaluation.reach_trials)

    def test_supported_preference_learns(self):
        learner = TeacherLearner(seed=8)
        context = self.env.context(self.env.sense()).tolist()
        mask = [True, True] + [False]*(N_GOALS-2)
        prior = [.5, .5] + [0.]*(N_GOALS-2)
        for _ in range(100):
            with torch.no_grad():
                logits, _ = learner.model(torch.tensor([context]),
                    torch.tensor([mask]), torch.tensor([prior]))
                logp = logits.log_softmax(-1)[0]
            trajectories = [[dict(context=context, mask=mask, prior=prior,
                action=a, logp=float(logp[a]), stage=0, reward=float(a == 1),
                discount=0., hit=False, elapsed=1)] for a in (0, 1)]
            metrics = learner.update(trajectories)
        self.assertGreater(metrics['teacher_contrast_supported_fraction'], 0.)
        logits, _ = learner.model(torch.tensor([context]),
            torch.tensor([mask]), torch.tensor([prior]))
        self.assertGreater(float(logits.softmax(-1)[0, 1]), .5)
        q = learner.model.q_values(torch.tensor([context]))[0]
        self.assertGreater(float(q[1]), float(q[0]))

    def test_real_s3r3_integration(self):
        env = make_environment(seed=4)
        controller = CurriculumController(seed=4)
        controller.publish([env])
        obs = env.reset()
        try:
            self.assertEqual(obs['image'].shape, (7, 7, 4))
            for step in range(600):
                obs, _, done, info = env.step(step % 7)
                controller.observe([info])
                controller.publish([env])
                if done:
                    env.reset()
                if step % 64 == 63:
                    controller.update_teacher()
                    controller.publish([env])
            metrics = controller.episode_metrics()
            self.assertEqual(metrics['Step'], 600)
            self.assertGreater(metrics['completed_episodes'], 0)
            self.assertIn('mean_episode_return', metrics)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
