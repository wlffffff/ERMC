"""Small host-learner interface; no policy optimizer, logger or file I/O."""
import numpy as np
from ermc.scheduler import Feedback, Settings
from ermc.teacher import TeacherLearner


class CurriculumController:
    """Coordinate learner-side feedback across ordered environment workers.

    Initialization: publish() to workers before their first reset.
    After each vector step: observe(infos), then publish() the scheduler state.
    At each host collection boundary: update_teacher(), then publish().
    Do not drop incomplete phase visits at policy rollout boundaries.
    Evaluation: evaluation_weights() into new workers, then freeze(True),
    without observe() or update_teacher(). Reward combination remains with
    the host learner; see combine_rewards().
    """

    def __init__(self, settings=None, seed=0):
        self.settings = settings or Settings()
        self.teacher = TeacherLearner(seed)
        self.feedback = Feedback(self.settings)
        self.pending = []
        self.steps = 0
        self.episodes = []

    def observe(self, infos):
        self.steps += len(infos)
        for info in infos:
            for event in info.get('scheduler_events', []):
                self.feedback.update(event)
            if info.get('teacher_trajectory'):
                self.pending.append(info['teacher_trajectory'])
            if 'episode_metrics' in info:
                self.episodes.append(info['episode_metrics'])

    def update_teacher(self):
        trajectories, self.pending = self.pending, []
        if self.settings.mode == 'learned':
            return self.teacher.update(trajectories)
        return {}

    def weights(self):
        weights = self.teacher.weights()
        weights.pop('_reachability', None)
        weights['_scheduler'] = self.feedback.state()
        return weights

    def publish(self, environments):
        """For subprocess workers, send weights() through their RPC equivalent."""
        weights = self.weights()
        for env in environments:
            env.set_teacher(weights)

    def evaluation_weights(self, environments):
        """Freeze the mean worker attainment counts, without rebroadcasting them
        to training workers (whose online counts remain independent).
        """
        states = [env.reachability_state() for env in environments]
        if not states:
            raise ValueError('At least one training worker is required')
        weights = self.weights()
        weights['_reachability'] = {
            key: np.mean([state[key] for state in states], axis=0)
            for key in ('trials', 'hits')
        }
        return weights

    def episode_metrics(self):
        """Drain completed episodes; external return is never shaped reward."""
        episodes, self.episodes = self.episodes, []
        metrics = {'Step': self.steps, 'completed_episodes': len(episodes)}
        if episodes:
            metrics.update(
                mean_episode_return=float(np.mean([e['return_ext'] for e in episodes])),
                success_rate=float(np.mean([e['success'] for e in episodes])),
                mean_episode_steps=float(np.mean([e['length'] for e in episodes])),
            )
        return metrics
