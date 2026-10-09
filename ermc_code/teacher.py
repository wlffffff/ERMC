"""Categorical teacher and supported observational DIRECT contrast."""
import numpy as np
import torch
from torch import nn
from ermc.goals import GOAL_FEATURES, N_GOALS, ACTIVE_STAGES, STATE_COUNT
from ermc._qualified import CONTEXT_DIM


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('goal_features', torch.tensor(GOAL_FEATURES))
        self.score_hidden = nn.Linear(CONTEXT_DIM+10, 32)
        self.score = nn.Linear(32, 1)
        self.value_hidden = nn.Linear(CONTEXT_DIM, 32)
        self.value = nn.Linear(32, 1)
        self.q_hidden = nn.Linear(CONTEXT_DIM+10, 32)
        self.q_output = nn.Linear(32, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.q_output.weight)
        nn.init.zeros_(self.q_output.bias)

    def inputs(self, context):
        return torch.cat((context[:, None].expand(-1, N_GOALS, CONTEXT_DIM),
                          self.goal_features[None].expand(len(context), -1, -1)), -1)

    def forward(self, context, mask, prior):
        score = self.score(torch.tanh(self.score_hidden(self.inputs(context)))).squeeze(-1)
        logits = (score+prior.clamp(min=1e-12).log()).masked_fill(~mask, -1e9)
        value = self.value(torch.tanh(self.value_hidden(context))).squeeze(-1)
        return logits, value

    def q_values(self, context):
        return self.q_output(torch.tanh(self.q_hidden(self.inputs(context)))).squeeze(-1).sigmoid()


def context_bin(context):
    # Stage, hand occupancy and current door readiness. These are coarse
    # observational strata, not identical simulator states or causal matching.
    offset = 12 + N_GOALS + 4
    stage = int(np.argmax(context[offset:offset+STATE_COUNT]))
    return min(stage, ACTIVE_STAGES-1)*4+2*int(context[12+N_GOALS])+int(context[10])


class TeacherLearner:
    def __init__(self, seed=0):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.model = Teacher()
        self.actor_parameters = [p for n,p in self.model.named_parameters() if not n.startswith('q_')]
        self.q_parameters = [p for n,p in self.model.named_parameters() if n.startswith('q_')]
        self.optimizer = torch.optim.Adam(self.actor_parameters, lr=1e-3)
        self.q_optimizer = torch.optim.Adam(self.q_parameters, lr=1e-3)
        self.credit = 'direct'
        self.updates = 0
        self.counts = torch.zeros(N_GOALS)
        self.support = torch.zeros(ACTIVE_STAGES*4, N_GOALS, 2)
        self.reachability = None

    def weights(self):
        result = {k:v.detach().cpu().numpy() for k,v in self.model.state_dict().items()}
        if self.reachability is not None:
            result['_reachability'] = self.reachability
        return result

    def update(self, trajectories):
        trajectories = [t for t in trajectories if t]
        if not trajectories:
            return {}
        if any(len({d['stage'] for d in t}) != 1 for t in trajectories):
            raise ValueError('Mixed-stage teacher trajectory')
        flat = [d for t in trajectories for d in t]
        x = torch.tensor([d['context'] for d in flat], dtype=torch.float32)
        mask = torch.tensor([d['mask'] for d in flat], dtype=torch.bool)
        prior = torch.tensor([d['prior'] for d in flat], dtype=torch.float32)
        action = torch.tensor([d['action'] for d in flat])
        z, values = self.model(x, mask, prior)
        dist = torch.distributions.Categorical(logits=z)
        logp = dist.log_prob(action)
        ratios = (logp.detach()-torch.tensor([d['logp'] for d in flat])).exp()
        rho = ratios.clamp(max=1)
        targets, advantages, returns = [torch.zeros_like(values) for _ in range(3)]
        cursor = 0
        for trajectory in trajectories:
            end = cursor+len(trajectory)
            next_target, future = values.new_zeros(()), 0.
            for i in reversed(range(cursor, end)):
                d = flat[i]
                next_value = values[i+1].detach() if i+1 < end else values.new_zeros(())
                delta = rho[i]*(d['reward']+d['discount']*next_value-values[i].detach())
                targets[i] = values[i].detach()+delta+d['discount']*rho[i]*(next_target-next_value)
                advantages[i] = rho[i]*(d['reward']+d['discount']*next_target-values[i].detach())
                next_target = targets[i]
                future = d['reward']+d['discount']*future
                returns[i] = future
            cursor = end
        q = self.model.q_values(x)
        q_loss = .5*(q.gather(1, action[:, None]).squeeze(1)-returns.detach()).square().mean()
        bins = torch.tensor([context_bin(d['context']) for d in flat])
        supported = (self.support[bins, :, 0] >= 8) & (self.support[bins, :, 1] >= 4) & mask
        supported[:, 0] = False
        contrast = q.detach()-q[:, :1].detach()
        if self.credit == 'direct':
            pg = -(dist.probs*contrast*supported).sum(-1).mean()
            prior_kl = (dist.probs*(dist.logits-prior.clamp(min=1e-12).log())).sum(-1).mean()
            regularizer = .005*prior_kl
        else:
            pg = -(logp*advantages.detach()).mean()
            regularizer = -.005*dist.entropy().mean()
            prior_kl = (dist.probs*(dist.logits-prior.clamp(min=1e-12).log())).sum(-1).mean()
        vf = .5*(values-targets.detach()).square().mean()
        loss = pg+vf+regularizer
        if not torch.isfinite(loss) or not torch.isfinite(q_loss):
            raise FloatingPointError('Non-finite teacher update')
        self.optimizer.zero_grad()
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.actor_parameters, .5)
        self.optimizer.step()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_parameters, .5)
        self.q_optimizer.step()
        for b,a,m in zip(bins, action, mask):
            if a == 0:
                # Count only DIRECT samples where the compared candidate was
                # actually available; fallback-only DIRECT is not a reference.
                self.support[b, m, 0] += 1
            else:
                self.support[b, a, 1] += 1
        self.counts += torch.bincount(action, minlength=N_GOALS)
        self.updates += 1
        metrics = dict(teacher_loss=float(loss.detach()), teacher_pg=float(pg.detach()),
            teacher_entropy=float(dist.entropy().mean().detach()), teacher_grad_norm=float(grad),
            teacher_ratio_mean=float(ratios.mean()), teacher_updates=self.updates,
            teacher_q_mse=float(2*q_loss.detach()), teacher_prior_kl=float(prior_kl.detach()),
            teacher_direct_probability=float(dist.probs[:,0].mean().detach()),
            teacher_contrast_supported_fraction=float(supported.sum()/mask[:,1:].sum().clamp(min=1)),
            teacher_contrast_abs=float((contrast.abs()*supported).sum()/supported.sum().clamp(min=1)))
        for stage in range(ACTIVE_STAGES):
            subset = [t for t in trajectories if t[0]['stage'] == stage]
            metrics['teacher/u'+str(stage)+'/trajectories'] = len(subset)
            metrics['teacher/u'+str(stage)+'/decisions'] = sum(len(t) for t in subset)
            if subset:
                metrics['teacher/u'+str(stage)+'/phase_success_rate'] = np.mean([t[-1]['reward']>0 for t in subset])
        return metrics
