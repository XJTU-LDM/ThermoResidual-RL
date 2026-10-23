"""Shared conditioned, Huber-critic SAC optimizer with delayed actor updates."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .learning import SAC
SEED = 20260930

class ConditionedSAC(SAC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditioning = None
        self.conditioning_hooks = []
        self.conditioning_scope = 'Shared Physics-only development-domain normalization'

    def install(self, mean, scale, *, already_transformed=False):
        if self.conditioning_hooks:
            raise ValueError('Do not condition twice')
        if self.cfg.actor_feedback_input or self.cfg.critic_feedback_input:
            raise ValueError('This transform requires the original 191-channel body input')
        mean, scale = (mean.to(self.device), scale.to(self.device))
        assert torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()
        self.conditioning = dict(mean=mean, scale=scale)
        specs = [(self.actor.body[0], mean, scale, self.actor_opt)]
        cm = torch.cat([mean, mean.new_zeros(2)])
        cs = torch.cat([scale, scale.new_ones(2)])
        specs += [(q.net[0], cm, cs, self.critic_opt) for q in self.critics]
        specs += [(q.net[0], cm, cs, None) for q in self.targets]
        for layer, m, s, optimizer in specs:
            if not already_transformed:
                with torch.no_grad():
                    layer.bias.add_(layer.weight @ m)
                    layer.weight.mul_(s)
                if optimizer is not None:
                    optimizer.state.pop(layer.weight, None)
                    optimizer.state.pop(layer.bias, None)
            self.conditioning_hooks.append(layer.register_forward_pre_hook(lambda module, args, m=m, s=s: ((args[0] - m) / s,)))

    @classmethod
    def load_learner_state(cls, path, *, device='cpu'):
        agent = super().load_learner_state(path, device=device)
        state = torch.load(path, map_location='cpu', weights_only=True)
        agent.conditioning_scope = state.get('conditioning_scope', agent.conditioning_scope)
        saved = state.get('input_conditioning')
        if saved:
            agent.install(saved['mean'], saved['scale'], already_transformed=True)
        return agent

    def save(self, path, metadata):
        SAC.save(self, path, metadata)
        if self.conditioning is None:
            return
        state = torch.load(path, map_location='cpu', weights_only=True)
        mean, scale = (self.conditioning[k].detach().cpu() for k in ('mean', 'scale'))
        entries = [('actor', 'body.0.', mean, scale)]
        cm, cs = (torch.cat([mean, mean.new_zeros(2)]), torch.cat([scale, scale.new_ones(2)]))
        entries += [(group, f'{i}.net.0.', cm, cs) for group in ('critics', 'targets') for i in range(2)]
        for group, prefix, m, s in entries:
            weights = state[group]
            raw = weights[prefix + 'weight'] / s
            weights[prefix + 'bias'] = weights[prefix + 'bias'] - raw @ m
            weights[prefix + 'weight'] = raw
        state['metadata']['ordinary_sac_input_preconditioning'] = dict(fitting_scope=self.conditioning_scope, export='Folded first-layer affine transform; no extra deployment measurements', auxiliary_model_loss=False)
        torch.save(state, path)

    def save_learner_state(self, path, metadata):
        super().save_learner_state(path, metadata)
        if self.conditioning is None:
            return
        state = torch.load(path, map_location='cpu', weights_only=True)
        for group in ('actor', 'critics', 'targets'):
            state[group] = {k: v.detach().cpu() for k, v in getattr(self, group).state_dict().items()}
        state['input_conditioning'] = {k: v.detach().cpu() for k, v in self.conditioning.items()}
        state['conditioning_scope'] = self.conditioning_scope
        if hasattr(self, 'initial_equivalence'):
            state['initial_equivalence'] = deepcopy(self.initial_equivalence)
        torch.save(state, path)

def regression_loss(prediction, target, spec):
    if spec['kind'] == 'mse':
        return (prediction - target).square().mean()
    if spec['kind'] == 'huber':
        delta = float(spec['delta'])
        if delta <= 0:
            raise ValueError('Positive Huber threshold required')
        return 2 * F.huber_loss(prediction, target, delta=delta)
    raise ValueError('Unknown regression loss')

class PrecisionSAC(ConditionedSAC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = dict(kind='mse')

    @classmethod
    def load_learner_state(cls, path, *, device='cpu'):
        agent = super().load_learner_state(path, device=device)
        agent.precision = deepcopy(agent.learner_resume_metadata.get('critic_precision', dict(kind='mse')))
        return agent

    def save(self, path, metadata):
        metadata = deepcopy(metadata)
        metadata['critic_precision'] = deepcopy(self.precision)
        super().save(path, metadata)

    def update(self, *, diagnostics=False):
        if self.frozen:
            raise RuntimeError('Evaluation checkpoint is frozen')
        c = self.cfg
        obs, action, reward, nxt, terminated = self.replay.sample(c.batch_size, self.device)
        alpha = self.log_alpha.exp().detach() if c.fixed_alpha is None else self.log_alpha.new_tensor(c.fixed_alpha)
        with torch.no_grad():
            next_action, lp = self.actor(nxt)
            tq = torch.minimum(*(q(nxt, next_action) for q in self.targets)) - alpha * lp
            target = reward + c.gamma * (1 - terminated) * tq
        critic_loss = sum((regression_loss(q(obs, action), target, self.precision) for q in self.critics))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        extra = {}
        if diagnostics:
            extra['critic_grad_norm'] = float(torch.sqrt(sum((p.grad.square().sum() for p in self.critics.parameters() if p.grad is not None))))
        if c.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), c.max_grad_norm)
        self.critic_opt.step()
        self.critics.requires_grad_(False)
        sampled, lp = self.actor(obs)
        actor_loss = (alpha * lp - torch.minimum(*(q(obs, sampled) for q in self.critics))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        if diagnostics:
            extra['actor_grad_norm'] = float(torch.sqrt(sum((p.grad.square().sum() for p in self.actor.parameters() if p.grad is not None))))
            extra['sample_tanh_slope_mean'] = float((1 - sampled.detach().square()).mean())
            extra['replay_reward_mean'] = float(reward.mean())
            extra['replay_reward_std'] = float(reward.std(unbiased=False))
        if c.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.actor_opt.step()
        self.critics.requires_grad_(True)
        if c.fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (lp.detach() + c.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        with torch.no_grad():
            for dst, src in zip(self.targets.parameters(), self.critics.parameters()):
                dst.lerp_(src, c.tau)
        self.updates += 1
        return dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()), alpha=float(alpha), updates=self.updates, **extra)

class DelayedSAC(PrecisionSAC):

    def update(self, *, diagnostics=False):
        delay = self.precision.get('policy_delay', 1)
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
            raise ValueError('Positive integer policy delay required')
        if delay > 1 and self.cfg.fixed_alpha is None:
            raise ValueError('Delay comparison requires a frozen entropy coefficient')
        perform = self.updates % delay == 0
        step = self.actor_opt.step
        try:
            if not perform:
                self.actor_opt.step = lambda *args, **kwargs: None
            result = super().update(diagnostics=diagnostics)
        finally:
            self.actor_opt.step = step
        result['actor_step_performed'] = int(perform)
        return result

class SharedRecipeSAC(DelayedSAC):

    @classmethod
    def load_learner_state(cls, path, *, device='cpu'):
        agent = super().load_learner_state(path, device=device)
        if agent.updates == 0 and agent.replay.count == 0:
            torch.manual_seed(SEED)
        return agent
