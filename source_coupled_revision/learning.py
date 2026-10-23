"""Identical SAC optimizer for residual and full-action control.

Only composition in NeuralController.act differs. No residual-only loss, extra
critic information, or gradient update is used during frozen evaluation.
"""
from dataclasses import asdict, dataclass
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .control import OBS_DIM, compose_residual_flow


def checkpoint_json_value(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported checkpoint metadata: {type(value).__name__}')


@dataclass(frozen=True)
class LearningConfig:
    hidden: int = 256
    learning_rate: float = .0003
    gamma: float = .99
    tau: float = .005
    batch_size: int = 256
    buffer_size: int = 100000
    initial_alpha: float = .2
    depth: int = 2
    activation: str = 'relu'
    layer_norm: bool = False
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    alpha_learning_rate: float | None = None
    feedback_skip: bool = False
    max_grad_norm: float | None = None
    action_coordinates: str = 'raw'
    target_entropy: float = -2.
    actor_feedback_input: bool = False
    critic_feedback_input: bool = False
    critic_quadratic_skip: bool = False
    fixed_alpha: float | None = None

    def __post_init__(self):
        if self.fixed_alpha is not None and (not np.isfinite(self.fixed_alpha) or self.fixed_alpha < 0):
            raise ValueError('Fixed entropy coefficient must be finite and nonnegative')
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError('Discount factor must be finite and in [0, 1)')
        if not np.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError('Target update coefficient must be finite and in (0, 1]')
        if (not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool)
                or not 1 <= self.batch_size <= self.buffer_size):
            raise ValueError('Integer batch size must fit the replay buffer')
        if self.depth < 1 or not isinstance(self.depth, int):
            raise ValueError('Positive integer network depth required')
        if self.activation not in ('relu', 'leaky_relu', 'silu'):
            raise ValueError('Unknown activation')
        if self.action_coordinates not in ('raw', 'feasible'):
            raise ValueError('Unknown action coordinates')
        if not np.isfinite(self.target_entropy):
            raise ValueError('Finite target entropy required')
        for value in (self.actor_learning_rate, self.critic_learning_rate,
                      self.alpha_learning_rate, self.max_grad_norm):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError('Positive finite optional optimizer settings required')


def hidden_layers(inputs, hidden, depth, activation, layer_norm):
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(inputs, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append({'relu': nn.ReLU, 'leaky_relu': lambda: nn.LeakyReLU(.01),
                       'silu': nn.SiLU}[activation]())
        inputs = hidden
    return layers


def feedback_features(obs):
    """Re-express existing causal channels; errors/trends are scaled by 10 kW."""
    current, previous = obs[..., 156:158], obs[..., 130:132]
    target, preview = obs[..., 182:184], obs[..., 184:186]
    previous_present = (obs[..., 155:156] > .5).to(obs.dtype)
    return torch.cat([10*(target-current), 10*(preview-target),
                      10*(current-previous)*previous_present,
                      10*(obs[..., 186:188]-obs[..., 188:190])], dim=-1)


def quadratic_action_features(obs, action):
    """Local quadratic action basis with causal feedback-action interactions."""
    feedback = feedback_features(obs)
    interactions = (feedback.unsqueeze(-1)*action.unsqueeze(-2)).flatten(-2)
    return torch.cat([action, action.square(), action[..., :1]*action[..., 1:], interactions], -1)


class Actor(nn.Module):
    def __init__(self, hidden, depth=2, activation='relu', layer_norm=False, feedback_skip=False,
                 feedback_input=False):
        super().__init__()
        self.feedback_input = feedback_input
        self.body = nn.Sequential(*hidden_layers(OBS_DIM+8*int(feedback_input), hidden, depth, activation, layer_norm))
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)
        self.feedback = nn.Linear(8, 2, bias=False) if feedback_skip else None
        if self.feedback is not None:
            nn.init.zeros_(self.feedback.weight)

    def forward(self, obs, deterministic=False):
        inputs = torch.cat([obs,feedback_features(obs)],dim=-1) if self.feedback_input else obs
        h = self.body(inputs)
        mu, logs = self.mean(h), self.log_std(h).clamp(-5, 1)
        if self.feedback is not None:
            mu = mu + self.feedback(feedback_features(obs))
        if deterministic:
            return mu.tanh(), None
        normal = Normal(mu, logs.exp())
        latent = normal.rsample()
        u = latent.tanh()
        # Work in latent space: 1 - tanh(x)^2 loses precision near saturation.
        logp = squashed_log_probability(normal, latent)
        return u, logp


def squashed_log_probability(normal, latent):
    log_jacobian = 2*(math.log(2.)-latent-F.softplus(-2*latent))
    return (normal.log_prob(latent)-log_jacobian).sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, hidden, depth=2, activation='relu', layer_norm=False, feedback_input=False,
                 quadratic_skip=False):
        super().__init__()
        self.feedback_input = feedback_input
        self.net = nn.Sequential(*hidden_layers(OBS_DIM+2+8*int(feedback_input), hidden, depth, activation, layer_norm),
                                 nn.Linear(hidden, 1))
        self.quadratic = nn.Linear(21, 1, bias=False) if quadratic_skip else None
        if self.quadratic is not None:
            nn.init.zeros_(self.quadratic.weight)

    def forward(self, obs, action):
        parts = [obs,action] + ([feedback_features(obs)] if self.feedback_input else [])
        value = self.net(torch.cat(parts, dim=-1))
        if self.quadratic is not None:
            value = value+self.quadratic(quadratic_action_features(obs, action))
        return value


class Replay:
    def __init__(self, size, seed):
        self.rng = np.random.default_rng(seed)
        self.size, self.count, self.pointer = size, 0, 0
        self.obs = np.zeros((size, OBS_DIM), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.action = np.zeros((size, 2), np.float32)
        self.reward = np.zeros((size, 1), np.float32)
        self.terminated = np.zeros((size, 1), np.float32)
        self.truncated = np.zeros((size, 1), np.float32)

    def add(self, obs, action, reward, next_obs, *, terminated, truncated):
        arrays = [np.asarray(v) for v in (obs, action, reward, next_obs)]
        if not all(np.isfinite(v).all() for v in arrays):
            raise ValueError("Nonfinite replay transition")
        if arrays[0].shape != (OBS_DIM,) or arrays[1].shape != (2,) or arrays[3].shape != (OBS_DIM,):
            raise ValueError("Replay dimension mismatch")
        i = self.pointer
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i] = arrays
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.pointer, self.count = (i+1)%self.size, min(self.size, self.count+1)

    def sample(self, n, device):
        if self.count < n:
            raise ValueError("Insufficient replay")
        idx = self.rng.integers(self.count, size=n)
        return [torch.as_tensor(v[idx], device=device) for v in
                (self.obs, self.action, self.reward, self.next_obs, self.terminated)]


class SAC:
    def __init__(self, config=None, *, seed=1, device="cpu"):
        self.cfg = config or LearningConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        c = self.cfg
        shape = (c.hidden, c.depth, c.activation, c.layer_norm)
        self.actor = Actor(*shape, feedback_skip=c.feedback_skip, feedback_input=c.actor_feedback_input).to(self.device)
        self.critics = nn.ModuleList([Critic(*shape,feedback_input=c.critic_feedback_input,
            quadratic_skip=c.critic_quadratic_skip) for _ in range(2)]).to(self.device)
        self.targets = nn.ModuleList([Critic(*shape,feedback_input=c.critic_feedback_input,
            quadratic_skip=c.critic_quadratic_skip) for _ in range(2)]).to(self.device)
        self.targets.load_state_dict(self.critics.state_dict())
        self.targets.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_learning_rate or c.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=c.critic_learning_rate or c.learning_rate)
        self.log_alpha = torch.tensor(np.log(c.initial_alpha), dtype=torch.float32, device=self.device,
                                      requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.alpha_learning_rate or c.learning_rate)
        self.replay = Replay(c.buffer_size, seed)
        self.updates = 0
        self.frozen = False
        self.checkpoint_metadata = None

    def update(self, *, diagnostics=False):
        if self.frozen:
            raise RuntimeError("Evaluation-only checkpoint is frozen")
        c = self.cfg
        obs, action, reward, nxt, terminated = self.replay.sample(c.batch_size, self.device)
        alpha = (self.log_alpha.exp().detach() if c.fixed_alpha is None
                 else self.log_alpha.new_tensor(c.fixed_alpha))
        with torch.no_grad():
            next_action, lp = self.actor(nxt)
            tq = torch.minimum(*(q(nxt, next_action) for q in self.targets))-alpha*lp
            target = reward+c.gamma*(1-terminated)*tq
        critic_loss = sum((q(obs, action)-target).square().mean() for q in self.critics)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        extra = {}
        if diagnostics:
            extra['critic_grad_norm'] = float(torch.sqrt(sum(
                p.grad.square().sum() for p in self.critics.parameters() if p.grad is not None)))
        if c.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), c.max_grad_norm)
        self.critic_opt.step()
        self.critics.requires_grad_(False)
        sampled, lp = self.actor(obs)
        actor_loss = (alpha*lp-torch.minimum(*(q(obs, sampled) for q in self.critics))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        if diagnostics:
            extra['actor_grad_norm'] = float(torch.sqrt(sum(
                p.grad.square().sum() for p in self.actor.parameters() if p.grad is not None)))
            with torch.no_grad():
                hidden = torch.cat([obs,feedback_features(obs)],dim=-1) if c.actor_feedback_input else obs
                for layer in self.actor.body:
                    hidden = layer(hidden)
                    if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.SiLU)):
                        break
                extra['actor_first_layer_inactive_fraction'] = float((hidden == 0).all(0).float().mean())
                extra['sample_tanh_slope_mean'] = float((1-sampled.square()).mean())
                extra['replay_reward_mean'] = float(reward.mean())
                extra['replay_reward_std'] = float(reward.std(unbiased=False))
        if c.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.actor_opt.step()
        self.critics.requires_grad_(True)
        if c.fixed_alpha is None:
            alpha_loss = -(self.log_alpha*(lp.detach()+c.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        with torch.no_grad():
            for dst, src in zip(self.targets.parameters(), self.critics.parameters()):
                dst.lerp_(src, c.tau)
        self.updates += 1
        return dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                    alpha=float(self.log_alpha.exp().detach()) if c.fixed_alpha is None else c.fixed_alpha,
                    updates=self.updates, **extra)

    def save(self, path, metadata):
        # NumPy scalar configuration values must remain weights-only loadable.
        metadata = json.loads(json.dumps(metadata,default=checkpoint_json_value))
        torch.save(dict(actor=self.actor.state_dict(), critics=self.critics.state_dict(),
                        targets=self.targets.state_dict(), log_alpha=self.log_alpha.detach().cpu(),
                        config=asdict(self.cfg), observation_dim=OBS_DIM, updates=self.updates,
                        metadata=metadata), path)

    def warm_start(self, path):
        """Load compatible networks only; explicitly reset optimizers and replay."""
        source = SAC.load_for_evaluation(path, device=str(self.device))
        old, new = asdict(source.cfg), asdict(self.cfg)
        for key in ('hidden', 'depth', 'activation', 'layer_norm', 'action_coordinates',
                    'actor_feedback_input','critic_feedback_input','critic_quadratic_skip'):
            if old[key] != new[key]:
                raise ValueError(f'Warm-start architecture mismatch: {key}')
        incompatible = self.actor.load_state_dict(source.actor.state_dict(), strict=False)
        allowed = ['feedback.weight'] if new['feedback_skip'] and not old['feedback_skip'] else []
        if incompatible.missing_keys != allowed or incompatible.unexpected_keys:
            raise ValueError('Unsupported actor warm-start conversion')
        self.critics.load_state_dict(source.critics.state_dict())
        self.targets.load_state_dict(source.targets.state_dict())
        with torch.no_grad():
            self.log_alpha.copy_(source.log_alpha)
        return dict(source_updates=source.updates, source_metadata=source.checkpoint_metadata,
                    optimizer_replay_reset=True, feedback_zero_initialized=bool(allowed))

    def save_learner_state(self, path, metadata):
        """Save the learner, not the external physical environments."""
        temporary = str(path)+'.tmp'
        self.save(temporary, metadata)
        state = torch.load(temporary, map_location='cpu', weights_only=True)
        replay = self.replay
        state['learner'] = dict(actor_optimizer=self.actor_opt.state_dict(),
            critic_optimizer=self.critic_opt.state_dict(), alpha_optimizer=self.alpha_opt.state_dict(),
            replay={k: torch.from_numpy(getattr(replay, k)[:replay.count].copy()) for k in
                    ('obs', 'next_obs', 'action', 'reward', 'terminated', 'truncated')},
            replay_count=replay.count, replay_pointer=replay.pointer, replay_rng=replay.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
            environment_state_saved=False)
        torch.save(state, temporary)
        import os
        os.replace(temporary, path)

    @classmethod
    def load_learner_state(cls, path, *, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        agent = cls(LearningConfig(**state['config']), device=device)
        for name in ('actor', 'critics', 'targets'):
            getattr(agent, name).load_state_dict(state[name])
        with torch.no_grad():
            agent.log_alpha.copy_(state['log_alpha'].to(agent.device))
        saved = state['learner']
        for name in ('actor', 'critic', 'alpha'):
            getattr(agent, name+'_opt').load_state_dict(saved[name+'_optimizer'])
        for name, tensor in saved['replay'].items():
            getattr(agent.replay, name)[:saved['replay_count']] = tensor.numpy()
        agent.replay.count, agent.replay.pointer = saved['replay_count'], saved['replay_pointer']
        agent.replay.rng.bit_generator.state = saved['replay_rng']
        agent.updates = state['updates']
        agent.learner_resume_metadata = state['metadata']
        torch.set_rng_state(saved['torch_rng'])
        if agent.device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'], agent.device)
        return agent

    @classmethod
    def load_for_evaluation(cls, path, *, device="cpu"):
        # Narrow compatibility for locally produced pre-normalization metadata.
        with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
            state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("observation_dim") != OBS_DIM:
            raise ValueError("Checkpoint observation interface mismatch")
        agent = cls(LearningConfig(**state["config"]), device=device)
        agent.actor.load_state_dict(state["actor"])
        agent.critics.load_state_dict(state["critics"])
        agent.targets.load_state_dict(state["targets"])
        with torch.no_grad():
            agent.log_alpha.copy_(state["log_alpha"].to(agent.device))
        for network in (agent.actor, agent.critics, agent.targets):
            network.eval().requires_grad_(False)
        agent.frozen = True
        agent.updates = state["updates"]
        agent.checkpoint_metadata = state["metadata"]
        return agent


class NeuralController:
    def __init__(self, agent, control_config, mode, *, deterministic=True):
        if mode not in ("rrl", "sac"):
            raise ValueError("Unknown action composition")
        if agent.checkpoint_metadata is not None:
            if (agent.checkpoint_metadata["controller"] != mode or
                    agent.checkpoint_metadata["nominal_controller"] != asdict(control_config)):
                raise ValueError("Checkpoint action composition/controller configuration mismatch")
            if not deterministic:
                raise ValueError("Evaluation checkpoint requires deterministic control")
        self.agent, self.cfg, self.mode = agent, control_config, mode
        self.deterministic = deterministic
        self.last_action = None

    def act(self, ctx):
        with torch.no_grad():
            obs = torch.tensor(ctx.observation, device=self.agent.device, dtype=torch.float32)[None]
            u, _ = self.agent.actor(obs, self.deterministic)
            self.last_action = u[0].cpu().numpy()
        from .feasible_actions import compose_actor_flow
        return compose_actor_flow(self.last_action, ctx, self.cfg, self.mode, self.agent.cfg.action_coordinates)
