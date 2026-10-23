"""Common causal observation, thermodynamic allocation and six-controller interfaces."""
from dataclasses import dataclass

import numpy as np
import osqp
from CoolProp.CoolProp import PropsSI
from scipy import sparse

from .contracts import CHANNELS, Observer, reward_components


@dataclass(frozen=True)
class ControlConfig:
    dt_s: float = 10.
    flow_max: float = .9
    rate_per_step: float = .06
    # None removes an extra branch residual cap; physical constraints still apply.
    residual_power: float | None = None
    residual_cooling: float | None = None
    nominal_eta: float = .85
    nominal_design_flow: float = 1.088
    nominal_offdesign: float = .7
    nominal_COP: float = .075
    power_tau_s: float = 35.
    cooling_tau_s: float = 55.
    pi_kp: float = .003
    pi_ki: float = .00003
    antiwindup_tau_s: float = 30.
    gspid_kp: float = .3
    gspid_ti_s: float = 120.
    gspid_td_s: float = 5.
    gspid_gain_floor: float = 10.
    mpc_horizon: int = 6
    mpc_rate_penalty: float = .002

    def __post_init__(self):
        optional = ('residual_power', 'residual_cooling')
        positive = (v for k, v in self.__dict__.items()
                    if not (k in optional and v is None) and k != 'gspid_td_s')
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Positive finite controller settings required")
        if not np.isfinite(self.gspid_td_s) or self.gspid_td_s < 0:
            raise ValueError("GS-PID derivative time must be nonnegative and finite")
        if not isinstance(self.mpc_horizon, int):
            raise ValueError("Integer horizon required")


@dataclass(frozen=True)
class Context:
    # Immutable value copies, never references to the evaluation plant.
    observation: tuple[float, ...]
    output: tuple[float, float]
    target: tuple[float, float]
    preview: tuple[float, float]
    previous_command: tuple[float, float]
    nominal_flow: tuple[float, float]
    pressure_MPa: float
    condenser_MPa: float


OBS_DIM = 191  # 7 * 26 + targets(4) + commands(2) + nominal(2) + forecast std(1)


class NominalAllocator:
    def __init__(self, config):
        self.cfg = config

    def coefficients(self, pressure, condenser):
        if not np.isfinite([pressure, condenser]).all() or min(pressure, condenser) <= 0:
            raise ValueError("Valid absolute pressure observations required")
        if pressure <= condenser:
            return 0., 0.
        hv, sv = PropsSI(["H", "S"], "P", pressure*1e6, "Q", 1, "Ammonia")
        hl = PropsSI("H", "P", condenser*1e6, "Q", 0, "Ammonia")
        h2 = PropsSI("H", "P", condenser*1e6, "S", sv, "Ammonia")
        return max(0., (hv-h2)/1000), max(0., (hv-hl)/1000)*self.cfg.nominal_COP

    def forward(self, flow, coefficients):
        c = self.cfg
        flow = np.asarray(flow)
        eta = c.nominal_eta*np.maximum(0., 1-c.nominal_offdesign*
                                      ((flow[..., 0]-c.nominal_design_flow)/c.nominal_design_flow)**2)
        return np.stack([flow[..., 0]*eta*coefficients[0], flow[..., 1]*coefficients[1]], axis=-1)

    def allocate(self, target, pressure, condenser):
        coefficients = self.coefficients(pressure, condenser)
        # Search a fixed nominal map, not hidden plant state or controller scores.
        grid = np.linspace(0., self.cfg.flow_max, 901)
        output = self.forward(np.column_stack([grid, grid]), coefficients)
        return np.array([grid[np.argmin(np.abs(output[:, i]-target[i]))] for i in range(2)])


def project(raw, previous, config):
    raw, previous = np.asarray(raw, float), np.asarray(previous, float)
    if raw.shape != (2,) or previous.shape != (2,) or not np.isfinite([raw, previous]).all():
        raise ValueError("Finite two-branch commands required")
    if np.any(previous < 0) or np.any(previous > config.flow_max):
        raise ValueError("Previous command outside admissible bounds")
    return np.clip(raw, np.maximum(0., previous-config.rate_per_step),
                   np.minimum(config.flow_max, previous+config.rate_per_step))


def residual_bounds(nominal, config):
    """Residual bounds before the common actuator projection, in kg/s."""
    nominal = np.asarray(nominal, float)
    if nominal.shape != (2,) or not np.isfinite(nominal).all():
        raise ValueError('Finite nominal two-branch flow required')
    lower, upper = [], []
    for value, cap in zip(nominal, (config.residual_power, config.residual_cooling)):
        if cap is None:
            if not 0 <= value <= config.flow_max:
                raise ValueError('Nominal flow outside physical command bounds')
            lower.append(-value)
            upper.append(config.flow_max-value)
        else:
            lower.append(-cap)
            upper.append(cap)
    return np.array(lower), np.array(upper)


def compose_residual_flow(action, nominal, config):
    """Zero action retains nominal flow; endpoints span the allowed correction."""
    action = np.asarray(action, float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError('Normalized residual action must be a finite pair in [-1, 1]')
    lower, upper = residual_bounds(nominal, config)
    return np.asarray(nominal) + action*np.where(action < 0, -lower, upper)


class ControlInterface:
    def __init__(self, episode_id, config=None, predictor=None, allocator=None,
                 tracking_weights=(1., 1.)):
        self.cfg = config or ControlConfig()
        weights = np.asarray(tracking_weights, float)
        if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Two positive finite tracking weights required')
        self.tracking_weights = tuple(weights)
        self.observer = Observer(episode_id, max_age_s=60., max_step_s=self.cfg.dt_s,
                                 predictor=predictor)
        self.allocator = allocator if allocator is not None else NominalAllocator(self.cfg)
        self.command = np.zeros(2)

    def receive(self, packet, decision_at_s, target, preview):
        predictor = self.observer.predictor
        if predictor is not None and hasattr(predictor, "ingest"):
            predictor.ingest(packet, decision_at_s)
        current = self.observer.update(packet, decision_at_s)
        pidx, pcidx = CHANNELS.index("wellhead_pressure_MPa"), CHANNELS.index("condenser_pressure_MPa")
        if not (current.valid[pidx] or current.pressure_estimated) or not current.valid[pcidx]:
            raise ValueError("Pressure unavailable; no hidden-state substitution permitted")
        pressure, condenser = current.values[pidx], current.values[pcidx]
        target, preview = np.asarray(target, float), np.asarray(preview, float)
        if target.shape != (2,) or preview.shape != (2,) or not np.isfinite([target, preview]).all():
            raise ValueError("Finite two-branch targets required")
        nominal = self.allocator.allocate(target, pressure, condenser)
        history = self.observer.policy_history().copy()
        history[:, :8] /= [100., 100., 5., 2000., 1., 1., 100., 5.]
        history[:, 16:24] /= 60.
        forecast_std = getattr(predictor, "last_std_MPa", 0.) if current.pressure_estimated else 0.
        obs = np.concatenate([history.ravel(), target/100., preview/100., self.command, nominal,
                              [forecast_std/5.]])
        assert obs.shape == (OBS_DIM,)
        return Context(tuple(obs), tuple(current.values[:2]), tuple(target), tuple(preview),
                       tuple(self.command), tuple(nominal), pressure, condenser)

    def apply(self, raw):
        command = project(raw, self.command, self.cfg)
        self.command = command.copy()
        return command


class PhysicsController:
    def act(self, ctx):
        return np.array(ctx.nominal_flow)


class PIController:
    def __init__(self, config):
        self.cfg = config
        self.integral_contribution = np.zeros(2)
        self.last = {}

    def act(self, ctx):
        c = self.cfg
        error = np.asarray(ctx.target)-ctx.output
        integral = self.integral_contribution+c.pi_ki*c.dt_s*error
        residual = c.pi_kp*error+integral
        lower, upper = residual_bounds(ctx.nominal_flow, c)
        bounded = np.clip(residual, lower, upper)
        raw = np.asarray(ctx.nominal_flow)+bounded
        applied = project(raw, ctx.previous_command, c)
        actual_residual = applied-ctx.nominal_flow
        self.integral_contribution = integral+c.dt_s/c.antiwindup_tau_s*(actual_residual-residual)
        self.last = dict(error=error.tolist(), integral=self.integral_contribution.tolist(),
                         raw_residual=residual.tolist(), bounded_residual=bounded.tolist())
        return raw


class GSPIDController:
    def __init__(self, config):
        self.cfg = config
        self.allocator = NominalAllocator(config)
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.last = {}

    def act(self, ctx):
        c = self.cfg
        coefficients = np.asarray(self.allocator.coefficients(ctx.pressure_MPa, ctx.condenser_MPa))
        gain = 1/np.maximum(coefficients, c.gspid_gain_floor)
        error = np.asarray(ctx.target)-ctx.output
        self.integral += gain*error*c.dt_s/c.gspid_ti_s
        proportional = c.gspid_kp*gain*error
        derivative = c.gspid_td_s*gain*(error-self.previous_error)/c.dt_s
        raw = np.asarray(ctx.nominal_flow)+proportional+self.integral+derivative
        applied = project(raw, ctx.previous_command, c)
        self.integral += c.dt_s/c.antiwindup_tau_s*(applied-raw)
        self.previous_error = error
        self.last = dict(error=error.tolist(), scheduled_gain=gain.tolist(),
                         proportional=proportional.tolist(), integral=self.integral.tolist(),
                         derivative=derivative.tolist())
        return raw


class MPCController:
    """2H distinct controls; one-step target preview followed by a hold.

    Uses only measured pressure and the nominal map. No true cycle phase, future
    heat trace, hidden mismatch parameters, or old superheat constraint.
    """
    def __init__(self, config, allocator=None, tracking_weights=(1., 1.)):
        self.cfg = config
        weights = np.asarray(tracking_weights, float)
        if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Two positive finite tracking weights required')
        self.tracking_weights = tuple(weights)
        self.allocator = allocator if allocator is not None else NominalAllocator(config)
        self.last_solution = None
        self.last = {}

    def act(self, ctx):
        c, H = self.cfg, self.cfg.mpc_horizon
        anchor = np.asarray(ctx.nominal_flow)
        coeff = self.allocator.coefficients(ctx.pressure_MPa, ctx.condenser_MPa)
        base = self.allocator.forward(anchor, coeff)
        jac = np.zeros((2, 2))
        for j in range(2):
            lo, hi = anchor.copy(), anchor.copy()
            lo[j], hi[j] = max(0., lo[j]-.002), min(c.flow_max, hi[j]+.002)
            jac[:, j] = (self.allocator.forward(hi, coeff)-self.allocator.forward(lo, coeff))/(hi[j]-lo[j])
        lag = np.exp(-c.dt_s/np.array([c.power_tau_s, c.cooling_tau_s]))
        A, B = np.diag(lag), np.diag(1-lag)@jac
        offset = (1-lag)*(base-jac@anchor)
        response = np.zeros((2*H, 2*H))
        state_matrix = np.zeros((2, 2*H))
        state_offset = np.asarray(ctx.output).copy()
        offsets = []
        for k in range(H):
            state_matrix = A@state_matrix
            state_matrix[:, 2*k:2*k+2] += B
            state_offset = A@state_offset+offset
            response[2*k:2*k+2] = state_matrix
            offsets.extend(state_offset)
        # Current interval target first, one-step preview for subsequent stages.
        targets = np.array([ctx.target]+[ctx.preview]*(H-1)).ravel()
        diff = np.eye(2*H)
        diff[2:, :-2] -= np.eye(2*(H-1))
        prior = np.zeros(2*H)
        prior[:2] = ctx.previous_command
        weights = np.tile(self.tracking_weights, H)
        P = 2*(response.T@(weights[:, None]*response)/100+c.mpc_rate_penalty*diff.T@diff)
        q = 2*(response.T@(weights*(np.array(offsets)-targets))/100-c.mpc_rate_penalty*diff.T@prior)
        constraints = sparse.vstack([sparse.eye(2*H), sparse.csc_matrix(diff)], format="csc")
        lower = np.r_[np.zeros(2*H), prior-c.rate_per_step]
        upper = np.r_[np.full(2*H, c.flow_max), prior+c.rate_per_step]
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(P)), q=q, A=constraints, l=lower, u=upper,
                     verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000, polishing=True)
        if self.last_solution is not None:
            solver.warm_start(x=self.last_solution)
        result = solver.solve(raise_error=False)
        success = result.info.status_val in (1, 2) and result.x is not None and np.isfinite(result.x).all()
        if success:
            success = (np.all(constraints@result.x >= lower-1e-5) and
                       np.all(constraints@result.x <= upper+1e-5))
        self.last = dict(status=result.info.status, fallback=not success, iterations=result.info.iter,
                         control_variables=2*H)
        if success:
            self.last_solution = np.r_[result.x[2:], result.x[-2:]]
            return project(result.x[:2], ctx.previous_command, c)
        return project(ctx.nominal_flow, ctx.previous_command, c)


def evaluate_transition(plant, interface, controller, context, target, **plant_kwargs):
    """Action precedes the state update; reward uses post-action outputs."""
    previous = interface.command.copy()
    command = interface.apply(controller.act(context))
    try:
        result = plant.step(command, **plant_kwargs)
    except Exception:
        interface.command = previous
        # A numerical failure ends this episode; controller memory must reset.
        raise
    reward = reward_components([plant.power, plant.cooling], target, command, context.previous_command,
                              power_weight=interface.tracking_weights[0],
                              cooling_weight=interface.tracking_weights[1])
    return result, {k: float(v) for k, v in reward.items()}
