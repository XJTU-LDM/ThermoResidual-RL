"""Conservative stochastic greybox with a dissipative, zero-holdup outlet.

This is an effective hydraulic extension, not an axial Huang-model reproduction.
Bulk storage remains two-phase; outlet loss is isenthalpic. Any liquid formed
is separated and returned through the rejection/receiver balance. No measured
pressure or heat output is overwritten. Noise amplitudes require calibration.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from .coupled import ProcessPerturbation, hydraulic_flows, liquid
from .foundation import WaterCooledLoop
from .foundation_runtime import build_research_loop


@dataclass(frozen=True)
class ProcessConfig:
    resistance_MPa_s2_kg2: float = .15
    resistance_log_std: float = .2
    sink_log_std: float = .05
    wall_log_std: float = .05
    fast_tau_s: float = 60.
    sink_tau_s: float = 300.
    wall_tau_s: float = 600.
    innovation_correlation: float = .5
    cadence_s: float = 10.
    oscillation_period_s: float = 360.
    oscillation_decay_s: float = 1080.
    oscillation_weight: float = 0.
    sink_phase_lead_rad: float = 0.
    water_UA_noise: bool = False
    sink_capacity_override_kJ_K: float = 0.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Finite stochastic configuration required')
        if not 0 <= self.resistance_MPa_s2_kg2 <= 1.:
            raise ValueError('Resistance outside declared effective-model range')
        if any(not 0 <= v <= .8 for v in (self.resistance_log_std,self.sink_log_std,self.wall_log_std)):
            raise ValueError('Log standard deviation outside declared envelope')
        if min(self.fast_tau_s,self.sink_tau_s,self.wall_tau_s,self.cadence_s) <= 0:
            raise ValueError('Positive timescales required')
        if not -.95 <= self.innovation_correlation <= .95:
            raise ValueError('Innovation covariance must be nonsingular')
        if not 0 <= self.oscillation_weight <= 1 or min(self.oscillation_period_s,self.oscillation_decay_s)<=2*self.cadence_s:
            raise ValueError('Resolved, stationary oscillatory process required')
        if self.sink_capacity_override_kJ_K<0:
            raise ValueError('Nonnegative optional effective condenser capacity required')


def process_stream(config, count, seed):
    """Stationary correlated OU samples, fixed on a physical-time grid.

    Bounded log multipliers have a symmetric mean-zero log, not mean-one
    multipliers. This small lognormal mean shift is retained, not hidden.
    """
    if count < 1 or int(count) != count:
        raise ValueError('Positive integer stream length required')
    tau = np.array([config.fast_tau_s,config.sink_tau_s,config.wall_tau_s])
    phi = np.exp(-config.cadence_s/tau)
    corr = np.eye(3)
    corr[0,1] = corr[1,0] = config.innovation_correlation
    cov = np.sqrt(1-phi**2)[:,None]*corr*np.sqrt(1-phi**2)[None,:]
    stationary = cov/(1-phi[:,None]*phi[None,:])
    rng = np.random.default_rng(seed)
    x = np.linalg.cholesky(stationary)@rng.normal(size=3)
    factor = np.linalg.cholesky(cov)
    output = np.empty((count,3))
    for k in range(count):
        output[k] = x
        x = phi*x+factor@rng.normal(size=3)
    if config.oscillation_weight:
        # A random-phase damped oscillator reproduces a spectral band, not the
        # historical phase. Independent RNG preserves paired-stream prefixes.
        rng_osc = np.random.default_rng(np.random.SeedSequence([seed,417]))
        state = rng_osc.normal(size=2)
        angle = 2*np.pi*config.cadence_s/config.oscillation_period_s
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        damping = np.exp(-config.cadence_s/config.oscillation_decay_s)
        w = config.oscillation_weight
        phase = config.sink_phase_lead_rad
        for k in range(count):
            oscillatory = np.array([state[0],state[0]*np.cos(phase)-state[1]*np.sin(phase)])
            output[k,:2] = np.sqrt(1-w)*output[k,:2]+np.sqrt(w)*oscillatory
            state = damping*rotation@state+np.sqrt(1-damping**2)*rng_osc.normal(size=2)
    # Smooth three-sigma bounding; never truncate physical output pressure.
    logs = 3*np.tanh(output/3)*np.array([config.resistance_log_std,config.sink_log_std,config.wall_log_std])
    result = np.exp(logs)
    result.setflags(write=False)
    return result


class ResistiveLoop(WaterCooledLoop):
    """Quasi-steady port loss coupled to conservative bulk storage and loads."""
    def __init__(self, *args, resistance_MPa_s2_kg2=.15, **kwargs):
        if not np.isfinite(resistance_MPa_s2_kg2) or resistance_MPa_s2_kg2 < 0:
            raise ValueError('Nonnegative resistance required')
        self.resistance = float(resistance_MPa_s2_kg2)
        self.resistance_multiplier = 1.
        super().__init__(*args, **kwargs)

    def hydraulics(self, command, bulk, pc, hydraulic_loop):
        resistance = self.resistance*self.resistance_multiplier
        if resistance == 0 or float(np.sum(command)) == 0:
            flow = hydraulic_flows(command,bulk.pressure_MPa,pc,self.cfg,hydraulic_loop)
            return flow,bulk.pressure_MPa
        head = max(bulk.pressure_MPa-pc,0.)
        maximum = min(float(np.sum(command)), math.sqrt(head/resistance))
        if maximum <= 1e-14:
            return np.zeros(2),bulk.pressure_MPa
        def residual(total):
            pressure = bulk.pressure_MPa-resistance*total**2
            return total-float(hydraulic_flows(command,pressure,pc,self.cfg,hydraulic_loop).sum())
        total = brentq(residual,0.,maximum,xtol=1e-12)
        port = bulk.pressure_MPa-resistance*total**2
        return hydraulic_flows(command,port,pc,self.cfg,hydraulic_loop),port

    def _delivery(self, command, bulk, condenser_MPa, hydraulic_loop):
        raw, port = self.hydraulics(command,bulk,condenser_MPa,hydraulic_loop)
        h = bulk.vapor_enthalpy_kJ_kg
        if abs(port-bulk.pressure_MPa) < 1e-12:
            fraction,hv,sv = 1.,h,bulk.vapor_entropy_J_kg_K
            entropy = 0.
        else:
            hl,hv = (float(PropsSI('H','P',port*1e6,'Q',q,'Ammonia'))/1000 for q in (0,1))
            if h < hl:
                raise ValueError('Outlet outside vapor-delivery regime')
            fraction = min(1.,(h-hl)/(hv-hl))
            if fraction == 1.:
                # If superheated, do not silently reset enthalpy to saturation.
                hv = h
            sv = float(PropsSI('S','P',port*1e6,'H',hv*1000,'Ammonia'))
            entropy = float(PropsSI('S','P',port*1e6,'H',h*1000,'Ammonia'))-bulk.vapor_entropy_J_kg_K
            if entropy < -1e-5:
                raise ValueError('Negative throttle entropy production')
        vapor = raw*fraction
        return raw,vapor,hv,sv,dict(withdrawal_kg_s=float(raw.sum()),
            separated_liquid_kg_s=float((raw-vapor).sum()),
            port_pressure_drop_MPa=bulk.pressure_MPa-port,
            throttle_entropy_production_W_K=float(raw.sum())*entropy)

    def _reported_source(self, bulk):
        pc = liquid(self.sink_T+self.cfg.condenser_approach_K)[0]/1000
        _,port = self.hydraulics(self.command,bulk,pc,self.loop)
        temperature = float(PropsSI('T','P',port*1e6,'H',bulk.vapor_enthalpy_kJ_kg*1000,'Ammonia'))
        return dict(pressure_MPa=port,source_temperature_K=temperature,
                    bulk_pressure_MPa=bulk.pressure_MPa,bulk_temperature_K=bulk.temperature_K)


class StochasticLoop:
    """Private common-random-number stream; controllers receive own outcomes."""
    def __init__(self, plant, config, duration_s, seed):
        if not isinstance(plant,ResistiveLoop):
            raise TypeError('ResistiveLoop required')
        if duration_s <= 0 or plant.time_s != 0:
            raise ValueError('Initialize a positive-duration stream at time zero')
        self._plant,self.process_config = plant,config
        self._stream = process_stream(config,int(np.ceil(duration_s/config.cadence_s))+1,seed)
        self.seed = int(seed)

    def __getattr__(self,name):
        return getattr(self._plant,name)

    def step(self,request,dt_s=None,**kwargs):
        if 'disturbance' in kwargs:
            raise ValueError('Frozen stochastic adapter owns process disturbances')
        dt = self.cfg.control_step_s if dt_s is None else float(dt_s)
        if not np.isfinite(dt) or not 0 < dt <= self.cfg.control_step_s:
            raise ValueError('Invalid control interval')
        index = int(np.floor((self.time_s+1e-8)/self.process_config.cadence_s))
        if index >= len(self._stream):
            raise ValueError('Stochastic stream exhausted')
        # One command interval may not silently cross a process-grid boundary.
        if self.time_s+dt > (index+1)*self.process_config.cadence_s+1e-7:
            raise ValueError('Split interval at the stochastic process-grid boundary')
        resistance,sink,wall = self._stream[index]
        ua_multiplier = float(sink)
        if self.process_config.water_UA_noise:
            water = self.exchanger
            sink = (-np.expm1(-water.UA_kW_K*ua_multiplier/water.capacity_rate_kW_K)
                    /water.effectiveness)
        previous = self._plant.resistance_multiplier
        self._plant.resistance_multiplier = float(resistance)
        try:
            result = self._plant.step(request,dt_s=dt,disturbance=ProcessPerturbation(
                wall_multiplier=float(wall),sink_conductance_multiplier=float(sink)),**kwargs)
        except Exception:
            self._plant.resistance_multiplier = previous
            raise
        result.update(process_resistance_multiplier=float(resistance),
                      process_sink_multiplier=float(sink),process_wall_multiplier=float(wall),
                      process_water_UA_multiplier=ua_multiplier)
        return result

    def measurement_packet(self,episode_id):
        return self._plant.measurement_packet(episode_id)


def build_stochastic_loop(model_file, config, duration_s, seed, *, port_pressure_MPa=3.,
                          initial_total_flow_kg_s=.6, **kwargs):
    # Initial mean pressure drop, not assimilation of future pressure samples.
    bulk = port_pressure_MPa+config.resistance_MPa_s2_kg2*initial_total_flow_kg_s**2
    plant,cc,provenance = build_research_loop(model_file,source_pressure_MPa=bulk,
        initial_total_flow_kg_s=initial_total_flow_kg_s,loop_type=ResistiveLoop,
        loop_options={'resistance_MPa_s2_kg2':config.resistance_MPa_s2_kg2},**kwargs)
    if config.sink_capacity_override_kJ_K:
        plant.cfg = replace(plant.cfg,sink_capacity_kJ_K=config.sink_capacity_override_kJ_K)
    provenance.update(process_configuration=asdict(config),process_seed=int(seed),
        pressure_definition='wellhead port after isenthalpic hydraulic loss and equilibrium separation',
        parameter_status='effective statistical model, not uniquely identified pipe resistance')
    return StochasticLoop(plant,config,duration_s,seed),cc,provenance


def load_stochastic_model(directory, duration_s, seed, *, purpose='research', **kwargs):
    """Load frozen parameters with an explicit research/formal-use gate."""
    from .audit import sha256
    directory = Path(directory)
    if purpose not in ('research','formal_training'):
        raise ValueError('Unknown model-use purpose')
    frozen = json.loads((directory/'frozen_process.json').read_text())
    model = directory.parent/'model_foundation_v2/frozen_model.json'
    if sha256(model)!=frozen['base_model_sha256']:
        raise ValueError('Base physical configuration changed after selection')
    release_file = directory/'release_manifest.json'
    release = json.loads(release_file.read_text()) if release_file.exists() else {}
    if purpose=='formal_training' and not release.get('formal_RL_training_authorized',False):
        raise ValueError('Formal training gate has not passed; use research-only scenarios')
    integration = release.get('accepted_integration') or dict(substep=.625,radial_cells=192,axial_slices=16)
    for key,value in integration.items():
        kwargs.setdefault(key,value)
    plant,cc,provenance = build_stochastic_loop(model,ProcessConfig(**frozen['parameters']),duration_s,seed,**kwargs)
    provenance.update(process_manifest_sha256=sha256(directory/'frozen_process.json'),purpose=purpose,
        field_statistical_consistency_passed=release.get('field_statistical_consistency_passed',False),
        integration_used={key:kwargs[key] for key in integration})
    return plant,cc,provenance
