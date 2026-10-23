"""Frozen-controller evaluation with explicit plant and sensor interventions."""
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from .audit import sha256, write_json
from .contracts import Measurement, MeasurementPacket
from .control import (GSPIDController, PhysicsController, PIController,
                      MPCController, evaluate_transition)
from .learning import SAC, NeuralController
from .pressure import Ensemble
from .simulation import make_episode, targets, score


STRESSES = {
    'reference': {},
    'resistance_low': {'resistance_factor': .75},
    'resistance_high': {'resistance_factor': 1.25},
    'cooling_slow': {'cooling_tau_factor': 1.5},
    'cycle_300': {'cycle_s': 300.},
    'cycle_600': {'cycle_s': 600.},
    'combined': {'return_tau_factor': 1.5, 'water_UA_factor_after_midpoint': .85},
}
SENSORS = {
    'clean_60s': {},
    'noise_001': {'noise_MPa': .01},
    'noise_005': {'noise_MPa': .05},
    'dropout_300': {'dropout_steps': 30},
    'dropout_900': {'dropout_steps': 90},
}


class PlantIntervention:
    def __init__(self, plant, scenario, steps):
        self.plant = plant
        self.parameters = dict(STRESSES[scenario])
        self.midpoint = steps // 2
        self.changed = False
        physical = plant._plant
        physical.resistance *= self.parameters.get('resistance_factor', 1.)
        physical.cfg = replace(physical.cfg,
            cooling_tau_s=physical.cfg.cooling_tau_s*self.parameters.get('cooling_tau_factor', 1.),
            cooling_cycle_period_s=self.parameters.get('cycle_s', physical.cfg.cooling_cycle_period_s))
        physical.loop = replace(physical.loop,
            return_tau_s=physical.loop.return_tau_s*self.parameters.get('return_tau_factor', 1.))

    def __getattr__(self, key):
        return getattr(self.plant, key)

    def step(self, command, **kwargs):
        if not self.changed and self.time_s >= self.midpoint*self.cfg.control_step_s-1e-7:
            factor = self.parameters.get('water_UA_factor_after_midpoint', 1.)
            if factor != 1.:
                water = self.plant._plant.exchanger
                self.plant._plant.exchanger = replace(water, UA_kW_K=water.UA_kW_K*factor)
            self.changed = True
        return self.plant.step(command, **kwargs)


class SensorChannel:
    """Pressure samples arrive every 60 s; all other channels are unchanged."""
    def __init__(self, scenario, seed, starts=(600, 1800)):
        self.parameters = SENSORS[scenario]
        self.starts = starts
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 17549, 60]))
        self.last_step = -1

    def transform(self, packet, step):
        if step <= self.last_step:
            raise ValueError('Sensor channel must be called once per increasing decision step')
        self.last_step = step
        readings = []
        for reading in packet.readings:
            if reading.channel != 'wellhead_pressure_MPa':
                readings.append(reading)
                continue
            if step % 6:
                continue
            # Consume at every sample, including outages, to preserve paired noise.
            noise = self.rng.normal()*self.parameters.get('noise_MPa', 0.)
            duration = self.parameters.get('dropout_steps', 0)
            missing = any(start <= step < start+duration for start in self.starts)
            value = None if missing else reading.value+noise
            if value is not None and value <= 0:
                raise ValueError('Nonpositive noisy pressure; no silent clipping')
            readings.append(replace(reading, value=value, valid=not missing))
        return MeasurementPacket(packet.episode_id, tuple(readings))


class PressureFallback:
    """Hold or recursively forecast on the completed 60-s sample grid."""
    def __init__(self, kind='hold', offset=0., model=None, ar=None):
        if kind not in ('hold', 'ar7', 'ensemble') or offset != 0.:
            raise ValueError('Unknown frozen pressure intervention')
        self.kind, self.offset, self.model, self.ar = kind, offset, model, ar
        self.samples = deque(maxlen=7)
        self.last_std_MPa = 0.
        self.cold_start_holds = 0
        self._episode = None
        self._cache_key = None
        self._cache = None

    def ingest(self, packet, decision_at_s):
        if self._episode is not None and self._episode != packet.episode_id:
            raise ValueError('Cross-episode pressure history')
        self._episode = packet.episode_id
        for r in packet.readings:
            if r.available_at_s > decision_at_s:
                raise ValueError('Future measurement')
            if r.channel != 'wellhead_pressure_MPa' or not r.valid:
                continue
            if self.samples and r.sampled_at_s <= self.samples[-1][0]:
                raise ValueError('Repeated pressure sample')
            if self.samples and not np.isclose(r.sampled_at_s-self.samples[-1][0], 60.):
                self.samples.clear()
            self.samples.append((r.sampled_at_s, r.value-self.offset))
            self._cache_key = None

    def predict(self, history, current):
        if not self.samples:
            raise ValueError('No causal pressure observation for fallback')
        last_time, last_value = self.samples[-1]
        n = int((current.decision_at_s-last_time+1e-7)//60)
        if not 1 <= n <= 15:
            raise ValueError('Forecast outside 60-s / 15-step verified horizon')
        if self.kind == 'hold' or len(self.samples) < 7:
            self.last_std_MPa = 0.
            self.cold_start_holds += int(self.kind != 'hold')
            return last_value+self.offset
        key = (last_time, n)
        if key == self._cache_key:
            return self._cache
        h = np.array([[v for _, v in self.samples]])
        sigma = 0.
        for k in range(n):
            if self.kind == 'ensemble':
                mu, sd = self.model.predict(h, np.array([last_time+60*k]))
                sigma = float(sd[0])
            else:
                z = np.column_stack([np.ones(1), (h-np.asarray(self.ar['mean']))/np.asarray(self.ar['scale'])])
                mu = h[:, -1]+z@np.asarray(self.ar['beta'])
            if not np.isfinite(mu).all():
                raise ValueError('Nonfinite pressure forecast')
            h = np.column_stack([h[:, 1:], mu])
        result = float(h[0, -1]+self.offset)
        if result <= 0:
            raise ValueError('Nonpositive forecast; transfer failure retained')
        self.last_std_MPa = sigma
        self._cache_key, self._cache = key, result
        return result


def controller_for(job, interface, cfg):
    if job.get('checkpoint'):
        if sha256(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        agent = SAC.load_for_evaluation(job['checkpoint'])
        if agent.checkpoint_metadata['protocol_sha256'] != sha256(Path(job['plan']['out'])/'protocol.json'):
            raise ValueError('Policy/plan metadata mismatch')
        if agent.checkpoint_metadata['nominal_controller'] != asdict(cfg):
            raise ValueError('Checkpoint controller configuration differs from protocol')
        return NeuralController(agent, cfg, job['mode'], deterministic=True)
    if job['mode'] == 'physics':
        return PhysicsController()
    if job['mode'] == 'pi':
        return PIController(cfg)
    if job['mode'] == 'gspid':
        return GSPIDController(cfg)
    if job['mode'] == 'mpc':
        return MPCController(cfg, allocator=interface.allocator, tracking_weights=interface.tracking_weights)
    raise ValueError('Unknown controller')


def evaluate(job):
    torch.set_num_threads(1)
    out = Path(job['evaluation_out'])
    out.mkdir(parents=True, exist_ok=True)
    result_path, csv_path = out/(job['name']+'.json'), out/(job['name']+'.csv')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['job'] != job or sha256(csv_path) != result['trace_sha256']:
            raise ValueError('Completed evaluation does not match frozen job')
        return result
    plan, steps = job['plan'], job['steps']
    plant, interface, cfg, provenance = make_episode(plan, job['process_seed'], steps, job['name'], job.get('overrides'))
    plant = PlantIntervention(plant, job.get('stress', 'reference'), steps)
    sensor = None
    if job.get('sensor'):
        sensor = SensorChannel(job['sensor'], job['process_seed'], tuple(job.get('dropout_starts', [600, 1800])))
        model = None
        if job.get('predictor') == 'ensemble':
            p = job['pressure_assets']
            model = Ensemble(Path(p['ensemble_directory']), p['ensemble_seeds'])
        interface.observer.predictor = PressureFallback(job.get('predictor', 'hold'), job.get('pressure_offset', 0.),
            model=model, ar=job.get('pressure_assets', {}).get('ar7'))
    controller = controller_for(job, interface, cfg)
    reference = targets(job['profile'], job['condition'], steps, plan['evaluation_offset'], plan['target_protocol'])
    rows = []
    for k in range(steps):
        packet = plant.measurement_packet(job['name'])
        if sensor:
            packet = sensor.transform(packet, k)
        observe_tick = time.perf_counter()
        ctx = interface.receive(packet, plant.time_s, reference[k], reference[k+1])
        previous = interface.command.copy()
        command = interface.apply(controller.act(ctx))
        controller_s = time.perf_counter()-observe_tick
        tick = time.perf_counter()
        result = plant.step(command)
        from .contracts import reward_components
        reward = reward_components([plant.power, plant.cooling], ctx.target, command, previous,
            power_weight=interface.tracking_weights[0], cooling_weight=interface.tracking_weights[1])
        row = dict(step=k+1, target_power_kW=float(reference[k, 0]), target_cooling_kW=float(reference[k, 1]),
            output_power_kW=plant.power, output_cooling_kW=plant.cooling,
            rate_violation=int(np.any(abs(command-previous)>cfg.rate_per_step+1e-8)),
            bound_violation=int(np.any((command<0)|(command>cfg.flow_max+1e-8))),
            command_variation=float(abs(command-previous).sum()),
            control_step_wall_s=controller_s+time.perf_counter()-tick, controller_only_wall_s=controller_s,
            observed_pressure_MPa=ctx.pressure_MPa,
            pressure_estimated=int(interface.observer.history[-1].pressure_estimated),
            **result, **{key: float(value) for key, value in reward.items()})
        if job['mode'] == 'mpc':
            row.update(mpc_fallback=controller.last['fallback'], mpc_iterations=controller.last['iterations'])
        if job['mode'] == 'pi':
            row.update(pi_integral_power=controller.last['integral'][0], pi_integral_cooling=controller.last['integral'][1])
        if job['mode'] == 'gspid':
            row.update(gspid_integral_power=controller.last['integral'][0],
                       gspid_integral_cooling=controller.last['integral'][1])
        rows.append(row)
        if (k+1) % 128 == 0:
            write_json(out/(job['name']+'.progress.json'), dict(completed=k+1, planned=steps))
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    metrics = score(frame, plan['warmup'])
    f = frame.iloc[plan['warmup']:]
    errors = f[['output_power_kW', 'output_cooling_kW']].to_numpy()-f[['target_power_kW', 'target_cooling_kW']].to_numpy()
    np.testing.assert_allclose(f.power_cost, errors[:, 0]**2/100*interface.tracking_weights[0], atol=1e-12)
    np.testing.assert_allclose(f.cooling_cost, errors[:, 1]**2/100*interface.tracking_weights[1], atol=1e-12)
    commands = frame[['command_power_kg_s', 'command_cooling_kg_s']].to_numpy()
    delta = commands[plan['warmup']:]-commands[plan['warmup']-1:-1]
    np.testing.assert_allclose(f.command_change_cost, .002*(delta**2).sum(1), atol=1e-12)
    expected = (f.power_cost+f.cooling_cost+f.command_change_cost).to_numpy()
    np.testing.assert_allclose(f.total_cost, expected, atol=1e-12)
    np.testing.assert_allclose(f.reward, -expected, atol=1e-12)
    metrics.update(controller_latency_mean_s=float(f.controller_only_wall_s.mean()),
        controller_latency_p95_s=float(f.controller_only_wall_s.quantile(.95)),
        supply_shortfall_mean_kg_s=float(f.supply_shortfall_kg_s.mean()),
        supply_shortfall_fraction=float((f.supply_shortfall_kg_s>1e-8).mean()))
    result = dict(job=job, **metrics, trace_sha256=sha256(csv_path), provenance=provenance)
    write_json(result_path, result)
    return result
