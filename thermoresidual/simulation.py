"""Shared simulation, target, scoring and training operations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .audit import sha256, write_json
from .control import (ControlInterface, GSPIDController, PIController, MPCController,
                      PhysicsController, evaluate_transition, OBS_DIM)
from .learning import SAC, LearningConfig, NeuralController
from .stochastic import load_stochastic_model


PROFILES = ('standard','medium_unseen','very_fast_unseen','mixed_step_ramp')



def targets(profile, condition, steps, offset=0, target_protocol=None):
    # Generate the fixed waveform basis before scaling to the protocol ranges.
    from .targets import SCENARIOS, target_for_profile
    spec = list(SCENARIOS.values())[condition]
    step = torch.arange(steps+1,dtype=torch.float32)+offset
    result = np.column_stack([target_for_profile(step,spec,condition,c,profile).numpy()
                              for c in (False,True)])
    if not np.isfinite(result).all() or not np.all((result[:,0]>=15)&(result[:,0]<=36)):
        raise ValueError('Invalid target envelope')
    if target_protocol is not None:
        # Known amplitude families are fixed, not estimated from a short window.
        basis = np.array([[spec['power_low'],spec['power_high']],
                        [spec['cooling_low'],spec['cooling_high']]])
        if profile=='mixed_step_ramp':
            basis[1]=[32.,48.] if condition in (0,2) else [28.,52.]
        new = np.asarray(target_protocol['low_amplitude' if condition in (0,2)
                                          else 'high_amplitude'],float)
        if new.shape != (2,2) or not np.isfinite(new).all() or np.any(new[:,1]<=new[:,0]):
            raise ValueError('Invalid coupled target envelope')
        result = new[:,0]+(result-basis[:,0])/(basis[:,1]-basis[:,0])*(new[:,1]-new[:,0])
        if np.any(result<new[:,0]-1e-5) or np.any(result>new[:,1]+1e-5):
            raise ValueError('Target waveform exceeded its declared amplitude family')
    return result

def make_episode(plan, process_seed, steps, episode_id, overrides=None):
    if plan.get('require_full_action_parity', False):
        tuning_keys = {'pi_kp', 'pi_ki', 'antiwindup_tau_s', 'mpc_horizon', 'mpc_rate_penalty',
                       'gspid_kp', 'gspid_ti_s', 'gspid_td_s', 'gspid_gain_floor'}
        if set(overrides or {}) - tuning_keys:
            raise ValueError('Parity protocol permits only controller tuning overrides')
    plant,cfg,provenance = load_stochastic_model(plan['model_root'],(steps+2)*10.,process_seed,
                                               purpose='research')
    shared = plan.get('controller_overrides', {})
    for key in ('residual_power', 'residual_cooling', 'flow_max', 'rate_per_step'):
        if key in shared and key in (overrides or {}) and overrides[key] != shared[key]:
            raise ValueError('Controller override conflicts with shared action permissions')
    cfg = replace(cfg, **{**shared, **(overrides or {})})
    if plan.get('require_full_action_parity', False):
        if cfg.residual_power is not None or cfg.residual_cooling is not None:
            raise ValueError('Full action parity requires both residual caps to be None')
    allocator = None
    if plan.get('joint_allocation_points'):
        from .joint_allocation import JointAllocator
        allocator = JointAllocator(plan['joint_allocation_points'])
    interface = ControlInterface(episode_id,cfg,allocator=allocator,
                                 tracking_weights=plan.get('tracking_weights', (1., 1.)))
    interface.command = plant.command.copy()
    return plant,interface,cfg,provenance

def rollout(plan, mode, process_seed, profile, condition, steps, *, agent=None,
            overrides=None, offset=0, progress=None):
    episode_id = f'{mode}_{profile}_{condition}_{process_seed}'
    plant,interface,cfg,provenance = make_episode(plan,process_seed,steps,episode_id,overrides)
    if agent is not None:
        controller = NeuralController(agent,cfg,mode,deterministic=True)
    else:
        if mode=='physics':
            controller = PhysicsController()
        elif mode=='pi':
            controller = PIController(cfg)
        elif mode=='mpc':
            controller = MPCController(cfg,allocator=interface.allocator,
                                       tracking_weights=interface.tracking_weights)
        elif mode=='gspid':
            controller = GSPIDController(cfg)
        else:
            raise ValueError('Unknown controller mode')
    reference = targets(profile,condition,steps,offset,plan.get('target_protocol'))
    ctx = interface.receive(plant.measurement_packet(episode_id),0.,reference[0],reference[1])
    rows = []
    for k in range(steps):
        tick = time.perf_counter()
        previous = interface.command.copy()
        result,reward = evaluate_transition(plant,interface,controller,ctx,ctx.target)
        latency = time.perf_counter()-tick
        command = interface.command.copy()
        row = dict(step=k+1,target_power_kW=float(reference[k,0]),target_cooling_kW=float(reference[k,1]),
            output_power_kW=plant.power,output_cooling_kW=plant.cooling,
            rate_violation=int(np.any(abs(command-previous)>cfg.rate_per_step+1e-8)),
            bound_violation=int(np.any((command<0)|(command>cfg.flow_max+1e-8))),
            command_variation=float(np.abs(command-previous).sum()),control_step_wall_s=latency,
            **result,**reward)
        if mode=='mpc':
            row.update(mpc_fallback=controller.last['fallback'],mpc_iterations=controller.last['iterations'])
        if mode=='pi':
            row.update(pi_integral_power=controller.last['integral'][0],
                       pi_integral_cooling=controller.last['integral'][1])
        if mode=='gspid':
            row.update(gspid_integral_power=controller.last['integral'][0],
                       gspid_integral_cooling=controller.last['integral'][1])
        rows.append(row)
        if progress is not None and (k+1)%128==0:
            write_json(progress,dict(completed=k+1,planned=steps,stage='evaluation'))
        if k+1<steps:
            ctx = interface.receive(plant.measurement_packet(episode_id),plant.time_s,
                                    reference[k+1],reference[k+2])
    return pd.DataFrame(rows),cfg,provenance

def score(frame,warmup):
    if len(frame)<=warmup or not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
        raise ValueError('Incomplete/nonfinite evaluation')
    f = frame.iloc[warmup:]
    e = f[['output_power_kW','output_cooling_kW']].to_numpy()-f[['target_power_kW','target_cooling_kW']].to_numpy()
    energy = float(frame.energy_residual_kJ.abs().sum()/frame.energy_throughput_kJ.sum())
    result = dict(samples=len(f),combined_RMSE=float(np.sqrt(np.mean(e**2))),
        power_RMSE=float(np.sqrt(np.mean(e[:,0]**2))),cooling_RMSE=float(np.sqrt(np.mean(e[:,1]**2))),
        mean_cost=float(f.total_cost.mean()),mean_reward=float(f.reward.mean()),
        command_change_cost=float(f.command_change_cost.mean()),max_error=float(np.abs(e).max()),
        power_positive_error_max=float(np.maximum(e[:,0],0).max()),
        cooling_positive_error_max=float(np.maximum(e[:,1],0).max()),
        command_total_variation=float(f.command_variation.sum()),
        energy_ratio=energy,mass_residual_kg=float(frame.mass_residual_kg.abs().max()),
        rate_violations=int(frame.rate_violation.sum()),bound_violations=int(frame.bound_violation.sum()),
        mean_step_wall_s=float(f.control_step_wall_s.mean()),
        mpc_fallbacks=int(frame.mpc_fallback.sum()) if 'mpc_fallback' in frame else 0)
    result['admissible'] = (result['rate_violations']==result['bound_violations']==0 and
                            result['mass_residual_kg']<=1e-6 and energy<=.01)
    return result

def initialize_worker():
    torch.set_num_threads(1)
