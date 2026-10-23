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
    # Reuse target generation only, never the legacy environment or boundary reader.
    from run_four_conditions import SCENARIOS, target_for_profile
    spec = list(SCENARIOS.values())[condition]
    step = torch.arange(steps+1,dtype=torch.float32)+offset
    result = np.column_stack([target_for_profile(step,spec,condition,c,profile).numpy()
                              for c in (False,True)])
    if not np.isfinite(result).all() or not np.all((result[:,0]>=15)&(result[:,0]<=36)):
        raise ValueError('Invalid target envelope')
    if target_protocol is not None:
        # Known amplitude families are fixed, not estimated from a short window.
        old = np.array([[spec['power_low'],spec['power_high']],
                        [spec['cooling_low'],spec['cooling_high']]])
        if profile=='mixed_step_ramp':
            old[1]=[32.,48.] if condition in (0,2) else [28.,52.]
        new = np.asarray(target_protocol['low_amplitude' if condition in (0,2)
                                          else 'high_amplitude'],float)
        if new.shape != (2,2) or not np.isfinite(new).all() or np.any(new[:,1]<=new[:,0]):
            raise ValueError('Invalid coupled target envelope')
        result = new[:,0]+(result-old[:,0])/(old[:,1]-old[:,0])*(new[:,1]-new[:,0])
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


def baseline_job(job):
    plan = job['plan']
    rows = []
    out = Path(plan['out'])/'selection'/job['name']
    out.mkdir()
    for condition in range(4):
        frame,cfg,_ = rollout(plan,job['mode'],plan['selection_process_seed'],'standard',condition,
            plan['selection_steps'],overrides=job['overrides'],offset=plan['selection_offset'])
        frame.to_csv(out/f'condition{condition}.csv',index=False)
        rows.append(dict(condition=condition,**score(frame,plan['warmup'])))
        write_json(out/'progress.json',dict(completed=condition+1,planned=4))
    result = dict(name=job['name'],mode=job['mode'],overrides=job['overrides'],scores=rows,
        selection_cost=float(np.mean([r['mean_cost'] for r in rows])),
        admissible=all(r['admissible'] for r in rows))
    write_json(out/'result.json',result)
    return result


def train_job(job):
    if job['plan'].get('vector_envs',1)>1:
        from .stochastic_vector_training import train_vector
        return train_vector(job)
    plan,mode,seed = job['plan'],job['mode'],job['seed']
    out = Path(plan['out'])/'training'/job['name']
    out.mkdir()
    device = job['device']
    agent = SAC(LearningConfig(**plan['learning']),seed=seed,device=device)
    history,validations = [],[]
    done,episode,best = 0,0,float('inf')
    start = time.perf_counter()
    while done<plan['interactions']:
        steps = min(plan['episode_steps'],plan['interactions']-done)
        process_seed = plan['training_process_seed_base']+episode
        episode_id = f'train_{seed}_{episode}'
        plant,interface,cfg,provenance = make_episode(plan,process_seed,steps,episode_id)
        condition = episode%4
        offset = int(np.random.default_rng(process_seed).integers(0,1600))
        reference = targets('standard',condition,steps,offset,plan.get('target_protocol'))
        controller = NeuralController(agent,cfg,mode,deterministic=False)
        ctx = interface.receive(plant.measurement_packet(episode_id),0.,reference[0],reference[1])
        rewards = []
        for k in range(steps):
            result,reward = evaluate_transition(plant,interface,controller,ctx,ctx.target)
            nxt = interface.receive(plant.measurement_packet(episode_id),plant.time_s,
                reference[k+1],reference[min(k+2,steps)])
            agent.replay.add(ctx.observation,controller.last_action,reward['reward'],nxt.observation,
                             terminated=False,truncated=k+1==steps)
            diag = {}
            if agent.replay.count>=plan['learning_starts']:
                diag = agent.update(diagnostics=(done+1)%128==0)
                if not all(np.isfinite(x) for x in diag.values()):
                    raise ValueError('Nonfinite learning diagnostics')
            rewards.append(reward['reward'])
            done += 1
            if done%128==0:
                history.append(dict(interactions=done,episode=episode,mean_reward=float(np.mean(rewards[-128:])),
                                    elapsed_wall_s=time.perf_counter()-start,**diag))
                pd.DataFrame(history).to_csv(out/'history.csv',index=False)
                write_json(out/'progress.json',dict(completed=done,planned=plan['interactions'],
                    updates=agent.updates,device=device,elapsed_wall_s=time.perf_counter()-start))
            ctx = nxt
        episode += 1
        if done%plan['checkpoint_every']==0 or done==plan['interactions']:
            scores = []
            for condition in range(4):
                frame,_,_ = rollout(plan,mode,plan['selection_process_seed'],'standard',condition,
                    plan['selection_steps'],agent=agent,offset=plan['selection_offset'])
                scores.append(score(frame,plan['warmup']))
            value = float(np.mean([s['mean_cost'] for s in scores]))
            validations.append(dict(interactions=done,selection_cost=value,scores=scores))
            write_json(out/'validation.json',validations)
            metadata = dict(controller=mode,nominal_controller=asdict(cfg),
                protocol_sha256=sha256(Path(plan['out'])/'protocol.json'),
                interactions=done,selection_cost=value,scope=plan['scope'])
            if all(s['admissible'] for s in scores) and value<best:
                best = value
                agent.save(out/'selected.pth',metadata)
            agent.save(out/'last.pth',metadata)
    if not (out/'selected.pth').exists():
        raise RuntimeError('No admissible development checkpoint; retain last and histories')
    result = dict(name=job['name'],mode=mode,seed=seed,device=device,interactions=done,
        updates=agent.updates,selection_cost=best,wall_s=time.perf_counter()-start,
        checkpoint=str(out/'selected.pth'),checkpoint_sha256=sha256(out/'selected.pth'),
        nominal_controller=asdict(cfg),provenance=provenance,
        optimizer_hyperparameters=asdict(agent.cfg),observation_dim=OBS_DIM)
    write_json(out/'result.json',result)
    return result


def evaluation_job(job):
    plan = job['plan']
    agent = SAC.load_for_evaluation(job['checkpoint']) if job.get('checkpoint') else None
    if agent is not None and agent.checkpoint_metadata['protocol_sha256']!=sha256(Path(plan['out'])/'protocol.json'):
        raise ValueError('Checkpoint protocol mismatch')
    out = Path(job['evaluation_out']) if 'evaluation_out' in job else Path(plan['out'])/'evaluation'
    frame,_,_ = rollout(plan,job['mode'],job['process_seed'],job['profile'],job['condition'],
        plan['evaluation_steps'],agent=agent,overrides=job.get('overrides'),offset=plan['evaluation_offset'],
        progress=out/(job['name']+'.progress.json'))
    path = out/(job['name']+'.csv')
    frame.to_csv(path,index=False)
    result = dict(name=job['name'],mode=job['mode'],training_seed=job.get('seed'),
        process_seed=job['process_seed'],profile=job['profile'],condition=job['condition'],
        trace_sha256=sha256(path),**score(frame,plan['warmup']))
    write_json(out/(job['name']+'.json'),result)
    return result


def stage(plan,name,jobs,function,workers):
    root = Path(plan['out'])
    write_json(root/(name+'_jobs.json'),[{k:v for k,v in j.items() if k!='plan'} for j in jobs],frozen=True)
    completed,failures = [],[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        futures = {pool.submit(function,j):j for j in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(dict(name=job['name'],error=repr(exc)))
            write_json(root/(name+'_status.json'),dict(completed=completed,failures=failures,planned=len(jobs)))
            print(json.dumps(dict(stage=name,completed=len(completed),failed=len(failures),planned=len(jobs))),flush=True)
    if failures:
        raise RuntimeError(f'{name}: failures preserved, no silent retries')
    return completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path('experiments/apen17549_source_coupled_revision/stochastic_model_final'))
    p.add_argument('--scale',choices=['pilot','small','full'],default='pilot')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--target-manifest',type=Path)
    p.add_argument('--power-residual-mode', choices=['full_range', 'fixed'], default='full_range',
                   help='Full physical power-flow range by default; fixed reproduces the legacy +/-0.035 cap')
    p.add_argument('--cooling-residual-mode', choices=['full_range', 'fixed'], default='full_range',
                   help='Full physical cooling-flow range by default; fixed reproduces the legacy +/-0.0875 cap')
    args = p.parse_args()
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('Two CUDA devices required for this declared run')
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('selection','training','evaluation'):
        (args.out/name).mkdir()
    pilot = args.scale=='pilot'
    small = args.scale=='small'
    plan = dict(out=str(args.out),model_root=str(args.model_root),scope='stochastic_greybox_'+args.scale,
        field_validated=False,old_field_acceptance_overridden=False,
        interactions=4096 if pilot else 768000,seeds=list(range(20260930,20260933)),vector_envs=8,
        episode_steps=512 if pilot else 2048,checkpoint_every=2048 if pilot else 128000,
        selection_steps=480 if pilot else 1800,evaluation_steps=640 if pilot else 1800,
        selection_offset=400,evaluation_offset=0,warmup=200,learning_starts=256,
        learning=asdict(LearningConfig()),training_process_seed_base=90000,selection_process_seed=8101,
        evaluation_process_seeds=[8201] if pilot else [8201,8202,8203],
        target_profiles=list(PROFILES),conditions=[0,1,2,3],target_policy='standard_only_during_training_and_selection',
        nominal_initialization='3 MPa port, 1.3 MPa condenser, 300.15 K water, 0.6 kg/s power-only initial flow',
        information='191D shared causal history, own closed-loop heat, measured pressure, one-step target preview',
        objective='(power_error/10)^2+(cooling_error/10)^2+0.002*||delta_command_kg_s||^2',
        selection='mean development cost over four conditions, admissible candidates only',
        source_mode='frozen stochastic source-load simulator; not replay of measured outputs',
        pressure_forecast='normal measured-pressure scenario; no old estimator loaded',
        controller_modes=['rrl','sac','pi','mpc'],residual_regularizer=False,
        statistical_unit='learning seed; average four conditions within each profile first',
        baseline_uncertainty='paired process realizations, not fictitious learning-seed std',
        full_convergence_claim_allowed=False,
        model_hash=sha256(args.model_root/'frozen_process.json'),
        historical_acceptance_hash=sha256(args.model_root/'release_manifest.json'),
        code_hashes={str(f):sha256(f) for f in Path('source_coupled_revision').glob('*.py')},
        target_code_hash=sha256(Path('run_four_conditions.py')),
        controller_overrides=dict(residual_power=None if args.power_residual_mode=='full_range' else .035,
                                  residual_cooling=None if args.cooling_residual_mode=='full_range' else .0875),
        require_full_action_parity=(args.power_residual_mode==args.cooling_residual_mode=='full_range'),
        power_action_interface=args.power_residual_mode,
        cooling_action_interface=args.cooling_residual_mode,
        checkpoint_compatibility='Fresh training required after action-interface changes')
    if small:
        plan.update(interactions=16384,seeds=[20260930,20260931],episode_steps=1024,
            checkpoint_every=8192,selection_steps=640,evaluation_steps=1200,
            evaluation_process_seeds=[8301],
            baseline_rule='Fixed development-check PI gains and MPC horizon; no extra tuning',
            statistical_caution='Two-seed small-budget diagnostic, not a convergence or significance claim')
    if args.target_manifest is not None:
        target_manifest = json.loads(args.target_manifest.read_text())
        if not target_manifest.get('development_gate_passed',False):
            raise ValueError('Coupled target development gate not passed')
        if target_manifest['model_hash']!=plan['model_hash']:
            raise ValueError('Coupled target model hash mismatch')
        plan.update(target_protocol=target_manifest['target_protocol'],
            joint_allocation_points=target_manifest['joint_allocation_points'],
            target_manifest_hash=sha256(args.target_manifest),
            episode_steps=min(plan['episode_steps'],int(target_manifest['waveform_horizon_s']/10)),
            allocator='shared development-identified quasi-steady joint map; no hidden-state access')
    elif not pilot:
        raise ValueError('Long training requires an accepted joint-target manifest; legacy targets are diagnostic only')
    write_json(args.out/'protocol.json',plan,frozen=True)
    _,_,base,_ = make_episode(plan,90000,1,'config')
    jobs = []
    for kp in ((1.,) if small else (.5,1.,2.)):
        for ki in ((1.,) if small else (1.,4.,16.,64.)):
            for aw in ((base.antiwindup_tau_s,) if small else (30.,120.)):
                jobs.append(dict(plan=plan,mode='pi',name=f'pi_kp{kp}_ki{ki}_aw{aw}',
                    overrides=dict(pi_kp=base.pi_kp*kp,pi_ki=base.pi_ki*ki,antiwindup_tau_s=aw)))
    for horizon in ((base.mpc_horizon,) if small else (6,12,18)):
        jobs.append(dict(plan=plan,mode='mpc',name=f'mpc_h{horizon}',overrides=dict(mpc_horizon=horizon)))
    selected = stage(plan,'selection',jobs,baseline_job,args.workers)
    winners = {mode:min((r for r in selected if r['mode']==mode and r['admissible']),
                       key=lambda r:r['selection_cost']) for mode in ('pi','mpc')}
    write_json(args.out/'selected_baselines.json',winners,frozen=True)
    jobs = [dict(plan=plan,mode=mode,seed=seed,name=f'{mode}_{seed}',device=f'cuda:{i}')
        for i,mode in enumerate(('rrl','sac')) for seed in plan['seeds']]
    trained = stage(plan,'training',jobs,train_job,min(len(jobs),6))
    controllers = [dict(mode=r['mode'],seed=r['seed'],checkpoint=r['checkpoint']) for r in trained]
    controllers += [dict(mode=m,overrides=winners[m]['overrides']) for m in ('pi','mpc')]
    jobs = []
    for c in controllers:
        for profile in PROFILES:
            for condition in range(4):
                for process_seed in plan['evaluation_process_seeds']:
                    jobs.append(dict(plan=plan,**c,profile=profile,condition=condition,process_seed=process_seed,
                        name=f'{c["mode"]}_{c.get("seed","fixed")}_{profile}_c{condition}_p{process_seed}'))
    evaluated = stage(plan,'evaluation',jobs,evaluation_job,args.workers)
    table = pd.DataFrame(evaluated)
    table.to_csv(args.out/'all_metrics.csv',index=False)
    metrics = ['mean_cost','combined_RMSE','power_RMSE','cooling_RMSE','command_change_cost','max_error']
    perseed = table.groupby(['mode','profile','training_seed'],dropna=False)[metrics].mean().reset_index()
    perseed.to_csv(args.out/'seed_metrics.csv',index=False)
    summary = perseed.groupby(['mode','profile'])[metrics].agg(['mean','std','count'])
    summary.to_csv(args.out/'summary.csv')
    write_json(args.out/'completion.json',dict(completed=True,scope=plan['scope'],
        training_runs=len(trained),interactions_per_run=plan['interactions'],evaluation_runs=len(evaluated),
        all_evaluations_admissible=all(r['admissible'] for r in evaluated),
        formal_budget_complete=args.scale=='full',convergence_established=False,
        statistical_caution='Pilot estimates are descriptive; deterministic baselines have no seed std',
        model_hash_after=sha256(args.model_root/'frozen_process.json')))


if __name__=='__main__':
    main()
