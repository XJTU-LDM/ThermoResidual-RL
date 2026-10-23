"""CPU process-isolated physical environments with batched CUDA actors."""
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .audit import sha256, write_json
from .control import evaluate_transition, compose_residual_flow
from .learning import SAC, LearningConfig
from .convergence import plateau_status, screening_plateau_status


def development_probe(job):
    from .simulation import rollout, score
    torch.set_num_threads(1)
    plan, condition = job['plan'], job['condition']
    agent = SAC.load_for_evaluation(job['checkpoint'])
    frame, _, _ = rollout(plan, job['mode'], plan['selection_process_seed'], 'standard', condition,
                          plan['selection_steps'], agent=agent, offset=plan['selection_offset'])
    if job.get('trace_path'):
        frame.to_csv(job['trace_path'], index=False)
    return job.get('probe_id',condition), score(frame, plan['warmup'])


def replay_updates_due(previous, current, starts, ratio):
    if not np.isfinite(ratio) or not 0 < ratio <= 4:
        raise ValueError('Update ratio must be in (0, 4]')
    return (int(np.floor(max(0, current-starts+1)*ratio)) -
            int(np.floor(max(0, previous-starts+1)*ratio)))


def environment_worker(connection,plan,slot,mode):
    from .simulation import make_episode, targets
    from .feasible_actions import compose_actor_flow
    torch.set_num_threads(1)
    class RawController:
        raw = None
        def act(self,ctx):
            return self.raw
    generation = 0
    def reset():
        nonlocal generation
        episode = plan.get('training_episode_offset', 0)+slot+generation*plan['vector_envs']
        generation += 1
        process_seed = plan['training_process_seed_base']+episode
        episode_id = f'vector_{episode}'
        plant,interface,cfg,provenance = make_episode(plan,process_seed,plan['episode_steps'],episode_id)
        offset = int(np.random.default_rng(process_seed).integers(0,1600))
        reference = targets('standard',episode%4,plan['episode_steps'],offset,plan.get('target_protocol'))
        ctx = interface.receive(plant.measurement_packet(episode_id),0.,reference[0],reference[1])
        return plant,interface,cfg,provenance,reference,ctx,episode_id
    try:
        plant,interface,cfg,provenance,reference,ctx,episode_id = reset()
        k = 0
        controller = RawController()
        connection.send(dict(obs=ctx.observation,config=asdict(cfg),provenance=provenance))
        while True:
            message = connection.recv()
            if message is None:
                break
            u = np.asarray(message,float)
            controller.raw = compose_actor_flow(u, ctx, cfg, mode,
                                               plan['learning'].get('action_coordinates', 'raw'))
            previous = interface.command.copy()
            result,reward = evaluate_transition(plant,interface,controller,ctx,ctx.target)
            if np.any(abs(interface.command-previous)>cfg.rate_per_step+1e-8) or abs(result['mass_residual_kg'])>1e-6:
                raise ValueError('Training transition violated physical interface')
            k += 1
            nxt = interface.receive(plant.measurement_packet(episode_id),plant.time_s,
                                    reference[k],reference[min(k+1,plan['episode_steps'])])
            final_obs = nxt.observation
            truncated = k==plan['episode_steps']
            if truncated:
                plant,interface,cfg,provenance,reference,ctx,episode_id = reset()
                k = 0
            else:
                ctx = nxt
            connection.send(dict(obs=ctx.observation,next_obs=final_obs,reward=reward['reward'],
                                 truncated=truncated))
    except EOFError:
        pass
    except Exception:
        connection.send(dict(error=traceback.format_exc()))
    finally:
        connection.close()


def train_vector(job):
    from .simulation import rollout, score
    from .control import ControlConfig, OBS_DIM
    plan,mode,seed = job['plan'],job['mode'],job['seed']
    if plan.get('development_windows') and plan.get('development_workers',1)<=1:
        raise ValueError('Multiple development windows require the explicit parallel probe path')
    out = Path(plan['out'])/'training'/job['name']
    out.mkdir()
    if plan.get('resume_learner_checkpoint'):
        path = Path(plan['resume_learner_checkpoint'])
        if sha256(path) != plan['resume_learner_sha256']:
            raise ValueError('Learner checkpoint changed')
        agent = SAC.load_learner_state(path, device=job['device'])
        if asdict(agent.cfg) != asdict(LearningConfig(**plan['learning'])):
            raise ValueError('Resume configuration must preserve learner hyperparameters')
    else:
        agent = SAC(LearningConfig(**plan['learning']),seed=seed,device=job['device'])
    context = mp.get_context('spawn')
    workers,connections = [],[]
    for slot in range(plan['vector_envs']):
        local,remote = context.Pipe()
        worker = context.Process(target=environment_worker,args=(remote,plan,slot,mode))
        worker.start()
        remote.close()
        workers.append(worker)
        connections.append(local)
    def receive(connection):
        if not connection.poll(600):
            raise TimeoutError('Physical environment produced no response in 600 seconds')
        message = connection.recv()
        if 'error' in message:
            raise RuntimeError(message['error'])
        return message
    histories,validations = [],[]
    done,best,next_checkpoint = 0,float('inf'),plan['checkpoint_every']
    stopping = dict(plateau=False,reason='budget_limit')
    start = time.perf_counter()
    dev_pool = None
    try:
        initial = [receive(c) for c in connections]
        if any(x['config']!=initial[0]['config'] for x in initial):
            raise ValueError('Environment controller configurations differ')
        cfg = ControlConfig(**initial[0]['config'])
        if plan.get('resume_learner_checkpoint'):
            metadata = agent.learner_resume_metadata
            if metadata['controller'] != mode or metadata['nominal_controller'] != asdict(cfg):
                raise ValueError('Learner resume controller interface mismatch')
            write_json(out/'learner_resume.json', dict(source_metadata=metadata,
                replay_count=agent.replay.count, optimizer_updates=agent.updates,
                environment_resume='Initialize independent simulated episodes'))
        if plan.get('development_workers', 1) > 1:
            dev_pool = ProcessPoolExecutor(plan['development_workers'], mp_context=context)
        obs = np.array([x['obs'] for x in initial],np.float32)
        rewards, residual_edges = [], []
        while done<plan['interactions']:
            if plan['interactions']-done<len(connections):
                raise ValueError('Budget must be divisible by vector environments')
            with torch.no_grad():
                u,_ = agent.actor(torch.as_tensor(obs,device=agent.device))
                action = u.cpu().numpy()
            residual_edges.extend((abs(action)>=.99).astype(float))
            for connection,a in zip(connections,action):
                connection.send(a)
            returned = [receive(c) for c in connections]
            for o,a,r in zip(obs,action,returned):
                agent.replay.add(o,a,r['reward'],r['next_obs'],terminated=False,truncated=r['truncated'])
                rewards.append(r['reward'])
            obs = np.array([r['obs'] for r in returned],np.float32)
            previous_done = done
            done += len(connections)
            diag = {}
            updates = replay_updates_due(previous_done, done, plan['learning_starts'],
                                         plan.get('updates_per_step', 1.))
            for update in range(updates):
                diag = agent.update(diagnostics=done%128==0 and update==updates-1)
                if not all(np.isfinite(x) for x in diag.values()):
                    raise ValueError('Nonfinite optimizer state')
            if done%128==0:
                histories.append(dict(interactions=done,mean_reward=float(np.mean(rewards[-128:])),
                    action_edge_power=float(np.mean(np.asarray(residual_edges[-128:])[:,0])),
                    action_edge_cooling=float(np.mean(np.asarray(residual_edges[-128:])[:,1])),
                    elapsed_wall_s=time.perf_counter()-start,**diag))
                pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
                write_json(out/'progress.json',dict(completed=done,planned=plan['interactions'],
                    updates=agent.updates,device=job['device'],elapsed_wall_s=time.perf_counter()-start))
            if done>=next_checkpoint or done==plan['interactions']:
                scores = []
                if dev_pool is not None:
                    candidate = out/'development_candidate.pth'
                    agent.save(candidate, dict(controller=mode, nominal_controller=asdict(cfg),
                        protocol_sha256=sha256(Path(plan['out'])/'protocol.json'), interactions=done))
                    folder = out/'development_traces'
                    folder.mkdir(exist_ok=True)
                    jobs = []
                    windows=plan.get('development_windows',[dict(steps=plan['selection_steps'],
                        offset=plan['selection_offset'],process_seed=plan['selection_process_seed'])])
                    for window_id,window in enumerate(windows):
                        probe_plan={**plan,'selection_steps':window['steps'],
                            'selection_offset':window['offset'],'selection_process_seed':window['process_seed']}
                        for condition in range(4):
                            suffix=f'{done}_condition{condition}' if len(windows)==1 else f'{done}_window{window_id}_condition{condition}'
                            jobs.append(dict(plan=probe_plan,mode=mode,checkpoint=str(candidate),condition=condition,
                                probe_id=4*window_id+condition,trace_path=str(folder/(suffix+'.csv'))
                                if plan.get('archive_checkpoints',False) else None))
                    scores = [s for _, s in sorted(dev_pool.map(development_probe, jobs))]
                for condition in range(4) if dev_pool is None else ():
                    frame,_,_ = rollout(plan,mode,plan['selection_process_seed'],'standard',condition,
                        plan['selection_steps'],agent=agent,offset=plan['selection_offset'])
                    scores.append(score(frame,plan['warmup']))
                    if plan.get('archive_checkpoints',False):
                        folder = out/'development_traces'
                        folder.mkdir(exist_ok=True)
                        frame.to_csv(folder/f'{done}_condition{condition}.csv',index=False)
                value = float(np.mean([s['mean_cost'] for s in scores]))
                recent = pd.DataFrame(histories).tail(16)
                training_diagnostics = {k: float(recent[k].mean()) for k in
                    ('mean_reward', 'actor_grad_norm', 'actor_first_layer_inactive_fraction') if k in recent}
                validations.append(dict(interactions=done,selection_cost=value,scores=scores,
                                        training_diagnostics=training_diagnostics))
                write_json(out/'validation.json',validations)
                metadata = dict(controller=mode,nominal_controller=asdict(cfg),
                    protocol_sha256=sha256(Path(plan['out'])/'protocol.json'),interactions=done,
                    selection_cost=value,scope=plan['scope'])
                if all(s['admissible'] for s in scores) and value<best:
                    best = value
                    agent.save(out/'selected.pth',metadata)
                    if plan.get('save_learner_state', False):
                        agent.save_learner_state(out/'selected_learner.pth', metadata)
                agent.save(out/'last.pth',metadata)
                if plan.get('archive_checkpoints',False):
                    folder = out/'checkpoints'
                    folder.mkdir(exist_ok=True)
                    agent.save(folder/f'step_{done}.pth',metadata)
                if plan.get('save_learner_state', False):
                    agent.save_learner_state(out/'learner_state.pth', metadata)
                next_checkpoint += plan['checkpoint_every']
                if plan.get('plateau_rule'):
                    stopping = (screening_plateau_status if plan.get('health_guarded_plateau') else plateau_status)(
                        validations, plan['plateau_rule'])
                    write_json(out/'convergence.json',dict(interactions=done,**stopping))
                    if stopping['plateau']:
                        break
        if not (out/'selected.pth').exists():
            raise RuntimeError('No admissible development checkpoint')
        result = dict(name=job['name'],mode=mode,seed=seed,device=job['device'],interactions=done,
            updates=agent.updates,selection_cost=best,wall_s=time.perf_counter()-start,
            checkpoint=str(out/'selected.pth'),checkpoint_sha256=sha256(out/'selected.pth'),
            budget_limit=plan['interactions'],stopping=stopping,
            termination_reason='local_plateau' if stopping['plateau'] else 'budget_limit',
            nominal_controller=asdict(cfg),provenance=initial[0]['provenance'],
            optimizer_hyperparameters=asdict(agent.cfg),observation_dim=OBS_DIM)
        write_json(out/'result.json',result)
        return result
    finally:
        if dev_pool is not None:
            dev_pool.shutdown(wait=True, cancel_futures=True)
        for connection in connections:
            try:
                connection.send(None)
            except (BrokenPipeError,EOFError,OSError):
                pass
        for worker in workers:
            worker.join(timeout=620)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        for connection in connections:
            connection.close()
