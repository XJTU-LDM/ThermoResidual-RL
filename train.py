"""Train either action formulation using the same frozen simulation recipe."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('rrl', 'sac'), required=True)
    parser.add_argument('--weight-ratio', choices=('1', '0.5', '2'), default='1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='1024 interactions and short development probes')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory; existing runs are never overwritten')
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from reproduce import load_jobs, verify
    verify()
    import torch
    from thermoresidual.audit import sha256, write_json
    from thermoresidual.simulation import make_episode
    from thermoresidual import training as training
    from thermoresidual.optimizer import SharedRecipeSAC
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu or install a compatible PyTorch build')
    suite = 'demo' if args.weight_ratio == '1' else 'weights'
    template = next(j for j in load_jobs(suite) if j['mode'] == args.controller and
                    (suite == 'demo' or float(j['ratio']) == float(args.weight_ratio)))
    plan = deepcopy(template['plan'])
    original_initial = Path(plan['resume_learner_checkpoint'])
    if sha256(original_initial) != plan['resume_learner_sha256']:
        raise ValueError('Initial learner changed')
    initial = torch.load(original_initial, map_location='cpu', weights_only=True)
    if initial['updates'] or initial['learner']['replay_count']:
        raise ValueError('Public training must start with an empty learner, not a trained policy')
    output.mkdir(parents=True)
    (output / 'training').mkdir()
    plan['out'] = str(output)
    if args.smoke:
        plan.update(interactions=1024, total_interactions=1024, vector_envs=2,
                    episode_steps=256, checkpoint_every=1024, development_workers=2,
                    development_windows=[dict(steps=256, offset=400, process_seed=8102)],
                    archive_checkpoints=False, save_learner_state=False)
    _, _, cfg, _ = make_episode(plan, 95000, 2, 'public_training_interface')
    stored = initial['metadata']['nominal_controller']
    current = asdict(cfg)
    if stored != current:
        raise ValueError('Initial learner and current control interface differ')
    initial_path = output / 'initial_learner.pth'
    torch.save(initial, initial_path)
    plan.update(resume_learner_checkpoint=str(initial_path), resume_learner_sha256=sha256(initial_path))
    write_json(output / 'protocol.json', plan, frozen=True)
    job = dict(plan=plan, mode=args.controller, seed=20260930,
               name=f'{args.controller}_20260930', device=args.device)
    write_json(output / 'job.json', job, frozen=True)
    training.SAC = SharedRecipeSAC
    result = training.train_vector(job)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
