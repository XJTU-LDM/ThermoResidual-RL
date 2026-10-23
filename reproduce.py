"""Run the public reference simulations without restricted field records."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ('rrl', 'sac', 'pi', 'gspid', 'mpc', 'physics')


def verify(root=ROOT):
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Missing or modified release file: {name}')
    return len(manifest['files'])


def load_jobs(suite, root=ROOT):
    return json.loads((root / 'configs' / f'{suite}_jobs.json').read_text())


def execute(job):
    os.chdir(ROOT)
    from source_coupled_revision.revision_evaluation import evaluate
    result = evaluate(job)
    return dict(name=job['name'], mode=job['mode'], profile=job['profile'],
                condition=job['condition'], process_seed=job['process_seed'],
                **{k: v for k, v in result.items() if k not in ('job', 'provenance')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('demo', 'main', 'stress', 'sensor', 'weights'), default='demo')
    parser.add_argument('--controllers', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--profile', choices=('standard', 'medium_unseen', 'very_fast_unseen', 'mixed_step_ramp'))
    parser.add_argument('--condition', type=int, choices=range(4))
    parser.add_argument('--steps', type=int, default=256, help='Demo only; formal suites retain 3200 steps')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='0 runs every matching job')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or args.steps <= 200:
        parser.error('Need limit >= 0, workers >= 1 and steps > 200 warmup steps')
    print(f'Checksums verified: {verify()} files', flush=True)
    if args.verify_only:
        return
    if args.suite != 'demo' and args.steps != 256:
        parser.error('--steps changes only the demo, not archived evaluation suites')
    out = args.output.resolve() / (f'demo_{args.steps}' if args.suite == 'demo' else args.suite)
    jobs = []
    for original in load_jobs(args.suite):
        if original['mode'] not in args.controllers:
            continue
        if args.profile and original['profile'] != args.profile:
            continue
        if args.condition is not None and original['condition'] != args.condition:
            continue
        job = deepcopy(original)
        if args.suite == 'demo':
            job['steps'] = args.steps
            job['name'] = f"{job['mode']}_simulated_demo_{args.steps}"
        job['evaluation_out'] = str(out)
        jobs.append(job)
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        parser.error('No jobs match these filters')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{len(jobs)} {row['name']}: combined RMSE = {row['combined_RMSE']:.6f} kW", flush=True)
            except Exception as exc:
                failures.append(dict(name=job['name'], error=repr(exc)))
            status = dict(data_type='simulation', planned=len(jobs), completed=len(rows),
                          failures=failures, results=sorted(rows, key=lambda r: r['name']))
            (out / 'run_status.json').write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    if rows:
        import pandas as pd
        pd.DataFrame(rows).sort_values('name').to_csv(out / 'metrics.csv', index=False)
    if failures:
        raise SystemExit(f'{len(failures)} evaluations failed; see {out / "run_status.json"}')
    print(f'Simulation outputs: {out}')


if __name__ == '__main__':
    main()
