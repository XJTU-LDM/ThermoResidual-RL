"""Retune feedback and MPC on development windows, never final evaluations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def candidates(mode):
    if mode == 'pi':
        for kp, ki, aw in itertools.product((.5, 1., 2.), (1., 4., 16., 64.), (30., 120.)):
            yield dict(pi_kp=.003*kp, pi_ki=.00003*ki, antiwindup_tau_s=aw)
    elif mode == 'mpc':
        for horizon, rate in itertools.product((6, 12, 18), (.002, .01, .05)):
            yield dict(mpc_horizon=horizon, mpc_rate_penalty=rate)
    else:
        for kp, ti, td, aw in itertools.product((.15, .30, .60), (60., 120., 240.), (0., 5., 20.), (30., 120.)):
            yield dict(gspid_kp=kp, gspid_ti_s=ti, gspid_td_s=td,
                       antiwindup_tau_s=aw, gspid_gain_floor=10.)


def probe(job):
    os.chdir(ROOT)
    import numpy as np
    import torch
    from source_coupled_revision.run_stochastic_benchmark import rollout, score
    torch.set_num_threads(1)
    rows = []
    for window in job['plan']['development_windows']:
        for condition in range(4):
            frame, _, _ = rollout(job['plan'], job['mode'], window['process_seed'],
                                  'standard', condition, window['steps'],
                                  overrides=job['overrides'], offset=window['offset'])
            rows.append(score(frame, job['plan']['warmup']))
    return dict(candidate=job['candidate'], mode=job['mode'], overrides=job['overrides'],
                selection_cost=float(np.mean([s['mean_cost'] for s in rows])),
                admissible=all(s['admissible'] for s in rows), scores=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('pi', 'gspid', 'mpc'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke', action='store_true', help='One candidate and four short probes')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or args.workers < 1:
        parser.error('Use a new output directory and positive workers')
    from reproduce import load_jobs, verify
    verify()
    plan = deepcopy(load_jobs('demo')[0]['plan'])
    settings = list(candidates(args.controller))
    if args.smoke:
        settings = settings[:1]
        plan['development_windows'] = [dict(steps=256, offset=400, process_seed=8102)]
    jobs = [dict(plan=plan, mode=args.controller, candidate=i, overrides=v) for i, v in enumerate(settings)]
    out.mkdir(parents=True)
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    rows, failures = [], []
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(probe, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(dict(candidate=futures[future]['candidate'], error=repr(exc)))
            (out / 'results.json').write_text(json.dumps(dict(results=rows, failures=failures), indent=2) + '\n')
    if failures:
        raise SystemExit('Incomplete grid; failures retained in results.json')
    eligible = [r for r in rows if r['admissible']]
    if not eligible:
        raise SystemExit('No admissible candidate')
    winner = min(eligible, key=lambda r: (r['selection_cost'], r['candidate']))
    (out / 'selected.json').write_text(json.dumps(winner, indent=2) + '\n')
    print(json.dumps(winner, indent=2))


if __name__ == '__main__':
    main()
