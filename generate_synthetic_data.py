"""Generate a boundary-like dataset from a simulated closed-loop trajectory."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/synthetic')
    args = parser.parse_args()
    if args.steps <= 200 or args.seed < 0:
        parser.error('Need steps > 200 and seed >= 0')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    from reproduce import load_jobs, verify
    verify()
    from source_coupled_revision.revision_evaluation import evaluate
    import pandas as pd
    job = deepcopy(next(j for j in load_jobs('demo') if j['mode'] == 'physics'))
    job.update(steps=args.steps, process_seed=args.seed, evaluation_out=str(output),
               name=f'synthetic_physics_seed{args.seed}_steps{args.steps}')
    result = evaluate(job)
    frame = pd.read_csv(output / (job['name'] + '.csv'))
    columns = ['time_s', 'pressure_MPa', 'condenser_pressure_MPa', 'source_heat_kW',
               'water_exchange_kW', 'cooling_water_inlet_K', 'cooling_water_outlet_K',
               'command_power_kg_s', 'command_cooling_kg_s', 'actual_power_kg_s',
               'actual_cooling_kg_s', 'output_power_kW', 'output_cooling_kW']
    data = frame[columns].copy()
    data.insert(0, 'data_type', 'simulation')
    data.to_csv(output / 'synthetic_boundary.csv', index=False)
    metadata = dict(data_type='simulation', raw_field_data=False, controller='physics',
                    process_seed=args.seed, samples=len(data), cadence_s=10,
                    description='Closed-loop model outputs; not an exogenous heat replay or a measured record',
                    columns={c: ('s' if c == 'time_s' else c.rsplit('_', 1)[-1]) for c in columns},
                    metrics={k: result[k] for k in ('energy_ratio', 'mass_residual_kg', 'combined_RMSE')})
    for c in columns:
        if c.endswith('_kg_s'):
            metadata['columns'][c] = 'kg/s'
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(output / 'synthetic_boundary.csv')


if __name__ == '__main__':
    main()
