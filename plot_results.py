"""Plot targets, outputs, errors and commands from simulated trajectories."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/demo_256'))
    parser.add_argument('--output', type=Path, default=Path('outputs/figures'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    colors = dict(rrl='#C84D4D', sac='#4477AA', pi='#228833', gspid='#AA3377', mpc='#EE9933', physics='#777777')
    traces = []
    for path in sorted(args.input.glob('*.csv')):
        meta = path.with_suffix('.json')
        if not meta.exists():
            continue
        result = json.loads(meta.read_text())
        if 'job' in result:
            traces.append((result['job'], pd.read_csv(path)))
    if not traces:
        parser.error('No trajectory CSV/JSON pairs found')
    identities = {(j['profile'], j['condition'], j['process_seed'], j.get('stress', 'reference'),
                   j.get('sensor'), j.get('predictor'), j.get('pressure_offset', 0.),
                   j.get('ratio', 1.), j['steps']) for j, _ in traces}
    if len(identities) != 1:
        parser.error('Plot one matched profile/condition/process realization at a time')
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout='constrained')
    reference = traces[0][1]
    for col, branch in enumerate(('power', 'cooling')):
        axes[0, col].plot(reference.time_s/60, reference[f'target_{branch}_kW'], 'k--', lw=1.3, label='Target')
        for job, frame in traces:
            mode, t = job['mode'], frame.time_s/60
            axes[0, col].plot(t, frame[f'output_{branch}_kW'], color=colors[mode], lw=1, label=mode)
            axes[1, col].plot(t, frame[f'output_{branch}_kW']-frame[f'target_{branch}_kW'], color=colors[mode], lw=1)
            axes[2, col].plot(t, frame[f'command_{branch}_kg_s'], color=colors[mode], lw=1)
        axes[0, col].set_title(branch.capitalize())
        axes[0, col].set_ylabel('Output (kW)')
        axes[1, col].set_ylabel('Error (kW)')
        axes[2, col].set_ylabel('Command (kg/s)')
        axes[2, col].set_xlabel('Time (min)')
        axes[0, col].legend(fontsize=7, ncol=2, loc='best', framealpha=.9)
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(args.output / f'simulated_tracking.{suffix}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
