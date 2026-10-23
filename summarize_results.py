"""Pool matched trajectory metrics using sample-weighted mean squared errors."""
import argparse
from pathlib import Path


def summarize(frame):
    import numpy as np
    import pandas as pd
    identity = ['mode', 'profile', 'condition', 'process_seed']
    if frame.duplicated(identity).any():
        raise ValueError('Duplicate trajectory identities; summarize one suite/scenario at a time')
    required = ['samples', 'power_RMSE', 'cooling_RMSE', 'mean_cost', 'rate_violations', 'bound_violations']
    if not np.isfinite(frame[required].to_numpy()).all() or (frame.samples <= 0).any():
        raise ValueError('Invalid trajectory metrics')
    rows = []
    for (mode, profile), group in frame.groupby(['mode', 'profile']):
        w = group.samples.to_numpy()
        p = np.average(group.power_RMSE**2, weights=w)
        q = np.average(group.cooling_RMSE**2, weights=w)
        rows.append(dict(mode=mode, profile=profile, trajectories=len(group),
                         samples=int(w.sum()), power_RMSE=float(np.sqrt(p)),
                         cooling_RMSE=float(np.sqrt(q)), combined_RMSE=float(np.sqrt((p+q)/2)),
                         mean_cost=float(np.average(group.mean_cost, weights=w)),
                         rate_violations=int(group.rate_violations.sum()),
                         bound_violations=int(group.bound_violations.sum())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/main/metrics.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/main/profile_summary.csv'))
    args = parser.parse_args()
    import pandas as pd
    result = summarize(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
