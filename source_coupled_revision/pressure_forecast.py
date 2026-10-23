"""Causal pressure forecasting with five-member ensembles and simple predictors."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .audit import sha256
from .timeline import checked_times, segments

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/apen17549_source_coupled_revision"
PRESSURE_COLUMN = "P1-井口压力(MPa)"
BLOCK_NAMES = ("fit", "selection", "calibration", "middle", "late")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class Block:
    history: np.ndarray
    current_time_s: np.ndarray
    target_time_s: np.ndarray
    target: np.ndarray
    runs: list


def build_blocks(times, pressure, cuts, history=7):
    times = checked_times(times)
    pressure = np.asarray(pressure, dtype=float)
    if pressure.shape != times.shape or not np.isfinite(pressure).all():
        raise ValueError("Finite recorded pressure required; no interpolation")
    if len(cuts) != 6 or np.any(np.diff(cuts) <= 0):
        raise ValueError("Five chronological blocks required")
    result = {}
    for name, left, right in zip(BLOCK_NAMES, cuts[:-1], cuts[1:]):
        examples, targets, current_t, target_t, runs = [], [], [], [], []
        for segment in segments(times):
            ids = np.arange(segment["start_row"], segment["end_row_exclusive"])
            ids = ids[(times[ids] >= left) & (times[ids] < right)]
            if len(ids) <= history:
                continue
            runs.append(ids)
            for k in range(history, len(ids)):
                window = ids[k-history:k+1]
                if not np.allclose(np.diff(times[window]), 60, rtol=0, atol=1e-6):
                    continue
                examples.append(pressure[window[:-1]])
                targets.append(pressure[window[-1]])
                current_t.append(times[window[-2]])
                target_t.append(times[window[-1]])
        if not examples:
            raise ValueError(f"No complete causal windows in {name}")
        result[name] = Block(np.asarray(examples), np.asarray(current_t),
                             np.asarray(target_t), np.asarray(targets), runs)
    return result


def read_data(protocol):
    frozen = json.loads((BASE / "input_manifest.json").read_text())
    for dataset in ("pressure", "control"):
        if sha256(ROOT / frozen[dataset]["path"]) != frozen[dataset]["sha256"]:
            raise ValueError(f"Input hash changed: {dataset}")
    frame = pd.read_excel(ROOT / frozen["pressure"]["path"])
    boundary = pd.read_csv(ROOT / frozen["control"]["path"], usecols=["time"])
    times = pd.to_numeric(frame["time"], errors="raise").to_numpy(float) * 60
    pressure = pd.to_numeric(frame[PRESSURE_COLUMN], errors="raise").to_numpy(float)
    ct = checked_times(boundary["time"].to_numpy(float) * 60)
    cuts = [float(ct[i] if i < len(ct) else np.nextafter(ct[-1], np.inf))
            for i in protocol["control_cut_rows"]]
    blocks = build_blocks(times, pressure, cuts, protocol["history_samples"])
    return blocks, times, pressure, dict(input_manifest=frozen, cut_times_s=cuts)


def features(history, times, with_time):
    history, times = np.asarray(history), np.asarray(times)
    return np.column_stack((history, times / 60)) if with_time else history.copy()


class ForecastNet(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def worker(protocol, out, variant, device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; no silent CPU fallback")
    blocks, _, _, metadata = read_data(protocol)
    fit, selection = blocks["fit"], blocks["selection"]
    with_time = variant == "with_time"
    x = features(fit.history, fit.current_time_s, with_time)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    dy = fit.target - fit.history[:, -1]
    ym, ys = float(dy.mean()), max(float(dy.std()), 1e-6)
    xfit = torch.tensor((x-xm)/xs, dtype=torch.float32, device=device)
    yfit = torch.tensor((dy-ym)/ys, dtype=torch.float32, device=device)
    xsel = torch.tensor((features(selection.history, selection.current_time_s, with_time)-xm)/xs,
                        dtype=torch.float32, device=device)
    lastsel = torch.tensor(selection.history[:, -1], dtype=torch.float32, device=device)
    ysel = torch.tensor(selection.target, dtype=torch.float32, device=device)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", dict(variant=variant, device=device,
               gpu_name=torch.cuda.get_device_name(torch.device(device)), torch_version=torch.__version__,
               source_sha256=sha256(__file__), protocol=protocol, metadata=metadata,
               normalizers_fit_only=True, label="next_recorded_pressure_minus_latest_recorded_pressure",
               pressure_datum="recorded_MPa_not_confirmed_absolute", outer_blocks_used_for_selection=False))
    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ForecastNet(x.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
        best, history, best_epoch = math.inf, [], None
        started = time.perf_counter()
        for epoch in range(1, protocol["epochs"]+1):
            model.train()
            loss = ((model(xfit)-yfit)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss seed={seed} epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % protocol["check_every_epochs"] == 0:
                model.eval()
                with torch.no_grad():
                    selected = lastsel + model(xsel)*ys + ym
                    mse = float(((selected-ysel)**2).mean())
                    train_mse = float((((model(xfit)-yfit)*ys)**2).mean())
                history.append(dict(seed=seed, epoch=epoch, train_MSE_MPa2=train_mse,
                                    selection_MSE_MPa2=mse, elapsed_s=time.perf_counter()-started))
                if mse < best:
                    best, best_epoch = mse, epoch
                    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                      variant=variant, x_mean=xm.tolist(), x_scale=xs.tolist(),
                                      y_mean=ym, y_scale=ys, seed=seed, epoch=epoch,
                                      selection_MSE_MPa2=best, source_sha256=sha256(__file__),
                                      metadata=metadata)
                    torch.save(checkpoint, out / f"seed_{seed}.pth")
                pd.DataFrame(history).to_csv(out / f"history_{seed}.csv", index=False)
                write_json(out / "progress.json", dict(status="running", seed=seed, epoch=epoch,
                            best_epoch=best_epoch, best_selection_RMSE_MPa=math.sqrt(best)))
        print(json.dumps(dict(variant=variant, seed=seed, best_epoch=best_epoch,
                              selection_RMSE_MPa=math.sqrt(best), elapsed_s=time.perf_counter()-started)), flush=True)
    write_json(out / "progress.json", dict(status="completed", completed_seeds=protocol["seeds"]))


class Ensemble:
    def __init__(self, directory, seeds, device="cpu"):
        self.device, self.members = device, []
        for seed in seeds:
            state = torch.load(directory / f"seed_{seed}.pth", map_location="cpu", weights_only=True)
            model = ForecastNet(len(state["x_mean"])).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            self.members.append((model, state))

    @torch.no_grad()
    def predict(self, history, times):
        predictions = []
        for model, state in self.members:
            x = features(history, times, state["variant"] == "with_time")
            x = (x-np.asarray(state["x_mean"])) / np.asarray(state["x_scale"])
            y = model(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
            predictions.append(history[:, -1]+y*state["y_scale"]+state["y_mean"])
        values = np.asarray(predictions, dtype=float)
        return values.mean(0), values.std(0, ddof=1)


def simple_models(fit):
    mean = float(fit.target.mean())
    last = float(fit.target[-1])
    models = {"fit_mean": lambda h, t: np.full(len(h), mean),
              "fit_last": lambda h, t: np.full(len(h), last),
              "persistence": lambda h, t: h[:, -1].copy()}
    for lags in (1, 7):
        x = fit.history[:, -lags:]
        xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = np.column_stack((np.ones(len(x)), (x-xm)/xs))
        penalty = np.eye(z.shape[1])*1e-3
        penalty[0, 0] = 0
        beta = np.linalg.solve(z.T@z+penalty, z.T@(fit.target-fit.history[:, -1]))
        def predict(h, t, lags=lags, xm=xm, xs=xs, beta=beta):
            return h[:, -1] + np.column_stack((np.ones(len(h)), (h[:, -lags:]-xm)/xs)) @ beta
        models[f"AR{lags}"] = predict
    return models


def metrics(y, pred):
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonempty finite evaluation required")
    err = pred-y
    denom = float(np.sum((y-y.mean())**2))
    return dict(n=len(y), MAE_MPa=float(abs(err).mean()), RMSE_MPa=float(np.sqrt(np.mean(err**2))),
                MaxAE_MPa=float(abs(err).max()), R2=float(1-np.sum(err**2)/denom) if denom > 0 else None,
                nonpositive_prediction_count=int((pred <= 0).sum()))


def conformal_multiplier(y, mu, std, floor, coverage):
    scores = np.sort(abs(y-mu)/np.maximum(std, floor))
    k = math.ceil((len(scores)+1)*coverage)
    if k > len(scores):
        raise ValueError("Insufficient calibration samples")
    return float(scores[k-1])


def rollout(predict, history, times, steps):
    """No true pressure is supplied after the common dropout start."""
    h, clock, outputs = history.copy(), times.copy(), []
    for _ in range(steps):
        y = np.asarray(predict(h, clock), dtype=float)
        if y.shape != (len(h),) or not np.isfinite(y).all():
            raise ValueError("Invalid recursive prediction; no clipping or truth replacement")
        outputs.append(y)
        h = np.column_stack((h[:, 1:], y))
        clock += 60
    return np.column_stack(outputs)


def evaluate(protocol, out):
    torch.set_num_threads(1)
    blocks, times, pressure, metadata = read_data(protocol)
    predictors = simple_models(blocks["fit"])
    ensembles = {v: Ensemble(out / v, protocol["seeds"]) for v in ("no_time", "with_time")}
    for variant, model in ensembles.items():
        predictors[f"ensemble_{variant}"] = lambda h, t, model=model: model.predict(h, t)[0]
    local = out / "local_only"
    local.mkdir(exist_ok=False)
    rows, coverage_rows, calibration = [], [], {}
    for variant, model in ensembles.items():
        block = blocks["calibration"]
        mu, sigma = model.predict(block.history, block.current_time_s)
        calibration[variant] = {str(level): conformal_multiplier(block.target, mu, sigma,
                                protocol["sigma_floor_MPa"], level) for level in (.90, .95)}
    for name, block in blocks.items():
        for method, predict in predictors.items():
            pred = predict(block.history, block.current_time_s)
            rows.append(dict(block=name, method=method, **metrics(block.target, pred)))
            pd.DataFrame(dict(target_time_s=block.target_time_s, recorded_pressure_MPa=block.target,
                              predicted_pressure_MPa=pred)).to_csv(local / f"{name}_{method}.csv", index=False)
        for variant, model in ensembles.items():
            mu, sigma = model.predict(block.history, block.current_time_s)
            scale = np.maximum(sigma, protocol["sigma_floor_MPa"])
            for level in (.9, .95):
                half = calibration[variant][str(level)]*scale
                coverage_rows.append(dict(block=name, variant=variant, nominal_coverage=level,
                    empirical_coverage=float((abs(mu-block.target) <= half).mean()),
                    mean_interval_width_MPa=float((2*half).mean()),
                    interval_scope="empirical_chronological_calibration_not_exchangeability_guarantee"))
    pd.DataFrame(rows).to_csv(out / "one_step_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out / "coverage.csv", index=False)
    write_json(out / "calibration.json", calibration)
    # All methods share the same nonoverlapping outage starts and initial noise.
    stress_rows = []
    horizons = protocol["dropout_horizons_samples"]
    maximum = max(horizons)
    for name in ("selection", "middle", "late"):
        initial, truth, clocks = [], [], []
        for ids in blocks[name].runs:
            for start in range(7, len(ids)-maximum+1, maximum):
                window = ids[start-7:start+maximum]
                if not np.allclose(np.diff(times[window]), 60, rtol=0, atol=1e-6):
                    continue
                initial.append(pressure[window[:7]])
                truth.append(pressure[window[7:]])
                clocks.append(times[window[6]])
        initial, truth, clocks = map(np.asarray, (initial, truth, clocks))
        noise = np.random.default_rng(protocol["noise_seed"]).normal(size=initial.shape)
        for noise_std in protocol["noise_std_MPa"]:
            for method, predict in predictors.items():
                output = rollout(predict, initial+noise_std*noise, clocks, maximum)
                for horizon in horizons:
                    stress_rows.append(dict(block=name, method=method, horizon_samples=horizon,
                        nominal_horizon_s=horizon*60, initial_history_noise_std_MPa=noise_std,
                        aggregation="all_samples_within_first_horizon_steps",
                        n_outages=len(initial), **metrics(truth[:, :horizon].ravel(), output[:, :horizon].ravel())))
                pd.DataFrame(dict(outage_start_s=np.repeat(clocks+60, maximum),
                    offset_samples=np.tile(np.arange(1, maximum+1), len(clocks)),
                    recorded_pressure_MPa=truth.ravel(), predicted_pressure_MPa=output.ravel())).to_csv(
                        local / f"dropout_{name}_{method}_noise{noise_std}.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(out / "dropout_metrics.csv", index=False)
    write_json(out / "completion.json", dict(status="completed", pressure_network_fits=10,
               controller_training_runs=0, models=list(predictors), metadata=metadata,
               forecast_cadence_s=60, all_checkpoints_exist=True,
               coupled_controller_deployment_validated=False,
               observation_latency="zero_latency_recorded_sample_scenario_not_sensor_calibration",
               scope="recorded_pressure_forecasting_not_action_conditioned_source_model",
               local_pointwise_sha256={p.name: sha256(p) for p in local.glob("*.csv")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", choices=["no_time", "with_time"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.worker:
        worker(protocol, args.out, args.worker, args.device)
        return
    if len(set(args.gpus)) != 2 or any(g < 0 or g >= torch.cuda.device_count() for g in args.gpus):
        raise ValueError("Two distinct available GPUs required")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "launcher.json", dict(source_sha256=sha256(__file__),
        protocol_sha256=sha256(args.protocol), physical_gpus=args.gpus,
        gpu_names=[torch.cuda.get_device_name(g) for g in args.gpus], status="running"))
    processes, handles = [], []
    try:
        for variant, gpu in zip(("no_time", "with_time"), args.gpus):
            log = (args.out / f"{variant}_worker.log").open("w")
            handles.append(log)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1"}
            command = [sys.executable, "-m", "source_coupled_revision.pressure_forecast",
                       "--protocol", str(args.protocol.resolve()), "--out", str((args.out / variant).resolve()),
                       "--worker", variant, "--device", "cuda:0"]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            write_json(args.out / "process_status.json", dict(
                processes=[dict(pid=p.pid, gpu=g, exit_code=p.poll()) for p, g in zip(processes, args.gpus)]))
            time.sleep(2)
        codes = [p.wait() for p in processes]
        write_json(args.out / "process_status.json", dict(exit_codes=codes, status="completed" if not any(codes) else "failed"))
        if any(codes):
            raise RuntimeError(f"Workers failed: {codes}; logs retained, no automatic retry")
        evaluate(protocol, args.out)
        print((args.out / "completion.json").read_text(), flush=True)
    finally:
        for p in processes:
            p.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
