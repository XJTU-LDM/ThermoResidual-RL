# ThermoResidual-RL

**Physics-guided residual reinforcement learning for coordinated geothermal power and cooling control under model mismatch.**

[中文说明](README.zh-CN.md) | [Experiment protocol](docs/PROTOCOL.md) | [Data and assets](docs/DATA.md) | [Reproducibility](docs/REPRODUCIBILITY.md)

This repository provides the conservative stochastic source--load simulator,
six controllers, training implementation, frozen policy checkpoints and
simulation examples for the accompanying ThermoResidual-RL study. It supports
running the controllers without access to the project's restricted field records.

## Method

ThermoResidual-RL uses a nominal thermodynamic allocator as an explicit action
reference and learns dynamic corrections from recent measurements. The source,
ammonia inventory, power branch, adsorption cooling branch and condenser return
are coupled through mass and energy balances. Stochastic effective hydraulic
and heat-transfer parameters introduce temporally correlated process variation.

The residual action mapping covers the complete admissible branch command
range. RRL and the information-aligned full-action soft actor--critic (SAC)
share measurements, history, target preview, architecture, initialization,
optimizer settings, interaction budget and hard actuator projection. Their
action composition differs. The main comparison uses no model-assisted online
refinement.

## Included Controllers

| CLI name | Controller | Role |
|---|---|---|
| `rrl` | ThermoResidual-RL | Thermodynamic action reference plus learned correction |
| `sac` | Full-action SAC | Information-aligned learned full command |
| `pi` | Physics+PI | Same allocator plus proportional--integral feedback |
| `gspid` | Gain-scheduled proportional--integral--derivative control | Scheduled feedback with anti-windup |
| `mpc` | Constrained model predictive control | Nominal-model prediction and OSQP optimization |
| `physics` | Physics-only | Nominal thermodynamic allocation |

All use their own causal closed-loop measurements and the common branch-flow
and command-rate constraints. Process seeds select common stochastic parameter
streams, not identical controller outputs.

## Installation

Tested with **Python 3.11 on Linux**. Frozen-policy inference and examples run
on CPU. CUDA is optional for training; physical environments are CPU workers.
Run commands from the repository root unless noted otherwise.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file uses portable `torch==2.6.0`, without a machine-specific
CUDA suffix. For a particular accelerator, install a compatible PyTorch build
in the environment. The original package versions and CUDA build are recorded
in `environment-tested.json`. Windows/macOS have not been execution-tested.

## Quick Start

Verify the shipped inputs without importing PyTorch:

```bash
python reproduce.py --verify-only
```

Run a short comparison of all six frozen controllers:

```bash
python reproduce.py --steps 256 --workers 2
python plot_results.py --input outputs/demo_256
```

Each demo has 256 decision steps at 10 s per step. The first 200 steps are
warmup, so only 56 steps contribute to demo tracking metrics. The demo checks
execution; use the formal suite below to reproduce the longer evaluation.
Generated CSV, JSON, status records and metrics are written under `outputs/`.
The plot command exports PDF, SVG and PNG. Existing completed evaluations are
reused only if their job configuration and trajectory checksum still match.

Run one controller only:

```bash
python reproduce.py --controllers rrl --steps 256
```

Generate a boundary-like, explicitly labelled **simulation dataset**:

```bash
python generate_synthetic_data.py --steps 512 --seed 1729
```

A generated example is already included in `data/synthetic/`. Pressure,
source extraction, condenser conditions, flow and outputs originate from the
same closed-loop simulation. This dataset is not a copy or anonymization of
field measurements, and is not replayed as an action-independent heat input.

## Main Evaluation

```bash
# One formal-duration example (3200 steps, 200-step warmup)
python reproduce.py --suite main --controllers rrl --limit 1

# Entire six-controller matrix: 480 trajectories
python reproduce.py --suite main --workers 4
python summarize_results.py
```

The matrix contains 6 controllers x 4 target families x 4 operating conditions
x 5 process realizations. The target families are `standard`, `medium_unseen`,
`very_fast_unseen` and `mixed_step_ramp`. Final target ranges in this revised
environment are 13--15 kW power / 14--18 kW cooling for the low-amplitude
conditions and 12--16 kW / 12--20 kW for high-amplitude conditions.

Filter a matched dynamic example for plotting:

```bash
python reproduce.py --suite main --profile very_fast_unseen --condition 0 --limit 6
```

`--limit` selects jobs in manifest order, not a statistical subsample. To plot
several controllers on one realization, place their matching CSV/JSON pairs in
one directory; the plotting tool rejects mixed profiles or process seeds.
Numerical latency depends on CPU, BLAS threads and system load.

## Training From Scratch

Both algorithms start from the included paired, untrained initial learner
states: zero optimizer updates and empty replay. These states contain model
weights and normalization parameters, not recorded field trajectories.

```bash
# Short training-interface checks, not paper performance results
python train.py --controller rrl --smoke --output outputs/train_smoke_rrl
python train.py --controller sac --smoke --output outputs/train_smoke_sac

# Full matched recipe, 768000 interactions per algorithm
python train.py --controller rrl --device cuda:0 --output outputs/train_rrl
python train.py --controller sac --device cuda:1 --output outputs/train_sac
```

The last two commands can be started in separate terminals on a two-GPU
machine. Each formal run uses 16 CPU environment workers plus development
probes; reduce simultaneous workloads on smaller machines. CPU training is
supported through `--device cpu` but is slower. No fixed training duration is
promised. Hyperparameters and checkpoint selection are documented in
[the protocol](docs/PROTOCOL.md). `--weight-ratio 0.5` and `--weight-ratio 2`
select the archived reward-weight variants.

The short `--smoke` recipe uses 1024 interactions and four 256-step development
probes. It deliberately changes the budget and probe duration, not the model,
action permissions or learning objective. Formal training retains the recorded
development windows and common recipe. Output directories must be new.

## Traditional Baseline Tuning

```bash
python tune_baselines.py --controller pi --smoke --output outputs/pi_check
python tune_baselines.py --controller pi --workers 4 --output outputs/pi_grid
python tune_baselines.py --controller mpc --workers 4 --output outputs/mpc_grid
python tune_baselines.py --controller gspid --workers 4 --output outputs/gspid_grid
```

The full grids contain 24 PI, 9 MPC and 54 GS-PID settings. Every setting is
evaluated in the same 12 development windows. The minimum admissible common
cost selects the configuration. Final evaluation is not used for tuning.
`selected.json` records a new selection; it never silently changes the shipped
frozen jobs. See `docs/REPRODUCIBILITY.md` for applying a new selection.

## Supplementary Evaluations

```bash
python reproduce.py --suite stress --limit 1
python reproduce.py --suite weights --limit 1
python reproduce.py --suite sensor --limit 1
```

- `stress`: 840 frozen-parameter jobs, including the common reference and six
  engineering sensitivity scenarios; no controller retuning.
- `weights`: 320 jobs for the four additional trained policies at power/cooling
  weight ratios 0.5 and 2.
- `sensor`: 984 archived **cross-cadence diagnostic** jobs, including the two
  historical pressure-offset settings. These change pressure acquisition to
  60 s while control remains at 10 s. They are not the normal 10-s control
  experiment. The 0-offset subset contains 792 jobs. The offset settings are
  archived diagnostics, not a recommended conversion for absolute pressure.

The five original controllers (without the later GS-PID addition) appear in
the stress and sensor suites. The code does not remove unfavorable outcomes.
The sensor diagnostics do not establish deployment reliability under outages.

## Repository Layout

```text
source_coupled_revision/    Simulator, interfaces, controllers and learning
experiments/               Frozen model parameters, paired learner assets,
                           trained policies and required training subclasses
configs/                   Demo and complete evaluation job manifests
data/synthetic/            Generated boundary-like example and metadata
results/                   Aggregated simulation metrics and learning histories
docs/                      Protocol, data dictionary and reproducibility notes
tests/                     Causality, action parity, objective and balance tests
tools/                     Public-file and checkpoint audit
reproduce.py               Frozen-policy evaluation
train.py                   Matched RRL/SAC training
tune_baselines.py          Development-only feedback/MPC search
generate_synthetic_data.py Synthetic data generator
plot_results.py            Matched trajectory plots
```

Some asset directories retain their experiment identifiers so that archived
plans and model checks resolve consistently. Only required simulation and
learning modules are included; manuscript editing tools, private documents,
credentials, caches and restricted raw data are excluded.

## Tests and Integrity

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/audit_public_release.py
```

`RELEASE_MANIFEST.json` records SHA-256 checksums. Public export replaces local
absolute paths with repository-relative paths and updates checkpoint protocol
metadata accordingly. **Learned numerical tensors remain unchanged.**
`PUBLIC_EXPORT.json` records original/public artifact hashes and tensor checks.
A checksum detects accidental changes; it does not authenticate a third-party
download. Load checkpoints only from trusted sources.

## Data Availability and Scope

All distributed trajectories are simulation outputs; other data are aggregate diagnostic/performance
statistics. Field CSV/Excel files, project reports and original measurement
logs are not distributed. Model coefficients and learned parameters are derived
research artifacts; they are distinguished from raw observations. Training the
pressure model on the original field labels or repeating raw-data calibration
requires the separately controlled source records. Model-based control
experiments and synthetic generation run without those records.

The simulator is a literature-supported, field-informed control-oriented
reference model. It is not a full axial reproduction of another team's heat
pipe solver. Physical balance checks, field statistical comparisons and
controller performance are separate forms of evidence. See
[data scope and units](docs/DATA.md).

## Citation

If you use this work, please cite the accompanying manuscript:

> Liu, D., Gao, T., Zhou, L., Liu, Z., Liu, L., Wang, X., and Cao, H.
> ThermoResidual-RL: physics-guided residual reinforcement learning for
> geothermal power--cooling control under model mismatch.

Machine-readable software citation metadata are in `CITATION.cff`. No journal
acceptance, publication DOI or permanent archive DOI is implied. A release DOI
can be added after a real archival record has been created.

## License and Contributions

Project source code, documentation and included project-generated model assets
are distributed under the [MIT License](LICENSE). Third-party packages are
installed separately under their own licenses; see [notices](THIRD_PARTY_NOTICES.md).
No license is granted to excluded project records or third-party publications.

Use GitHub issues for reproducible bug reports and feature proposals. Include
the command, environment and simulated example rather than private records.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the development checks.
