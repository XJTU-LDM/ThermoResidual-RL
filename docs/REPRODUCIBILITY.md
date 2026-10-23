# Reproducing and Extending the Release

## Three Levels

1. **Execution check:** verify checksums, run tests, six short inference demos
   and `train.py --smoke`. Short-run metrics are not the full study result.
2. **Frozen-policy reproduction:** use `--suite main` and supplementary suites.
   These preserve trained weights, physical configurations and final matrices.
3. **Fresh training:** use `train.py` separately for RRL and SAC with the same
   chosen weight ratio. Retain all training results and select checkpoints by
   the recorded common development objective.

Floating-point implementation, hardware and BLAS/CUDA kernels can introduce
small differences. Timing is not a bitwise reproducibility target. Frozen
checkpoint tensors are preserved by public export; relative paths and protocol
hash metadata are changed explicitly in `PUBLIC_EXPORT.json`.

## Release Export

The public tree is an allowlisted export, not a copy of the author's workspace
or Git history. Required simulator/learning modules, coefficients, policies,
initial states and numeric simulation results are retained. Manuscript editing
scripts, old optimization launchers, reports, caches and raw records are omitted.
The private single-point operating tuple formerly used to derive electrical
efficiency and conductance is also excluded: the public implementation contains
only the already-fitted effective coefficients, with exactly the same values.

Public per-run protocol JSON has local paths replaced with repository-relative
paths. Evaluation checkpoint metadata is rebound to the public protocol digest.
All actor, critic, target, entropy and normalization tensors are checked for
exact equality before and after export. `checkpoint_sha256` in public jobs
refers to the newly serialized public checkpoint, not the private archive.

Directory names are retained for import/path compatibility. They do not mean
that all historical development stages are necessary to run a frozen policy.
`run_four_conditions.py` contains only the reused target-generation functions,
not the old boundary-replay environment.

## Applying New Baseline Settings

`tune_baselines.py` writes a full grid and `selected.json`. It deliberately does
not edit shipped jobs. For a new experiment, load a matching public job, replace
only `overrides` with the selected settings, and use a new name/output directory:

```python
from copy import deepcopy
from pathlib import Path
import json
from reproduce import load_jobs
from source_coupled_revision.revision_evaluation import evaluate

job = deepcopy(next(j for j in load_jobs('main') if j['mode'] == 'pi'))
selected = json.loads(Path('outputs/pi_grid/selected.json').read_text())
job['overrides'] = selected['overrides']
job['name'] += '_new_development_selection'
job['evaluation_out'] = 'outputs/new_pi_evaluation'
result = evaluate(job)
```

Run the snippet from the repository root and retain the selected configuration
and the final evaluation separately. Never choose settings from the final
evaluation metrics.

## Evaluating a Newly Trained Policy

`outputs/train_rrl/training/rrl_20260930/result.json` records the selected
checkpoint. A corresponding new frozen job must use the new training plan,
checkpoint path and checksum, not only swap network weights into an old job:

```python
from copy import deepcopy
from pathlib import Path
import json
from reproduce import load_jobs
from source_coupled_revision.revision_evaluation import evaluate

run = Path('outputs/train_rrl')
trained = json.loads((run / 'training/rrl_20260930/result.json').read_text())
job = deepcopy(next(j for j in load_jobs('main') if j['mode'] == 'rrl'))
job['plan'] = json.loads((run / 'protocol.json').read_text())
job['checkpoint'] = trained['checkpoint']
job['checkpoint_sha256'] = trained['checkpoint_sha256']
job['name'] += '_new_training'
job['evaluation_out'] = 'outputs/new_rrl_evaluation'
result = evaluate(job)
```

A smoke-trained plan is a software test, not a replacement for the full-budget
policy. Preserve the train/test protocol and describe changed budgets explicitly.

## Statistical Unit

All observations within a trajectory share plant dynamics. Do not treat time
points as independent replicates. For frozen-policy comparisons use paired
process seeds and preserve each profile/condition block. These repeats estimate
process variability conditional on a trained policy, not training variability.
The source tables preserve reported outcomes without replacing a result because
another method performs better.

## Common Problems

- **Checksum failure:** extract a fresh release, or identify and document the
  intended modification before creating a new release manifest.
- **PyTorch/CUDA import issue:** use a Python 3.11 environment with the declared
  dependencies; run CPU inference first.
- **Missing restricted field file:** use the public entrypoints. Fitting original
  pressure labels and recalibrating field records are distinct tasks and need
  separately authorized data.
- **Output already exists:** training/tuning never overwrite runs; choose a new
  directory. Inference resumes only matching completed jobs.
- **No admissible checkpoint/candidate:** retain the failure and inspect balance,
  rate, supply and optimization records; do not silently exclude the run.
- **Too many processes:** limit `--workers`; avoid running both full training
  commands on a machine without enough CPU/memory resources.

## GitHub Upload

Create an empty repository, then upload the **contents** of this directory,
including hidden `.github/` and `.gitignore`, not just the ZIP file. Keep the
included MIT license rather than generating a conflicting second license.
Use a clean Git history so private workspace data never enter any commit.
Tag the tested public state `v1.0.0` and attach the code ZIP to that release if
desired. Replace the code-availability placeholder only after the public URL
actually exists. This package creation does not itself publish to GitHub.
