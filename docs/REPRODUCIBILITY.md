# Reproducibility

## What Can Be Run

The package supports frozen policy inference, all included evaluation suites,
training from empty paired learners, baseline tuning and simulated-data
generation without original field data. `README.md` gives runnable commands.
The pressure predictor's original label-fitting dataset is not distributed.

## Integrity and Execution

`RELEASE_MANIFEST.json` records file hashes, while `ASSETS_MANIFEST.json` lists
the six selected policies. Each job checks its checkpoint hash and each policy
checks its frozen protocol hash. The physical/process configuration hashes are
also validated. Output directories are separate from read-only released assets.

Evaluation creates pointwise CSV, per-job metrics/provenance, run status and an
aggregate metrics CSV. Failed jobs remain recorded. Re-running an identical
completed job verifies its trace hash before reuse. Training and tuning require
new output directories. CLI `--help` lists plotting and aggregation options.

Use `python -m pytest -q` for unit and short physical integration checks and
`python tools/audit_public_release.py` for release screening. Normal smoke runs
are intentionally short and do not replace the full evaluation budget. CPU/GPU
floating-point differences and solver/runtime variation can affect fresh training;
distributed policy tensors provide a fixed starting point for inference.

## Updating Assets

If intentionally changing source/configuration, save a new release and rebuild
its manifest with `python tools/audit_public_release.py --refresh-manifest`.
Changing a frozen model or policy additionally requires updating the associated
protocol and job hashes; simply disabling integrity checks is not equivalent.

## GitHub Publication

Create an empty GitHub repository, place the contents of this folder at its
root, and commit those files. Do not copy the surrounding research workspace.
The `.gitignore` excludes generated outputs, environment files and caches.
Add the actual repository URL to citation and manuscript availability text after
publication; no public URL or DOI is invented in this package. A GitHub source
release and an archival DOI can then identify the exact distributed files.
