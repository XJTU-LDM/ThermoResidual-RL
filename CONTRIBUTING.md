# Contributing

Please open an issue describing the problem and a small simulation-based
reproducer before substantial changes. Include Python/package versions and the
command used. Do not attach private measurements, project documents, credentials
or third-party full texts.

Run `python -m pytest -q` and `python tools/audit_public_release.py` before
proposing a change. Numerical changes should include a physical balance check,
action-interface test and a paired simulation example. Keep evaluation settings
and checkpoint selection separate. Report failed evaluations as well as successes.

For intentional source/configuration changes, `reproduce.py` checksum validation
will fail until the release manifest is regenerated. Do this only for a new
clearly identified release, preserving the provenance of published results.
Do not commit generated `outputs/`, caches or private datasets.

Contributions must be your own work or compatible with this repository's MIT
license. Third-party attribution and license notices must be preserved.
