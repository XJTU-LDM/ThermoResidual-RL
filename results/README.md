# Archived Simulation Results

`main_five_controller_metrics.csv` contains the original 400 main simulation
trajectories (RRL, SAC, Physics+PI, MPC and Physics-only). `gspid/trajectory_metrics.csv`
contains the 80 subsequent GS-PID trajectories in the same frozen matrix. Together
they form the six-controller 480-trajectory main comparison. The main job
manifest includes all six controllers directly.
`main_six_controller_metrics.csv` combines those unchanged trajectory rows;
`main_six_controller_profile_summary.csv` pools their MSE with sample-count
weights before taking square roots.

`supplementary/` contains the archived numeric stress, weight and sensor
diagnostic results. The sensor tables include 984 historical jobs; the zero-offset
subset has 792. Sensor acquisitions are 60 s and are distinct from the normal
10-s main interface. Neither the historical offset variants nor the sensor
diagnostic counts should be conflated with the main matrix.

Learning histories and development checkpoint records remain alongside their
policy asset directories. Empty cells in history CSV indicate that a diagnostic
was not logged at that update; they are not missing reward or output samples.
Latency fields record the original hardware and may differ when rerun.

Hashes recorded in archived CSV tables identify the original serialized policy
artifacts. The public export changes protocol metadata only, so public policy
file hashes differ while learned tensor values are unchanged. The mapping is
in `PUBLIC_EXPORT.json`; current public jobs refer to public artifact hashes.

All trajectory results here are computational. No original field sample
sequences are included. Derived parameter and validation summaries are separate
from confidential raw field records.
