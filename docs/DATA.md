# Data and Model Assets

| Location | Material | Origin |
|---|---|---|
| `thermoresidual/` | Simulator and controller code | Author implementation |
| `assets/model/` | Physical/process coefficients | Published context, reference assumptions and fitted summaries |
| `assets/policies/` | Selected policies and empty initial states | Simulated learning and initialization |
| `assets/normalization.pth` | Input normalization | Simulated development observations |
| `assets/pressure/` | Five pressure-network parameter sets | Derived learned parameters |
| `configs/` | Frozen job definitions | Computational protocol |
| `data/synthetic/` | Example trajectory and metadata | Coupled-model simulation |
| `docs/figures/` | One method illustration | Author-provided algorithm diagram |

Original field CSV/Excel, raw pressure labels, project reports, account details,
private article copies and nonempty replay buffers are not distributed. The
repository has no raw-field loader and does not download private records.
Model weights and effective fitted coefficients are derived assets; they are
distinct from original measurements. The pressure architecture and inference are
included; original raw-label refitting requires separately authorized records.

`data/synthetic/synthetic_boundary.csv` labels every row `data_type=simulation`.
`metadata.json` gives the generation settings; `reference_trace.csv` contains the
simulated response and conservation diagnostics. A historical heat sequence is
not used to impose simulated output: each control action changes future states.

| Quantity | Unit |
|---|---|
| `time_s` | s, end of command interval |
| `pressure_MPa`, `condenser_pressure_MPa` | MPa absolute |
| `source_heat_kW`, `water_exchange_kW` | kW, distinct control-volume heat rates |
| `cooling_water_inlet_K`, `cooling_water_outlet_K` | K |
| `command_*_kg_s`, `actual_*_kg_s` | kg/s, requests and delivered flows |
| `output_*_kW`, `target_*_kW` | kW |
| `mass_residual_kg`, `energy_residual_kJ` | kg, kJ |
| `energy_throughput_kJ` | kJ, energy-closure denominator |
| `*_wall_s` | s, machine-dependent elapsed runtime |

The full trace can include hidden plant diagnostics for post-processing; their
presence in output does not make them controller inputs. `contracts.py` and
`control.py` define the measurement interface.

The audit checks the file inventory, text, checkpoint metadata, replay contents
and the allowlisted method image. Future contributions should be audited again
before publication. The MIT license does not grant rights to excluded records.
