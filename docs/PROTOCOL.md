# Frozen Computational Protocol

## Plant and Signals

The stochastic reference model couples radial formation heat storage, ammonia
inventory and internal energy, hydraulic delivery, power production, adsorption
cooling and condenser return. The public physical and stochastic configurations
are under `experiments/apen17549_source_coupled_revision/` in
`model_foundation_v2/frozen_model.json` and `stochastic_model_final/`.
The accepted numerical discretization is read from `release_manifest.json`:
1.25 s maximum substep, 96 radial cells and 8 axial slices.

Process perturbations act on effective transport and heat-transfer parameters,
not by overwriting simulated pressure or heat outputs. The same process seed
generates a prefix-consistent common random stream. Each controller nevertheless
has its own physical state and measured response.

Reference-model assumptions and physical parameter provenance are implemented
in `source_coupled_revision/caofeidian.py`, `formation.py`, `coupled.py`,
`foundation.py` and `stochastic.py`. The same-well published studies support
selected site quantities; unmeasured geometry and dynamic quantities are explicit
scenario parameters. No third-party model code or publication PDF is bundled.

## Causal Observation

The observation has **191 components**: seven history frames of 26 components,
plus 9 common current/reference components. Each frame includes eight channel
values, eight measurement-valid flags, eight ages, a pressure-estimated flag and
a frame-present flag. Zero-padding represents unavailable initial history.

Eight measured channels:

1. Power output, kW.
2. Cooling output, kW.
3. Wellhead pressure, MPa (absolute).
4. Measured exchange heat rate, kW.
5. Actual power-branch flow, kg/s.
6. Actual cooling-branch flow, kg/s.
7. Cooling-water inlet temperature, degrees C.
8. Condenser pressure, MPa (absolute).

The final nine entries are current targets (2), next-step target preview (2),
previous commands (2), nominal allocation (2), and forecast standard deviation
(1). `ControlInterface.receive` defines the exact normalization. Measurements
have sample time, availability time and validity. The observer rejects future
arrivals, cross-episode packets and unresolved acquisition gaps. Plant truth is
not a fallback for missing pressure.

Historical measured heat is not replayed into the policy. Each simulated
controller receives heat measured from its own preceding closed-loop response.
Normal control pressure is sampled at the 10 s control cadence. The pressure
forecast network is separate from the physical plant solver.

## Action Composition and Common Constraints

Each branch command is in [0, 1.088] kg/s with a maximum change of 0.06 kg/s
per 10 s decision step. This is a per-branch simulated command envelope, not a
claim that the well can always supply the sum of both maximum requests. Actual
delivery is determined by the coupled source and hydraulic model; supply
shortfall is recorded.

RRL maps a normalized action of zero to nominal allocation. The negative and
positive endpoints reach zero and the branch maximum respectively. SAC maps
its normalized full action over the same range. Both then use the same hard
box/rate projection. There are **no extra power/cooling residual caps** in the
frozen main protocol. No obsolete superheat reward or thermal soft projection
is used in this comparison.

## Objective and Metrics

The common per-step cost is

```text
cost = wW * ((power - target_power) / 10)^2
     + wQ * ((cooling - target_cooling) / 10)^2
     + 0.002 * sum((command - previous_command)^2)
reward = -cost
```

Flow increments are in kg/s. The nominal main weights are (1, 1).
The additional weight pairs are (2/3, 4/3) and (4/3, 2/3), preserving a total
weight of two. The frozen job plans are authoritative.

After 200 warmup steps, power and cooling RMSE are computed separately.
Combined RMSE is `sqrt(mean([power_error^2, cooling_error^2]))` over retained
time points and both channels. Pooling trajectories requires averaging their
MSE with sample-count weighting before taking the square root; do not average
RMSE and call it pooled RMSE. Also report mean common cost, positive error
maxima, command variation, supply shortfall, violations, conservation residuals
and runtime. Conservation diagnostics include the full trajectory.

## Matched Learning

| Setting | Value |
|---|---|
| Backbone | Two 256-unit layers, leaky ReLU, layer normalization |
| Feedback skip | Shared trainable eight-feature linear path |
| Actor / critic learning rates | 0.000003 / 0.00003 |
| Discount / target Polyak coefficient | 0.5 / 0.005 |
| Batch / replay capacity | 512 / 100000 |
| Entropy coefficient | Fixed 0.00001 |
| Critic loss | Scaled Huber, delta = 0.001 |
| Critic updates | Four per interaction after replay warmup |
| Actor delay | One actor step per four critic updates |
| Gradient norm limit | 1.0 |
| Formal interactions | 768000 per run |
| Environments / episode length | 16 / 1024 steps |
| Replay learning starts | 512 interactions |
| Development checkpoints | Every 32768 interactions and at the final budget |
| Training initialization seed | 20260930 |

The paired initial learners have identical network tensors, empty replay and
empty optimizer states. Their common normalization is derived from simulated
Physics-only development-domain observations. Frozen evaluation checkpoints
fold the input affine transform into the first layer. Public export changes
only protocol metadata, not network tensors.

The minimum admissible mean cost over three development windows x four
conditions selects a checkpoint. Each window has 1200 steps and process seed
8102; offsets are 200, 800 and 1400 steps. Frozen final-evaluation seeds and
profiles are not used by the selection routine. These simulation settings have
a development history and are not described as a newly acquired blind dataset.

The reference actor is optimized with standard SAC losses; the main recipe has
no auxiliary teacher/model loss and no model-assisted online action refinement.

## Traditional Baselines

PI uses the same nominal allocator and branch command envelope. Its integral
state updates causally, with back-calculation using actual projected commands;
gains are fixed during evaluation. The search multiplies base proportional
gain 0.003 by {0.5,1,2}, base integral gain 0.00003 by {1,4,16,64}, and uses
anti-windup time constants {30,120} s.

MPC uses the controller's nominal model, causal pressure, current targets and
shared one-step preview; it does not read hidden plant parameters or the future
stochastic stream. Later preview entries are held consistently with the
implemented MPC horizon. Search horizons {6,12,18} and rate penalties
{0.002,0.01,0.05}. OSQP status and physics fallback are logged.

GS-PID schedules feedback using the inverse nominal output coefficient with a
fixed gain floor. Search proportional factors {0.15,0.30,0.60}, integral times
{60,120,240} s, derivative times {0,5,20} s, anti-windup times {30,120} s,
and gain floor 10. All grids use the same development windows and objective.

## Final and Supplementary Matrices

- Main: 480 trajectories, including GS-PID; 4 target families x 4 conditions x
  5 process realizations x 6 controllers. Each is 3200 steps, warmup 200.
- Stress: 840 trajectories; 7 settings x 2 profiles x 4 conditions x 3 process
  realizations x 5 controllers. Settings cover resistance x0.75/x1.25,
  cooling time x1.5, cycle lengths 300/600 s, and combined slower return plus
  a midpoint water-side conductance reduction, as well as the reference.
- Weights: 320 trajectories for four additional frozen policies; the common
  weight main policies remain in the main suite.
- Archived sensor diagnostics: 984 tasks, of which 792 use zero pressure offset.
  Acquisition is 60 s, control 10 s; noise/outages, hold/AR(7)/ensemble fallback
  and historical offset variants are explicitly encoded in each task.

Process-realization comparisons describe variation of fixed trained policies.
They do not measure variability across independent training initializations.
Metric tables and archived comparisons retain all reported outcomes.
