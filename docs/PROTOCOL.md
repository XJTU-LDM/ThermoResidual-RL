# Computational Protocol

## Simulator and Information

`assets/model/` contains the frozen physical model, stochastic coefficients and
integration settings: 1.25 s maximum substep, 96 radial cells and eight axial
slices. `thermoresidual/site.py` defines same-well-informed geometry and effective
surface-loop coefficients. Other geometry and dynamics are declared reference
scenario assumptions, not independently identified site parameters.

Noise changes effective resistance, wall exchange and condenser exchange.
Simulated pressure and heat are not replaced by prescribed output sequences.
Common process seeds provide prefix-consistent paired disturbances while each
controller evolves its own closed-loop state. All pressures are in MPa absolute.

The 191-component observation comprises seven 26-component frames and nine
current/reference entries. Each frame contains eight channel values, eight
validity flags, eight ages, a pressure-estimated flag and a frame-present flag.
The eight channels are power, cooling, wellhead pressure, exchange heat, actual
power/cooling flows, water inlet temperature and condenser pressure. The final
nine entries contain current/next targets, previous commands, nominal allocation
and forecast standard deviation. Exact scaling is in `control.py`.

Packets carry sample/arrival times and validity. The observer rejects future
measurements and cross-episode data. Normal pressure acquisition and control
are both 10 s. The separate sensor diagnostic uses 60 s pressure acquisition,
hold/AR(7)/ensemble fallback, noise of 0.01/0.05 MPa and 300/900 s outages.
The five-member predictor uses seven 60 s pressure samples without absolute time.
It is separate from the physical pressure solver.

## Commands, Objective and Metrics

Each branch command lies in [0, 1.088] kg/s and changes by at most 0.06 kg/s
per 10 s step. Actual supply is determined by hydraulic/source dynamics.
RRL maps zero actor output to nominal allocation; its negative/positive endpoints
reach zero/maximum branch flow. SAC covers the same endpoints. Both pass through
the same box/rate projection, with no additional branch residual caps.

```text
cost = wW * ((power - target_power) / 10)^2
     + wQ * ((cooling - target_cooling) / 10)^2
     + 0.002 * sum((command - previous_command)^2)
reward = -cost
```

Weights are (1, 1) for the main comparison, (2/3, 4/3) and (4/3, 2/3) for
the weight study. After 200 warmup steps, combined RMSE pools the squared errors
of both channels and all retained samples before taking the square root. It is
not an arithmetic average of branch or trajectory RMSE. Logs also record common
cost, positive overshoot, command variation, supply shortfall, hard violations,
mass/energy residuals and elapsed runtime. Conservation uses the full trajectory.

Low-amplitude target ranges are 13–15 kW power and 14–18 kW cooling;
high-amplitude ranges are 12–16 and 12–20 kW. The four target families are
standard, medium-frequency, very-fast-frequency and mixed step–ramp.

## Learning and Selection

| Parameter | Common value |
|---|---|
| Network | Two 256-unit layers, leaky ReLU and layer normalization |
| Linear feedback skip | Eight shared input features |
| Actor / critic learning rates | 3e-6 / 3e-5 |
| Discount / target coefficient | 0.5 / 0.005 |
| Batch / replay capacity | 512 / 100000 |
| Fixed entropy coefficient | 1e-5 |
| Critic loss | Scaled Huber, delta 0.001 |
| Updates | Four critic updates per interaction; actor every fourth critic update |
| Gradient norm limit | 1 |
| Training | 768000 interactions; 16 environments; 1024-step episodes |
| Learning starts | 512 interactions |
| Checkpoint interval | 32768 interactions and final budget |

Paired initial learners have matching network tensors, empty replay and empty
optimizer states. Normalization derives from simulated Physics-only development
observations. Evaluation weights fold that affine transform into the first layer.
The common initializer and seeds are encoded in the distributed protocols.
There is no auxiliary model/teacher loss or online model-assisted refinement.

Select the minimum admissible mean common cost over three development windows
and four conditions. Windows have 1200 steps, offsets 200/800/1400 and process
seed 8102. Evaluation seeds/profiles do not select checkpoints. Fixed budget does
not itself establish convergence or training-initialization uncertainty.

## Baselines and Evaluation Matrices

Physics+PI uses the common allocator and constraints, with causal integral
updates and back-calculation from projected commands. Search base gains 0.003
and 0.00003 with proportional multipliers {0.5,1,2}, integral multipliers
{1,4,16,64} and anti-windup times {30,120} s: 24 configurations.

MPC uses the nominal model and shared one-step preview, holding later references.
Search horizons {6,12,18} and rate penalties {0.002,0.01,0.05}: nine configurations.
OSQP statuses and physics fallback are logged; hidden plant parameters and future
disturbances are not exposed to the controller.

GS-PID schedules inverse nominal gain with floor 10. Search proportional factors
{0.15,0.30,0.60}, integral times {60,120,240}, derivative times {0,5,20} and
anti-windup times {30,120} s: 54 configurations. All baseline searches use the
same development windows and common cost, then freeze gains for evaluation.

The main matrix has 480 trajectories (six controllers); stress has 840 (five
controllers); weights has 320 (four additional policies); the absolute-pressure
sensor-interface diagnostic has 792. Formal trajectories contain 3200 steps.
The process-realization axis measures disturbance variation of fixed policies.
