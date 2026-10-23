"""Conservative lumped prototype, not a Huang-model or field calibration.

Ammonia properties use CoolProp HEOS. No historical boundary file is read.
The receiver is zero-holdup with pumped instantaneous mass return. The cooling
cycle remains a cycle-averaged performance surrogate, not a two-bed ODE model.
"""
from dataclasses import asdict, dataclass, fields
import math

import numpy as np
from CoolProp.CoolProp import PropsSI

from .contracts import (
    Measurement, MeasurementPacket, SourceAdvance, SourceBoundary, SourceModel,
    SourceState, finite,
)


@dataclass(frozen=True)
class GreyboxConfig:
    # These defaults are numerical development assumptions, not site estimates.
    near_capacity_kJ_K: float = 8e6
    far_capacity_kJ_K: float = 6e7
    near_far_conductance_kW_K: float = 12.0
    far_deep_conductance_kW_K: float = 5.0
    wall_conductance_kW_K: float = 20.0
    deep_temperature_K: float = 390.0
    near_initial_K: float = 365.0
    far_initial_K: float = 380.0
    ammonia_mass_kg: float = 1000.0
    ammonia_volume_m3: float = 10.0
    ammonia_initial_K: float = 330.0
    sink_capacity_kJ_K: float = 100000.0
    sink_initial_K: float = 298.15
    ambient_K: float = 288.15
    sink_conductance_kW_K: float = 60.0
    condenser_approach_K: float = 5.0
    pump_efficiency: float = 0.7
    flow_conductance_kg_s_sqrt_MPa: float = 0.7
    flow_max_kg_s: float = 0.9
    command_rate_kg_s2: float = 0.006
    turbine_efficiency_max: float = 0.85
    turbine_design_flow_kg_s: float = 1.088
    turbine_offdesign_coefficient: float = 0.7
    power_tau_s: float = 35.0
    cooling_tau_s: float = 55.0
    cooling_cycle_period_s: float = 480.0
    cooling_switch_window_s: float = 30.0
    cooling_cycle_amplitude: float = 0.12
    cooling_switch_loss: float = 0.30
    cooling_COP: float = 0.075
    max_substep_s: float = 1.0
    control_step_s: float = 10.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            finite(value)
            if value <= 0:
                raise ValueError(f"Positive {name} required")
        if not 0 < self.pump_efficiency <= 1 or not 0 < self.turbine_efficiency_max <= 1:
            raise ValueError("Invalid efficiency")
        if self.cooling_switch_window_s >= self.cooling_cycle_period_s / 2:
            raise ValueError("Switch window overlaps consecutive switches")
        if self.cooling_cycle_amplitude + self.cooling_switch_loss >= 1:
            raise ValueError("Cycle factor must remain positive")

    @classmethod
    def from_manifest(cls, manifest):
        values = manifest["parameters"]
        expected = {f.name for f in fields(cls)}
        if set(values) != expected:
            raise ValueError(f"Explicit parameter set required: {set(values) ^ expected}")
        if set(manifest["parameter_sources"]) != expected:
            raise ValueError("Every parameter needs provenance")
        if any(not v for v in manifest["parameter_sources"].values()):
            raise ValueError("Empty parameter provenance")
        return cls(**values)


@dataclass(frozen=True)
class AmmoniaState:
    temperature_K: float
    pressure_MPa: float
    bulk_quality: float
    vapor_enthalpy_kJ_kg: float
    vapor_entropy_J_kg_K: float


def ammonia_state(mass_kg, energy_kJ, volume_m3):
    finite(mass_kg, energy_kJ, volume_m3)
    if mass_kg <= 0 or volume_m3 <= 0:
        raise ValueError("Invalid ammonia inventory")
    values = PropsSI(["T", "P", "Q"], "D", mass_kg / volume_m3,
                     "U", energy_kJ * 1000 / mass_kg, "Ammonia")
    temperature, pressure, quality = map(float, values)
    if not 0 < quality < 1:
        raise ValueError("Outside two-phase inventory domain; no property clipping allowed")
    h, s = PropsSI(["H", "S"], "P", pressure, "Q", 1, "Ammonia")
    return AmmoniaState(temperature, pressure / 1e6, quality, float(h) / 1000, float(s))


class LumpedSource(SourceModel):
    """Two rock nodes and fixed-volume ammonia, with explicit return enthalpy.

    Equal outflow and pumped return imply constant total mass. Outlet quality=1
    is a separator assumption, not the quality of the bulk two-phase inventory.
    """
    def __init__(self, config: GreyboxConfig):
        self.cfg = config

    def initial_state(self):
        c = self.cfg
        u = PropsSI("U", "T", c.ammonia_initial_K, "D",
                    c.ammonia_mass_kg / c.ammonia_volume_m3, "Ammonia") / 1000
        energy = c.ammonia_mass_kg * u
        p = ammonia_state(c.ammonia_mass_kg, energy, c.ammonia_volume_m3).pressure_MPa
        return SourceState(0.0, (c.near_initial_K, c.far_initial_K),
                           c.ammonia_mass_kg, energy, p)

    def rates(self, rock, mass, energy, request, boundary):
        c = self.cfg
        a = ammonia_state(mass, energy, c.ammonia_volume_m3)
        request = np.asarray(request, dtype=float)
        if request.shape != (2,) or not np.isfinite(request).all() or (request < 0).any():
            raise ValueError("Two nonnegative finite flow requests required")
        finite(boundary.return_enthalpy_kJ_kg, boundary.surface_backpressure_MPa)
        if boundary.surface_backpressure_MPa <= 0:
            raise ValueError("Backpressure must be absolute and positive")
        supply = c.flow_conductance_kg_s_sqrt_MPa * math.sqrt(
            max(a.pressure_MPa - boundary.surface_backpressure_MPa, 0))
        total = float(request.sum())
        actual = request * min(1.0, supply / total) if total else request.copy()
        near, far = rock
        qdeep = c.far_deep_conductance_kW_K * (c.deep_temperature_K - far)
        qfar = c.near_far_conductance_kW_K * (far - near)
        qgeo = c.wall_conductance_kW_K * (near - a.temperature_K)
        extraction = float(actual.sum()) * (a.vapor_enthalpy_kJ_kg - boundary.return_enthalpy_kJ_kg)
        derivative = np.array([(qfar - qgeo) / c.near_capacity_kJ_K,
                               (qdeep - qfar) / c.far_capacity_kJ_K, qgeo - extraction])
        return derivative, actual, a, qgeo, qdeep, extraction

    def advance(self, state, requested_flows_kg_s, boundary, dt_s):
        finite(dt_s)
        if dt_s <= 0:
            raise ValueError("Positive timestep required")
        y = np.array([*state.rock_temperature_K, state.fluid_internal_energy_kJ], float)
        start = y.copy()
        elapsed, heat, extracted, actual_sum, hsum = 0.0, 0.0, 0.0, np.zeros(2), 0.0
        while elapsed < dt_s - 1e-12:
            h = min(self.cfg.max_substep_s, dt_s - elapsed)
            d0, *_ = self.rates(y[:2], state.fluid_mass_kg, y[2], requested_flows_kg_s, boundary)
            mid = y + .5 * h * d0
            d, flow, a, q, _, ext = self.rates(mid[:2], state.fluid_mass_kg, mid[2], requested_flows_kg_s, boundary)
            y += h * d
            heat += h * q
            extracted += h * ext
            actual_sum += h * flow
            hsum += h * a.vapor_enthalpy_kJ_kg
            elapsed += h
        a = ammonia_state(state.fluid_mass_kg, y[2], self.cfg.ammonia_volume_m3)
        result = SourceState(state.time_s + dt_s, tuple(y[:2]), state.fluid_mass_kg, float(y[2]), a.pressure_MPa)
        return SourceAdvance(result, tuple(actual_sum / dt_s), heat / dt_s, hsum / dt_s,
                             0.0, float(y[2] - start[2] - heat + extracted))


@dataclass(frozen=True)
class PlantState:
    source: SourceState
    sink_temperature_K: float
    power_kW: float
    cooling_kW: float
    command_kg_s: tuple[float, float]
    actual_flow_kg_s: tuple[float, float]
    last_sink_input_kW: float = 0.0


class CoupledPlant:
    """RK2 integration with shared fluxes in all component energy accounts."""
    def __init__(self, config: GreyboxConfig):
        self.cfg = config
        self.source_model = LumpedSource(config)
        self.state = PlantState(self.source_model.initial_state(), config.sink_initial_K,
                                0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

    def _pack(self, state):
        return np.array([*state.source.rock_temperature_K, state.source.fluid_internal_energy_kJ,
                         state.sink_temperature_K, state.power_kW, state.cooling_kW], float)

    def _storage(self, y):
        c = self.cfg
        return (c.near_capacity_kJ_K * y[0] + c.far_capacity_kJ_K * y[1] + y[2]
                + c.sink_capacity_kJ_K * y[3] + c.power_tau_s * y[4])

    def _rates(self, y, t, command, ambient):
        c = self.cfg
        mass = self.state.source.fluid_mass_kg
        a = ammonia_state(mass, y[2], c.ammonia_volume_m3)
        tc = y[3] + c.condenser_approach_K
        pc, hl, rho = map(float, PropsSI(["P", "H", "D"], "T", tc, "Q", 0, "Ammonia"))
        hl /= 1000
        wpump = max(a.pressure_MPa * 1e6 - pc, 0) / rho / c.pump_efficiency / 1000
        ds, flow, a, qgeo, qdeep, extraction = self.source_model.rates(
            y[:2], mass, y[2], command, SourceBoundary(hl + wpump, pc / 1e6))
        if flow[0] > 0:
            h2s = PropsSI("H", "P", pc, "S", a.vapor_entropy_J_kg_K, "Ammonia") / 1000
            eta = max(0.0, c.turbine_efficiency_max * (
                1 - c.turbine_offdesign_coefficient * ((flow[0] - c.turbine_design_flow_kg_s)
                                                       / c.turbine_design_flow_kg_s)**2))
            shaft = flow[0] * eta * max(a.vapor_enthalpy_kJ_kg - h2s, 0)
        else:
            shaft = 0.0
        drive = flow[1] * (a.vapor_enthalpy_kJ_kg - hl)
        power_condenser = flow[0] * (a.vapor_enthalpy_kJ_kg - hl) - shaft
        if drive < -1e-9 or power_condenser < -1e-9:
            raise ValueError("Unsupported reversed heat cycle")
        phase = t % c.cooling_cycle_period_s
        half = c.cooling_cycle_period_s / 2
        distance = min(phase, abs(phase - half), c.cooling_cycle_period_s - phase)
        switching = max(0.0, 1 - distance / (c.cooling_switch_window_s / 2))
        factor = 1 + c.cooling_cycle_amplitude * math.sin(2 * math.pi * phase / c.cooling_cycle_period_s)
        factor -= c.cooling_switch_loss * switching
        cooling_ss = c.cooling_COP * factor * drive
        # Cycle-averaged chiller rejection includes both driving heat and cooling load.
        sink_input = power_condenser + drive + y[5]
        ambient_rejection = c.sink_conductance_kW_K * (y[3] - ambient)
        derivative = np.array([*ds, (sink_input - ambient_rejection) / c.sink_capacity_kJ_K,
                               (shaft - y[4]) / c.power_tau_s, (cooling_ss - y[5]) / c.cooling_tau_s])
        diagnostics = dict(source_heat_kW=qgeo, deep_heat_kW=qdeep, extraction_kW=extraction,
                           drive_heat_kW=drive, turbine_shaft_kW=shaft, sink_input_kW=sink_input,
                           ambient_rejection_kW=ambient_rejection, pump_work_kW=float(flow.sum()) * wpump,
                           power_kW=float(y[4]), cooling_kW=float(y[5]),
                           flow_power_kg_s=float(flow[0]), flow_cooling_kg_s=float(flow[1]))
        return derivative, diagnostics

    def step(self, requested_kg_s, dt_s=None, *, ambient_K=None):
        c = self.cfg
        dt = c.control_step_s if dt_s is None else float(dt_s)
        ambient = c.ambient_K if ambient_K is None else float(ambient_K)
        finite(dt, ambient)
        if dt <= 0 or dt > c.control_step_s + 1e-9 or ambient <= 0:
            raise ValueError("Invalid control interval or unhandled gap")
        request = np.asarray(requested_kg_s, float)
        if request.shape != (2,) or not np.isfinite(request).all() or (request < 0).any():
            raise ValueError("Two nonnegative finite requests required")
        before = self.state
        previous = np.asarray(before.command_kg_s)
        command = np.clip(np.clip(request, 0, c.flow_max_kg_s),
                          previous - c.command_rate_kg_s2 * dt, previous + c.command_rate_kg_s2 * dt)
        y = self._pack(before)
        initial = y.copy()
        elapsed, totals = 0.0, {}
        while elapsed < dt - 1e-12:
            h = min(c.max_substep_s, dt - elapsed)
            # Resolve the triangular switching loss corners consistently in refinement tests.
            t = before.source.time_s + elapsed
            half = c.cooling_cycle_period_s / 2
            for offset in (0, c.cooling_switch_window_s / 2, half - c.cooling_switch_window_s / 2):
                next_corner = (math.floor((t - offset + 1e-9) / half) + 1) * half + offset
                if next_corner - t > 1e-9:
                    h = min(h, next_corner - t)
            d0, _ = self._rates(y, t, command, ambient)
            d, diag = self._rates(y + .5 * h * d0, t + .5 * h, command, ambient)
            y += h * d
            for key, value in diag.items():
                totals[key] = totals.get(key, 0.0) + h * value
            elapsed += h
        a = ammonia_state(before.source.fluid_mass_kg, y[2], c.ammonia_volume_m3)
        source = SourceState(before.source.time_s + dt, tuple(y[:2]), before.source.fluid_mass_kg,
                             float(y[2]), a.pressure_MPa)
        last_d, last_diag = self._rates(y, source.time_s, command, ambient)
        flow = (last_diag["flow_power_kg_s"], last_diag["flow_cooling_kg_s"])
        # Commit only after every property/domain check passes.
        self.state = PlantState(source, float(y[3]), float(y[4]), float(y[5]), tuple(command), flow,
                                last_diag["sink_input_kW"])
        balance = (totals["deep_heat_kW"] + totals["pump_work_kW"] + totals["cooling_kW"]
                   - totals["power_kW"] - totals["ambient_rejection_kW"])
        result = {key: value / dt for key, value in totals.items()}
        result.update(time_s=source.time_s, pressure_MPa=a.pressure_MPa,
                      bulk_quality=a.bulk_quality, source_temperature_K=a.temperature_K,
                      condenser_pressure_MPa=PropsSI("P", "T", y[3] + c.condenser_approach_K, "Q", 0, "Ammonia") / 1e6,
                      near_temperature_K=float(y[0]), far_temperature_K=float(y[1]),
                      source_balance_residual_kJ=float(y[2] - initial[2] - totals["source_heat_kW"] + totals["extraction_kW"]),
                      total_balance_residual_kJ=float(self._storage(y) - self._storage(initial) - balance),
                      mass_balance_residual_kg=0.0, energy_throughput_kJ=sum(abs(totals[k]) for k in (
                          "deep_heat_kW", "pump_work_kW", "cooling_kW", "power_kW", "ambient_rejection_kW")),
                      supply_shortfall_kg_s=max(0.0, float(command.sum()) - sum(flow)),
                      command_power_kg_s=float(command[0]), command_cooling_kg_s=float(command[1]))
        return result

    def measurement_packet(self, episode_id):
        """End-of-step measured signals only; no rock state or true cycle phase."""
        s, c = self.state, self.cfg
        pc = PropsSI("P", "T", s.sink_temperature_K + c.condenser_approach_K, "Q", 0, "Ammonia") / 1e6
        values = dict(power_kW=s.power_kW, cooling_kW=s.cooling_kW,
                      wellhead_pressure_MPa=s.source.outlet_pressure_MPa,
                      measured_exchange_kW=s.last_sink_input_kW,
                      flow_power_kg_s=s.actual_flow_kg_s[0], flow_cooling_kg_s=s.actual_flow_kg_s[1],
                      cooling_water_inlet_C=s.sink_temperature_K - 273.15,
                      condenser_pressure_MPa=pc)
        return MeasurementPacket(episode_id, tuple(Measurement(k, v, s.source.time_s, s.source.time_s)
                                                   for k, v in values.items()))
