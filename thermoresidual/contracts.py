"""Timestamped measurement, observation and conservative source interfaces."""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


CHANNELS = (
    "power_kW", "cooling_kW", "wellhead_pressure_MPa",
    "measured_exchange_kW", "flow_power_kg_s", "flow_cooling_kg_s",
    "cooling_water_inlet_C", "condenser_pressure_MPa",
)


def finite(*values):
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Non-finite value")


@dataclass(frozen=True)
class Measurement:
    channel: str
    value: float | None
    sampled_at_s: float
    available_at_s: float
    valid: bool = True

    def __post_init__(self):
        finite(self.sampled_at_s, self.available_at_s)
        if self.channel not in CHANNELS:
            raise ValueError("Unapproved observation channel")
        if self.available_at_s < self.sampled_at_s:
            raise ValueError("A measurement cannot arrive before it is sampled")
        if self.valid:
            if self.value is None:
                raise ValueError("A valid measurement requires a value")
            finite(self.value)
        elif self.value is not None:
            raise ValueError("Missing/invalid readings must not expose a hidden value")


@dataclass(frozen=True)
class MeasurementPacket:
    episode_id: str
    readings: tuple[Measurement, ...]

    def __post_init__(self):
        if not self.episode_id or not isinstance(self.readings, tuple):
            raise ValueError("An episode and immutable readings are required")
        if not all(isinstance(m, Measurement) for m in self.readings):
            raise TypeError("Only Measurement objects can enter a packet")
        if len({m.channel for m in self.readings}) != len(self.readings):
            raise ValueError("Duplicate measurement channel")


@dataclass(frozen=True)
class Observation:
    decision_at_s: float
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    ages_s: tuple[float, ...]
    pressure_estimated: bool = False


class PressurePredictor(Protocol):
    def predict(self, history: tuple[Observation, ...], current: Observation) -> float:
        """Use only causal observations; return an absolute pressure in MPa."""


class Observer:
    """Measured-pressure-first history; fail closed when packets are noncausal.

    Sensor ages/validity are explicit. Delayed arrivals do not replace fresher
    measurements. A missing pressure is not silently replaced with plant truth.
    """

    def __init__(self, episode_id: str, *, max_age_s: float,
                 max_step_s: float, predictor: PressurePredictor | None = None):
        finite(max_age_s, max_step_s)
        if not episode_id or max_age_s <= 0 or max_step_s <= 0:
            raise ValueError("Episode, maximum age and maximum step are required")
        self.episode_id = episode_id
        self.max_age_s = max_age_s
        self.max_step_s = max_step_s
        self.predictor = predictor
        self._latest: dict[str, Measurement] = {}
        self._history: deque[Observation] = deque(maxlen=7)

    @property
    def history(self) -> tuple[Observation, ...]:
        return tuple(self._history)

    def update(self, packet: MeasurementPacket, decision_at_s: float) -> Observation:
        finite(decision_at_s)
        if packet.episode_id != self.episode_id:
            raise ValueError("Cross-episode measurement; initialize a new observer")
        if self._history:
            step = decision_at_s - self._history[-1].decision_at_s
            if not 0 < step <= self.max_step_s:
                raise ValueError("Nonmonotone decision time or unhandled acquisition gap")
        latest = dict(self._latest)
        for reading in packet.readings:
            if reading.available_at_s > decision_at_s:
                raise ValueError("Future/unavailable measurement presented to controller")
            previous = latest.get(reading.channel)
            if previous is None or reading.sampled_at_s > previous.sampled_at_s:
                latest[reading.channel] = reading
            elif reading.sampled_at_s == previous.sampled_at_s and reading != previous:
                raise ValueError("Conflicting readings at the same sample timestamp")
        values, valid, ages = [], [], []
        for channel in CHANNELS:
            reading = latest.get(channel)
            age = decision_at_s - reading.sampled_at_s if reading else self.max_age_s
            usable = reading is not None and reading.valid and age <= self.max_age_s
            values.append(float(reading.value) if usable else 0.0)
            valid.append(bool(usable))
            ages.append(min(age, self.max_age_s))
        current = Observation(decision_at_s, tuple(values), tuple(valid), tuple(ages))
        pressure_index = CHANNELS.index("wellhead_pressure_MPa")
        if not valid[pressure_index] and self.predictor is not None:
            estimate = self.predictor.predict(self.history, current)
            finite(estimate)
            if estimate <= 0:
                raise ValueError("Pressure estimate must be positive and absolute")
            values[pressure_index] = float(estimate)
            # 'valid' remains false: an estimate must not become a measured label.
            current = Observation(decision_at_s, tuple(values), tuple(valid), tuple(ages), True)
        self._latest = latest
        self._history.append(current)
        return current

    def policy_history(self) -> np.ndarray:
        """Oldest first: current plus six previous frames, zero-padded with mask.

        Per frame: values, measurement-valid flags, ages, pressure-estimated flag,
        frame-present flag. Absolute timestamps and hidden state are not included.
        This is an unnormalized interface, not a checkpoint-compatible observation.
        """
        width = 3 * len(CHANNELS) + 2
        frames = np.zeros((7, width), dtype=np.float64)
        for i, observation in enumerate(self.history, start=7-len(self._history)):
            frames[i] = (*observation.values, *observation.valid, *observation.ages_s,
                         observation.pressure_estimated, True)
        return frames


@dataclass(frozen=True)
class SourceState:
    """Private solver state, never an observation or a controller argument."""
    time_s: float
    rock_temperature_K: tuple[float, ...]
    fluid_mass_kg: float
    fluid_internal_energy_kJ: float
    outlet_pressure_MPa: float


@dataclass(frozen=True)
class SourceBoundary:
    """Candidate physical interface, pending verification against site topology."""
    return_enthalpy_kJ_kg: float
    surface_backpressure_MPa: float


@dataclass(frozen=True)
class SourceAdvance:
    state: SourceState
    actual_branch_flows_kg_s: tuple[float, float]
    source_heat_kW: float
    outlet_enthalpy_kJ_kg: float
    mass_balance_residual_kg: float
    energy_balance_residual_kJ: float


class SourceModel(ABC):
    @abstractmethod
    def advance(self, state: SourceState, requested_flows_kg_s: tuple[float, float],
                boundary: SourceBoundary, dt_s: float) -> SourceAdvance:
        """Advance verified coupled equations; never read the historical heat trace."""
        raise NotImplementedError


def reward_components(outputs_kW, targets_kW, command_kg_s, previous_command_kg_s,
                      *, power_weight=1.0, cooling_weight=1.0):
    """Common cost in physical flow units; deliberately no 2 C margin input."""
    arrays = [np.asarray(a, dtype=np.float64) for a in
              (outputs_kW, targets_kW, command_kg_s, previous_command_kg_s)]
    if arrays[0].ndim < 1 or arrays[0].shape[-1] != 2:
        raise ValueError("Expected (..., 2) power/cooling pairs")
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("All pairs must have identical shape and finite values")
    finite(power_weight, cooling_weight)
    if power_weight <= 0 or cooling_weight <= 0:
        raise ValueError("Tracking weights must be positive")
    error = (arrays[0] - arrays[1]) / 10.0
    power = power_weight * error[..., 0]**2
    cooling = cooling_weight * error[..., 1]**2
    change = 0.002 * np.sum((arrays[2] - arrays[3])**2, axis=-1)
    total = power + cooling + change
    return {"power_cost": power, "cooling_cost": cooling,
            "command_change_cost": change, "total_cost": total, "reward": -total}
