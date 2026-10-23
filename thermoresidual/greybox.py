"""Conservative reference-model parameters and two-phase ammonia state."""
from dataclasses import asdict, dataclass, fields
import numpy as np
from CoolProp.CoolProp import PropsSI
from .contracts import finite



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
