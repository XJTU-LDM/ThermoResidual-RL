"""Same-well-informed geometry and effective surface-loop parameters."""
from dataclasses import asdict, dataclass, replace
import numpy as np
from .coupled import LoopConfig
from .formation import FormationConfig
from .greybox import GreyboxConfig

@dataclass(frozen=True)
class SiteScenario:
    power_conductance_multiplier: float = 1.0
    cooling_conductance_multiplier: float = 1.0
    length_m: float = 3000.0
    active_start_m: float = 1400.0
    active_mean_C: float = 119.0
    active_gradient_K_m: float = 0.03
    equivalent_inner_diameter_m: float = 0.2
    formation_inner_radius_m: float = 0.12
    conductivity_W_mK: float = 2.4
    wall_areal_resistance_m2_K_W: float = 0.005
    fill_volume_fraction: float = 0.2
    fill_temperature_K: float = 298.15
    initial_source_K: float = 330.59
    preconditioning_days: float = 51.0
    preconditioning_max_step_s: float = 3600.0

    def __post_init__(self):
        if min(self.power_conductance_multiplier, self.cooling_conductance_multiplier) <= 0:
            raise ValueError('Positive conductance multipliers required')
        if not all((np.isfinite(v) for v in asdict(self).values())):
            raise ValueError('Nonfinite site scenario')
        if not 0 <= self.active_start_m < self.length_m or self.preconditioning_days < 0:
            raise ValueError('Invalid depth or prehistory')
        if self.equivalent_inner_diameter_m <= 0 or self.formation_inner_radius_m <= self.equivalent_inner_diameter_m / 2 or self.preconditioning_max_step_s <= 0:
            raise ValueError('Invalid geometry or preconditioning step')

def electrical_reference():
    """Frozen effective coefficients; original project measurements are excluded."""
    return dict(effective_electric_eta_max=0.36913660916421515, branch_conductance=2.547984534683023, source='frozen_effective_reference_coefficients', scope='derived_model_parameters_not_raw_measurements')

def configuration(site=SiteScenario(), *, substep=0.625, radial_cells=192, axial_slices=16):
    half_depth = (site.length_m - site.active_start_m) / 2
    temps = (273.15 + site.active_mean_C - site.active_gradient_K_m * half_depth, 273.15 + site.active_mean_C + site.active_gradient_K_m * half_depth)
    formation = FormationConfig(inner_radius_m=site.formation_inner_radius_m, active_depth_start_m=site.active_start_m, active_depth_end_m=site.length_m, conductivity_W_mK=site.conductivity_W_mK, radial_cells=radial_cells, axial_slices=axial_slices, radial_stretch_power=2.0, temperature_profile_depths_m=(site.active_start_m, site.length_m), temperature_profile_K=temps, wall_areal_resistance_m2_K_W=site.wall_areal_resistance_m2_K_W)
    loop = LoopConfig(length_m=site.length_m, bore_internal_diameter_m=site.equivalent_inner_diameter_m, fill_volume_fraction=site.fill_volume_fraction, fill_temperature_K=site.fill_temperature_K, initial_temperature_K=site.initial_source_K)
    plant = GreyboxConfig(max_substep_s=substep)
    ref = electrical_reference()
    plant = replace(plant, flow_max_kg_s=1.088, turbine_efficiency_max=ref['effective_electric_eta_max'])
    loop = replace(loop, power_branch_conductance=ref['branch_conductance'] * site.power_conductance_multiplier, cooling_branch_conductance=ref['branch_conductance'] * site.cooling_conductance_multiplier)
    return (plant, formation, loop)

def controller_configuration(site=SiteScenario()):
    from .control import ControlConfig
    plant, _, _ = configuration(site)
    return ControlConfig(flow_max=plant.flow_max_kg_s, nominal_eta=plant.turbine_efficiency_max, nominal_design_flow=plant.turbine_design_flow_kg_s, nominal_offdesign=plant.turbine_offdesign_coefficient, nominal_COP=plant.cooling_COP, power_tau_s=plant.power_tau_s, cooling_tau_s=plant.cooling_tau_s)
