"""Same-well-informed scenario, not a reproduction of the water field trial.

Only the 3000 m length and 119 C active-section mean are taken from Huang's
same-well publications. Unmeasured geometry, inventory and dynamics are explicit
scenario assumptions. The project report supports a pumped surface power loop;
this does not model axial gravity reflux inside the underground heat pipe.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI

from .coupled import LoopConfig, ReferenceLoop
from .formation import FormationConfig, RadialFormation
from .greybox import GreyboxConfig


@dataclass(frozen=True)
class SiteScenario:
    surface_version: int = 1
    power_conductance_multiplier: float = 1.
    cooling_conductance_multiplier: float = 1.
    length_m: float = 3000.
    active_start_m: float = 1400.
    active_mean_C: float = 119.
    active_gradient_K_m: float = .03
    equivalent_inner_diameter_m: float = .20
    formation_inner_radius_m: float = .12
    conductivity_W_mK: float = 2.4
    wall_areal_resistance_m2_K_W: float = .005
    fill_volume_fraction: float = .20
    fill_temperature_K: float = 298.15
    initial_source_K: float = 330.59
    preconditioning_days: float = 51.
    preconditioning_max_step_s: float = 3600.

    def __post_init__(self):
        if self.surface_version not in (1, 2):
            raise ValueError("Unsupported surface model version")
        if min(self.power_conductance_multiplier, self.cooling_conductance_multiplier) <= 0:
            raise ValueError("Positive conductance multipliers required")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Nonfinite site scenario")
        if not 0 <= self.active_start_m < self.length_m or self.preconditioning_days < 0:
            raise ValueError("Invalid depth or prehistory")
        if (self.equivalent_inner_diameter_m <= 0 or
                self.formation_inner_radius_m <= self.equivalent_inner_diameter_m/2 or
                self.preconditioning_max_step_s <= 0):
            raise ValueError("Invalid geometry or preconditioning step")


def electrical_reference():
    """Frozen effective coefficients; original project measurements are excluded."""
    return dict(effective_electric_eta_max=0.36913660916421515,
                branch_conductance=2.547984534683023,
                source="frozen_effective_reference_coefficients",
                scope="derived_model_parameters_not_raw_measurements")


def configuration(site=SiteScenario(), *, substep=.625, radial_cells=192, axial_slices=16):
    half_depth = (site.length_m-site.active_start_m)/2
    temps = (273.15+site.active_mean_C-site.active_gradient_K_m*half_depth,
             273.15+site.active_mean_C+site.active_gradient_K_m*half_depth)
    formation = FormationConfig(
        inner_radius_m=site.formation_inner_radius_m,
        active_depth_start_m=site.active_start_m, active_depth_end_m=site.length_m,
        conductivity_W_mK=site.conductivity_W_mK,
        radial_cells=radial_cells, axial_slices=axial_slices, radial_stretch_power=2.,
        temperature_profile_depths_m=(site.active_start_m, site.length_m),
        temperature_profile_K=temps,
        wall_areal_resistance_m2_K_W=site.wall_areal_resistance_m2_K_W)
    loop = LoopConfig(length_m=site.length_m,
                      bore_internal_diameter_m=site.equivalent_inner_diameter_m,
                      fill_volume_fraction=site.fill_volume_fraction,
                      fill_temperature_K=site.fill_temperature_K,
                      initial_temperature_K=site.initial_source_K)
    plant = GreyboxConfig(max_substep_s=substep)
    if site.surface_version == 2:
        ref = electrical_reference()
        plant = replace(plant, flow_max_kg_s=1.088,
                        turbine_efficiency_max=ref["effective_electric_eta_max"])
        loop = replace(loop, power_branch_conductance=ref["branch_conductance"]*site.power_conductance_multiplier,
                       cooling_branch_conductance=ref["branch_conductance"]*site.cooling_conductance_multiplier)
    return plant, formation, loop


def controller_configuration(site=SiteScenario()):
    from .control import ControlConfig
    plant, _, _ = configuration(site)
    return ControlConfig(flow_max=plant.flow_max_kg_s,
                         nominal_eta=plant.turbine_efficiency_max,
                         nominal_design_flow=plant.turbine_design_flow_kg_s,
                         nominal_offdesign=plant.turbine_offdesign_coefficient,
                         nominal_COP=plant.cooling_COP,
                         power_tau_s=plant.power_tau_s, cooling_tau_s=plant.cooling_tau_s)


def thermal_history(formation, duration_s, wall_K, max_step_s=3600.):
    """Prescribed-wall preparation, not fabricated field operating history.

    Age is internal to the thermal model; controller clock starts at zero.
    Identical fluxes enter the rock storage and heat-extraction accounts.
    """
    if duration_s < 0 or max_step_s <= 0 or not np.isfinite([duration_s, max_step_s, wall_K]).all():
        raise ValueError("Invalid history")
    elapsed, residual, extraction = 0., 0., 0.
    while elapsed < duration_s-1e-8:
        dt = min(max_step_s, max(10., .05*max(elapsed, 1.)), duration_s-elapsed)
        row = formation.advance(wall_K, dt)
        residual += abs(row["energy_residual_kJ"])
        extraction += abs(row["source_heat_kW"])*dt
        elapsed += dt
    return dict(duration_s=duration_s, wall_K=wall_K, max_step_s=max_step_s,
                energy_residual_ratio=residual/max(extraction, 1e-12),
                scope="assumed_prescribed_wall_preconditioning_not_field_replay")


def build_loop(site=SiteScenario(), **numerics):
    plant, formation, loop = configuration(site, **numerics)
    result = ReferenceLoop(plant, formation, loop)
    result.preconditioning = thermal_history(result.formation, site.preconditioning_days*86400,
                                            site.initial_source_K, site.preconditioning_max_step_s)
    return result


def provenance():
    """Keep sources and permissions separate from numerical parameter values."""
    return {
        "same_well_identity": {"status": "author_confirmation", "claim": "same Caofeidian well and technology"},
        "length_m": {"status": "same_well_published", "doi": "10.1016/j.energy.2022.123492",
                     "locator": "Abstract", "note": "historical water trial; current working fluid is ammonia"},
        "active_mean_C": {"status": "same_well_published", "doi": "10.1016/j.energy.2022.123527",
                          "locator": "Introduction: Tangshan trial", "note": "119 C active-section mean, not 95.6 C whole-pipe mean"},
        "initial_source_K": {"status": "project_design_reference", "locator": "TPRI report section 2.1 Table 1",
                             "note": "57.44 C design point, not independently identified initial condition"},
        "surface_return": {"status": "project_topology", "locator": "TPRI report section 2.1, Figures 1 and 13",
                           "note": "plunger working-fluid pump and buffer tank; not internal gravity reflux"},
        "turbine_design_flow_kg_s": {"status": "project_design_reference", "value": 1.088,
                                     "note": "turbine design flow, not shared heat-pipe capacity"},
        "other_site_parameters": {"status": "explicit_scenario_assumptions",
                                  "note": "not measured Caofeidian properties; sensitivity required"},
        "long_term_six_points": {"status": "author_supplied_independent_heat_only_test",
                                 "unit": "kW", "use": "context only: unknown condenser state and history; no curve fitting"},
    }


def scenario_manifest(site=SiteScenario(), **numerics):
    plant, formation, loop = configuration(site, **numerics)
    sources = provenance()
    parameter_sources = {}
    for group, cfg in (("site", site), ("plant", plant), ("formation", formation), ("loop", loop)):
        for key in asdict(cfg):
            parameter_sources[f"{group}.{key}"] = "explicit_scenario_or_numerical_assumption"
    for key in ("site.length_m", "loop.length_m", "formation.active_depth_end_m"):
        parameter_sources[key] = "same_well_published_length"
    parameter_sources["site.active_mean_C"] = "same_well_published_active_mean"
    parameter_sources["site.initial_source_K"] = "project_design_reference_as_initial_condition"
    parameter_sources["loop.initial_temperature_K"] = "project_design_reference_as_initial_condition"
    parameter_sources["plant.turbine_design_flow_kg_s"] = "project_design_reference_not_total_capacity"
    if site.surface_version == 2:
        parameter_sources.update({
            "plant.flow_max_kg_s": "simulation_branch_envelope_at_turbine_design_flow_not_shared_capacity",
            "plant.turbine_efficiency_max": "effective_gross_electrical_coefficient_fitted_Table_12_once",
            "plant.flow_conductance_kg_s_sqrt_MPa": "legacy_field_inactive_in_v2",
            "loop.power_branch_conductance": "local_quadratic_valve_envelope_from_Table_12_not_full_valve_curve",
            "loop.cooling_branch_conductance": "assumed_equal_branch_conductance_requires_sensitivity",
        })
    report_hash = None
    return dict(scope="Caofeidian_same_well_informed_ammonia_greybox_scenario",
                site=asdict(site), plant=asdict(plant), formation=asdict(formation), loop=asdict(loop),
                sources=sources, parameter_sources=parameter_sources,
                private_project_report_sha256=report_hash,
                power_definition=("gross_electrical_equivalent_with_lumped_rejection_losses"
                                  if site.surface_version == 2 else "shaft_equivalent"),
                electrical_reference=electrical_reference() if site.surface_version == 2 else None,
                field_dynamic_calibrated=False, Huang_axial_model_reproduced=False,
                historical_Q_input_to_policy=False, formal_RL_results=False)


def write_manifest(path, site=SiteScenario(), **numerics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario_manifest(site, **numerics), indent=2)+"\n")
