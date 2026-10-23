"""Initialize the frozen control-oriented water-cooled reference model."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI

from .audit import sha256
from .site import SiteScenario, configuration, controller_configuration
from .foundation import WaterCooledLoop, WaterExchanger, initialize_operating_formation


def build_research_loop(model_file, *, source_pressure_MPa=3., condenser_pressure_MPa=1.3,
                        water_inlet_K=300.15, initial_total_flow_kg_s=.6, substep=.625,
                        radial_cells=192, axial_slices=16, loop_type=WaterCooledLoop,
                        loop_options=None):
    """Initialize a declared simulation scenario, not a reconstructed field state.

    The caller must not promote this loader to a site-calibrated training gate.
    All controllers receive the same measurement packet and nominal parameters.
    """
    path = Path(model_file)
    manifest = json.loads(path.read_text())
    choice = manifest['configuration']
    if not choice['water_boundary'] or not choice['balanced_initialization']:
        raise ValueError('Loader requires the water-cooled balanced-initialization model')
    if not np.isfinite([source_pressure_MPa,condenser_pressure_MPa,initial_total_flow_kg_s]).all():
        raise ValueError('Finite operating conditions required')
    if not source_pressure_MPa > condenser_pressure_MPa > 0 or initial_total_flow_kg_s <= 0:
        raise ValueError('Positive pressure head and initial flow required')
    parameters = choice['calibration_parameters']
    initial_K = float(PropsSI('T','P',source_pressure_MPa*1e6,'Q',1,'Ammonia'))
    site = SiteScenario(initial_source_K=initial_K,
        fill_volume_fraction=parameters['fill_volume_fraction'])
    cfg,formation,loop = configuration(site,substep=substep,radial_cells=radial_cells,axial_slices=axial_slices)
    if initial_total_flow_kg_s > cfg.flow_max_kg_s:
        raise ValueError('Initial flow outside command envelope')
    cfg = replace(cfg,sink_capacity_kJ_K=parameters['sink_capacity_kJ_K'],
        sink_initial_K=float(PropsSI('T','P',condenser_pressure_MPa*1e6,'Q',0,'Ammonia'))-cfg.condenser_approach_K)
    loop = replace(loop,return_tau_s=parameters['return_tau_s'])
    if not issubclass(loop_type, WaterCooledLoop):
        raise TypeError('WaterCooledLoop subclass required')
    plant = loop_type(cfg,formation,loop,WaterExchanger(**manifest['water_exchanger']),water_inlet_K,
                     **(loop_options or {}))
    prior = initialize_operating_formation(plant,initial_total_flow_kg_s)
    plant.command = np.array([initial_total_flow_kg_s,0.])
    plant.actual,plant.return_flow = plant.command.copy(),initial_total_flow_kg_s
    # Output filters start at a stated power-only nominal operating point.
    from .control import NominalAllocator
    cc = controller_configuration(site)
    allocator = NominalAllocator(cc)
    plant.power = float(allocator.forward(plant.command,allocator.coefficients(
        source_pressure_MPa,condenser_pressure_MPa))[0])
    plant.water_rejection_kW = plant.exchanger.rejection_kW(
        plant.sink_T+cfg.condenser_approach_K,water_inlet_K)
    provenance = dict(model_sha256=sha256(path),model_candidate=manifest['candidate'],
        initialization=prior,source_pressure_MPa=source_pressure_MPa,condenser_pressure_MPa=condenser_pressure_MPa,
        initial_flow_kg_s=initial_total_flow_kg_s,water_inlet_K=water_inlet_K,
        scope='development_reference_simulator_not_field_dynamic_validation',formal_training_authorized=False)
    return plant,cc,provenance
