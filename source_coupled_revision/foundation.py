"""Water-cooled conservative research loop with explicit initialization priors.

The epsilon-NTU closure assumes a regulated cooling-water inlet and a uniform
condensing temperature. Neither this assumption nor the radial initial profile
constitutes field identification of the unobserved axial ammonia dynamics.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
from CoolProp.CoolProp import PropsSI

from .contracts import Measurement, MeasurementPacket
from .coupled import ReferenceLoop, liquid
from .greybox import ammonia_state


TIN = 'T6-冷凝器入水温度(℃)'
TOUT = 'T7-冷凝器出水温度(℃)'
HEAT = 'q-采热量(kW)'
PC = 'P3-冷凝器压力(MPa)'


@dataclass(frozen=True)
class WaterExchanger:
    capacity_rate_kW_K: float
    UA_kW_K: float

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Positive finite water-side coefficients required')

    @property
    def effectiveness(self):
        return float(-np.expm1(-self.UA_kW_K/self.capacity_rate_kW_K))

    @property
    def conductance_kW_K(self):
        return self.capacity_rate_kW_K*self.effectiveness

    def rejection_kW(self, condensing_K, inlet_K):
        return self.conductance_kW_K*(condensing_K-inlet_K)


def fit_water_exchanger(frame, fit_stop, pressure_offset_MPa=0.):
    """Fit ONLY the supplied prefix; invalid rows are counted, never imputed.

    q/(Tout-Tin) and -Cdot*log(1-epsilon) are paired water-calorimetry estimates.
    q is not used as a geothermal-source input or as an online plant forcing.
    """
    if not isinstance(fit_stop, int) or not 0 < fit_stop <= len(frame):
        raise ValueError('Explicit nonempty development prefix required')
    data = frame.iloc[:fit_stop]
    p = data[PC].to_numpy()+pressure_offset_MPa
    tc = np.full(len(data), np.nan)
    valid_p = np.isfinite(p) & (p > .001) & (p < 10.)
    tc[valid_p] = PropsSI('T', 'P', p[valid_p]*1e6, 'Q', 0, 'Ammonia')-273.15
    tin, tout, q = (data[c].to_numpy() for c in (TIN, TOUT, HEAT))
    with np.errstate(divide='ignore', invalid='ignore'):
        capacity = q/(tout-tin)
        epsilon = (tout-tin)/(tc-tin)
        ua = -capacity*np.log1p(-epsilon)
    valid = np.isfinite(capacity) & np.isfinite(ua) & (capacity > 0) & (ua > 0)
    valid &= (tout > tin) & (tc > tout) & (epsilon > 0) & (epsilon < 1)
    if valid.sum() < 10:
        raise ValueError('Insufficient physically ordered water-side observations')
    model = WaterExchanger(float(np.median(capacity[valid])), float(np.median(ua[valid])))
    summary = dict(model=asdict(model), fit_rows=[0, fit_stop], valid_rows=int(valid.sum()),
        excluded_rows=int((~valid).sum()), pressure_offset_MPa=pressure_offset_MPa,
        capacity_quantiles_kW_K=np.quantile(capacity[valid], [.05,.5,.95]).tolist(),
        UA_quantiles_kW_K=np.quantile(ua[valid], [.05,.5,.95]).tolist(),
        epsilon_quantiles=np.quantile(epsilon[valid], [.05,.5,.95]).tolist(),
        effective_conductance_kW_K=model.conductance_kW_K,
        interpretation='effective constant coefficients conditional on saturated condenser and calorimetry circuit',
        online_heat_input=False)
    return model, summary


def initialize_operating_formation(plant, past_mean_flow_kg_s):
    """Set a bounded radial initial profile from a local steady-balance prior.

    The inferred thermal penetration is an initial-state assumption, not a
    measured operating age. No balance is enforced after initialization.
    """
    if not np.isfinite(past_mean_flow_kg_s) or past_mean_flow_kg_s <= 0:
        raise ValueError('Positive causal past mean flow required')
    a = ammonia_state(plant.mass, plant.energy, plant.volume)
    pc, hl, _, rho = liquid(plant.sink_T+plant.cfg.condenser_approach_K)
    wp = max(a.pressure_MPa*1000-pc, 0)/rho/plant.cfg.pump_efficiency
    demand = past_mean_flow_kg_s*(a.vapor_enthalpy_kJ_kg-hl-wp)
    f, wall = plant.formation, a.temperature_K
    c = f.cfg
    if np.any(f.far_temperature_K <= wall) or demand <= 0:
        raise ValueError('Operating thermal prior outside extraction regime')
    contact = c.conductivity_W_mK*c.wall_areal_resistance_m2_K_W/c.inner_radius_m
    radial = contact+np.log(f.centers/c.inner_radius_m)
    numerator = float(np.sum(f.inner_conductance_W_K*(f.far_temperature_K-wall)))*radial[0]/1000
    denominator = numerator/demand
    if denominator < radial[0] or denominator > contact+np.log(c.outer_radius_m/c.inner_radius_m):
        raise ValueError('Required initial heat outside finite-domain profile family')
    ratio = np.minimum(radial/denominator, 1.)
    f.temperature_K = wall+(f.far_temperature_K-wall)[:, None]*ratio
    actual = float(np.sum(f.inner_conductance_W_K*(f.temperature_K[:,0]-wall)))/1000
    return dict(initialization='causal local steady-balance radial profile',
        past_mean_flow_kg_s=float(past_mean_flow_kg_s), initial_heat_kW=actual,
        initial_required_heat_kW=float(demand), initial_balance_error_kW=float(actual-demand),
        effective_penetration_radius_m=float(c.inner_radius_m*np.exp(denominator-contact)),
        measured_operating_age=False)


class WaterCooledLoop(ReferenceLoop):
    """Same conservative solver, with epsilon-NTU heat rejection to measured Tin.

    The old sink state is an equivalent storage coordinate: Tc = sink_T +
    approach. Transforming the inlet by the same approach preserves the exact
    energy ledger while Qwater = Cdot*epsilon*(Tc-Tin).
    """
    def __init__(self, plant, formation, loop, exchanger, inlet_K):
        if not isinstance(exchanger, WaterExchanger):
            raise TypeError('WaterExchanger required')
        if not np.isfinite(inlet_K) or not 260 < inlet_K < 370:
            raise ValueError('Cooling-water inlet outside research domain')
        self.exchanger = exchanger
        self.water_inlet_K = float(inlet_K)
        self.water_rejection_kW = 0.
        super().__init__(replace(plant, sink_conductance_kW_K=exchanger.conductance_kW_K,
            ambient_K=inlet_K-plant.condenser_approach_K), formation, loop)

    def step(self, request, dt_s=None, *, cooling_water_inlet_K=None, disturbance=None):
        inlet = self.water_inlet_K if cooling_water_inlet_K is None else float(cooling_water_inlet_K)
        if not np.isfinite(inlet) or not 260 < inlet < 370:
            raise ValueError('Cooling-water inlet outside research domain')
        result = super().step(request, dt_s, ambient_K=inlet-self.cfg.condenser_approach_K,
                              disturbance=disturbance)
        self.water_inlet_K = inlet
        self.water_rejection_kW = result['ambient_rejection_kW']
        result.update(cooling_water_inlet_K=inlet,
            cooling_water_outlet_K=inlet+self.water_rejection_kW/self.exchanger.capacity_rate_kW_K,
            cooling_water_rejection_kW=self.water_rejection_kW)
        return result

    def measurement_packet(self, episode_id):
        packet = super().measurement_packet(episode_id)
        replacements = {'cooling_water_inlet_C':self.water_inlet_K-273.15,
                        'measured_exchange_kW':self.water_rejection_kW}
        return MeasurementPacket(episode_id, tuple(
            Measurement(r.channel, replacements.get(r.channel, r.value), r.sampled_at_s,
                        r.available_at_s, r.valid) for r in packet.readings))
