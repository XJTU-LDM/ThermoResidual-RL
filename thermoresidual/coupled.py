"""Literature-reference formation coupled to an uncalibrated ammonia loop.

This is a conservative equilibrium greybox, not a reproduced axial heat-pipe
model or a calibrated Caofeidian plant. The liquid receiver has variable volume;
its pressure-volume work is retained in the external energy ledger. The power
lag has an explicit equivalent energy buffer, not a claim about rotor inertia.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.linalg import cho_solve_banded
from scipy.optimize import brentq

from .contracts import Measurement, MeasurementPacket
from .formation import FormationConfig, RadialFormation
from .greybox import GreyboxConfig, ammonia_state


@dataclass(frozen=True)
class ProcessPerturbation:
    """Private process coefficients, not pressure/heat output overrides.

    Unity is exactly the unperturbed model. Disturbances are held over one
    command interval and do not appear in a controller's measurement packet.
    """
    wall_multiplier: float = 1.
    power_valve_multiplier: float = 1.
    cooling_valve_multiplier: float = 1.
    return_tau_multiplier: float = 1.
    sink_conductance_multiplier: float = 1.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Positive finite process multipliers required")


@dataclass(frozen=True)
class LoopConfig:
    # Geometry/fill from the Slovenia reference case, NOT the project well.
    length_m: float = 1800.
    bore_internal_diameter_m: float = .1778
    fill_volume_fraction: float = .20
    fill_temperature_K: float = 240.15
    initial_temperature_K: float = 330.
    # Explicit engineering assumptions pending identification.
    receiver_initial_mass_kg: float = 100.
    return_tau_s: float = 30.
    level_tau_s: float = 120.
    # None retains the historical shared-resistance development model.
    power_branch_conductance: float | None = None
    cooling_branch_conductance: float | None = None

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values() if v is not None):
            raise ValueError("Positive finite loop parameters required")
        if (self.power_branch_conductance is None) != (self.cooling_branch_conductance is None):
            raise ValueError("Both branch conductances must be specified together")
        if not 0 < self.fill_volume_fraction < 1:
            raise ValueError("Invalid fill fraction")


def liquid(t):
    p, h, u, rho = PropsSI(["P", "H", "U", "D"], "T", t, "Q", 0, "Ammonia")
    return float(p)/1000, float(h)/1000, float(u)/1000, float(rho)


def lag_update(value, target, dt, tau):
    end = target + (value-target)*math.exp(-dt/tau)
    mean = target + (value-target)*tau*(-math.expm1(-dt/tau))/dt
    return end, mean


def hydraulic_flows(command, pressure_MPa, condenser_MPa, plant, loop):
    """Local quadratic branch envelopes, not a geothermal shared supply rating.

    The coupled inventory/energy equations determine pressure and heat delivery.
    Independent envelopes do not imply independently available thermal power.
    """
    command = np.asarray(command, float)
    head = math.sqrt(max(pressure_MPa-condenser_MPa, 0.))
    if loop.power_branch_conductance is None:
        supply = plant.flow_conductance_kg_s_sqrt_MPa*head
        return command*min(1., supply/max(float(command.sum()), 1e-30))
    return np.minimum(command, head*np.array([loop.power_branch_conductance,
                                             loop.cooling_branch_conductance]))


class ReferenceLoop:
    """Implicit radial/source heat coupling and finite pumped-return inventory.

    Solver/private state is never passed to a controller. No historical heat or
    future boundary row is read. A zero minimum command permits safe shutdown;
    the maximum and slew limits remain development assumptions.
    """
    def __init__(self, plant=None, formation=None, loop=None):
        self.cfg = plant or GreyboxConfig(max_substep_s=2.5)
        self.loop = loop or LoopConfig()
        self.formation = RadialFormation(formation or FormationConfig())
        c, l = self.cfg, self.loop
        self.volume = math.pi*l.bore_internal_diameter_m**2/4*l.length_m
        rl = PropsSI("D", "T", l.fill_temperature_K, "Q", 0, "Ammonia")
        rv = PropsSI("D", "T", l.fill_temperature_K, "Q", 1, "Ammonia")
        self.mass = self.volume*(l.fill_volume_fraction*rl+(1-l.fill_volume_fraction)*rv)
        self.energy = self.mass*PropsSI("U", "T", l.initial_temperature_K,
                                      "D", self.mass/self.volume, "Ammonia")/1000
        self.receiver_mass = l.receiver_initial_mass_kg
        self.total_mass = self.mass+self.receiver_mass
        self.sink_T = c.sink_initial_K
        self.power = self.cooling = self.return_flow = self.time_s = 0.
        self.command = np.zeros(2)
        self.actual = np.zeros(2)
        self.exchange = 0.
        self._response_cache = {}

    def _rock_affine(self, dt, wall_multiplier=1.):
        f = self.formation
        factors = f._factors(dt, False, wall_multiplier)
        rhs = f.capacity_J_K/dt*f.temperature_K
        rhs[:, -1] += f.outer_conductance_W_K*f.far_temperature_K
        base = np.array([cho_solve_banded((a, True), r) for a, r in zip(factors, rhs)])
        key = dt, wall_multiplier
        if key not in self._response_cache:
            unit = np.zeros_like(rhs)
            unit[:, 0] = f.inner_conductance_W_K*wall_multiplier
            if len(self._response_cache) >= 64:
                self._response_cache.pop(next(iter(self._response_cache)))
            self._response_cache[key] = np.array([
                cho_solve_banded((a, True), r) for a, r in zip(factors, unit)])
        return base, self._response_cache[key]

    def _delivery(self, command, bulk, condenser_MPa, hydraulic_loop):
        """Bulk-port withdrawal and delivered vapor for the base loop."""
        flow = hydraulic_flows(command, bulk.pressure_MPa, condenser_MPa, self.cfg, hydraulic_loop)
        return flow, flow, bulk.vapor_enthalpy_kJ_kg, bulk.vapor_entropy_J_kg_K, {}

    def _reported_source(self, bulk):
        return dict(pressure_MPa=bulk.pressure_MPa, source_temperature_K=bulk.temperature_K)

    def _substep(self, command, dt, ambient, disturbance=None):
        c, l, f = self.cfg, self.loop, self.formation
        d = disturbance or ProcessPerturbation()
        a = ammonia_state(self.mass, self.energy, self.volume)
        pc, hl, ul, rho = liquid(self.sink_T+c.condenser_approach_K)
        if d.power_valve_multiplier != 1. or d.cooling_valve_multiplier != 1.:
            if l.power_branch_conductance is None:
                raise ValueError("Branch perturbations require explicit branch hydraulics")
            from dataclasses import replace
            hydraulic_loop = replace(l, power_branch_conductance=l.power_branch_conductance*d.power_valve_multiplier,
                                     cooling_branch_conductance=l.cooling_branch_conductance*d.cooling_valve_multiplier)
        else:
            hydraulic_loop = l
        withdrawal, flow, delivery_h, delivery_s, delivery_diag = self._delivery(command, a, pc/1000, hydraulic_loop)
        out = float(withdrawal.sum())
        r_end, r_mean = lag_update(self.return_flow, out, dt, l.return_tau_s*d.return_tau_multiplier)
        ret = max(0., r_mean+(self.receiver_mass-l.receiver_initial_mass_kg)/l.level_tau_s)
        ret = min(ret, self.receiver_mass/dt+out-1e-9)
        mnew = self.mass+dt*(ret-out)
        mrnew = self.receiver_mass+dt*(out-ret)
        wpump = max(a.pressure_MPa*1000-pc, 0)/rho/c.pump_efficiency
        hv = a.vapor_enthalpy_kJ_kg
        # The site configuration supplies an effective gross-electric efficiency.
        # All non-exported energy goes to the lumped rejection loop; no extra
        # component-efficiency factors are multiplied into this effective map.
        converted_power = 0.
        if flow[0] > 0:
            h2 = PropsSI("H", "P", pc*1000, "S", delivery_s, "Ammonia")/1000
            eta = c.turbine_efficiency_max*max(0., 1-c.turbine_offdesign_coefficient*
                ((flow[0]-c.turbine_design_flow_kg_s)/c.turbine_design_flow_kg_s)**2)
            converted_power = flow[0]*eta*max(delivery_h-h2, 0.)
        drive = flow[1]*max(delivery_h-hl, 0.)
        phase = (self.time_s+dt/2)%c.cooling_cycle_period_s
        half = c.cooling_cycle_period_s/2
        distance = min(phase, abs(phase-half), c.cooling_cycle_period_s-phase)
        switching = max(0., 1-distance/(c.cooling_switch_window_s/2))
        factor = 1+c.cooling_cycle_amplitude*math.sin(2*math.pi*phase/c.cooling_cycle_period_s)
        factor -= c.cooling_switch_loss*switching
        pnew, pmean = lag_update(self.power, converted_power, dt, c.power_tau_s)
        qnew, qmean = lag_update(self.cooling, c.cooling_COP*factor*drive, dt, c.cooling_tau_s)
        base, response = self._rock_affine(dt, d.wall_multiplier)
        wall_conductance = f.inner_conductance_W_K*d.wall_multiplier
        q0 = float(np.sum(wall_conductance*base[:, 0]))/1000
        qcoef = float(np.sum(wall_conductance*(response[:, 0]-1)))/1000

        def source_residual(t):
            u = mnew*PropsSI("U", "T", t, "D", mnew/self.volume, "Ammonia")/1000
            return u-self.energy-dt*(q0+qcoef*t-out*hv+ret*(hl+wpump))

        tf = brentq(source_residual, 260., 390., xtol=1e-10)
        enew = mnew*PropsSI("U", "T", tf, "D", mnew/self.volume, "Ammonia")/1000
        ammonia_state(mnew, enew, self.volume)
        rock = base+response*tf
        source_heat = q0+qcoef*tf
        far_heat = float(np.sum(f.outer_conductance_W_K*(f.far_temperature_K-rock[:, -1])))/1000
        rock_change = float(np.sum(f.capacity_J_K*(rock-f.temperature_K)))/1000
        receiver_before = self.receiver_mass*ul
        volume_before = self.receiver_mass/rho
        # Variable-volume liquid receiver with explicit boundary work.
        def sink_terms(t):
            pn, hn, un, rn = liquid(t+c.condenser_approach_K)
            work = .5*(pc+pn)*(mrnew/rn-volume_before)
            ambient_heat = c.sink_conductance_kW_K*d.sink_conductance_multiplier*(t-ambient)
            storage = c.sink_capacity_kJ_K*(t-self.sink_T)+mrnew*un-receiver_before
            return storage-dt*(out*hv-converted_power-ret*hl+qmean-ambient_heat)+work

        ts = brentq(sink_terms, 260., 385., xtol=1e-10)
        pn, hn, un, rn = liquid(ts+c.condenser_approach_K)
        work = .5*(pc+pn)*(mrnew/rn-volume_before)
        ambient_heat = c.sink_conductance_kW_K*d.sink_conductance_multiplier*(ts-ambient)
        receiver_change = mrnew*un-receiver_before
        water_exchange = c.sink_capacity_kJ_K*(ts-self.sink_T)/dt+ambient_heat
        storage = (rock_change+enew-self.energy+receiver_change+
                   c.sink_capacity_kJ_K*(ts-self.sink_T)+c.power_tau_s*(pnew-self.power))
        external = dt*(far_heat+ret*wpump+qmean-pmean-ambient_heat)-work
        diag = dict(source_heat_kW=source_heat, far_heat_kW=far_heat, power_kW=pmean,
                    power_conversion_kW=converted_power, net_power_kW=pmean-ret*wpump,
                    cooling_kW=qmean, pump_work_kW=ret*wpump, ambient_rejection_kW=ambient_heat,
                    water_exchange_kW=water_exchange,
                    receiver_boundary_work_kW=work/dt, return_flow_kg_s=ret,
                    energy_throughput_kJ=dt*(abs(far_heat)+abs(ret*wpump)+abs(qmean)+abs(pmean)+abs(ambient_heat))+abs(work),
                    energy_residual_kJ=storage-external,
                    source_residual_kJ=enew-self.energy-dt*(source_heat-out*hv+ret*(hl+wpump)),
                    mass_residual_kg=mnew+mrnew-self.total_mass)
        diag.update(delivery_diag)
        self.mass, self.energy, self.receiver_mass = mnew, enew, mrnew
        self.sink_T, self.power, self.cooling, self.return_flow = ts, pnew, qnew, r_end
        self.actual, self.exchange = flow, water_exchange
        self.time_s += dt
        f.temperature_K, f.time_s = rock, self.time_s
        return diag

    def step(self, request, dt_s=None, *, ambient_K=None, disturbance=None):
        dt = self.cfg.control_step_s if dt_s is None else float(dt_s)
        ambient = self.cfg.ambient_K if ambient_K is None else float(ambient_K)
        request = np.asarray(request, float)
        if disturbance is not None and not isinstance(disturbance, ProcessPerturbation):
            raise TypeError("ProcessPerturbation required, not measured output overrides")
        if (request.shape != (2,) or not np.isfinite(request).all() or
                not np.isfinite(dt) or not 0 < dt <= self.cfg.control_step_s or
                not np.isfinite(ambient) or ambient <= 0):
            raise ValueError("Invalid command, interval or ambient")
        command = np.clip(np.clip(request, 0, self.cfg.flow_max_kg_s),
                          self.command-self.cfg.command_rate_kg_s2*dt,
                          self.command+self.cfg.command_rate_kg_s2*dt)
        names = ("mass", "energy", "receiver_mass", "sink_T", "power", "cooling", "return_flow",
                 "time_s", "actual", "exchange")
        snapshot = {n: getattr(self, n) for n in names}
        rock_before, time_before = self.formation.temperature_K, self.formation.time_s
        totals, elapsed = {}, 0.
        try:
            while elapsed < dt-1e-10:
                h = min(self.cfg.max_substep_s, dt-elapsed)
                d = (self._substep(command, h, ambient) if disturbance is None else
                     self._substep(command, h, ambient, disturbance))
                for k, v in d.items():
                    totals[k] = totals.get(k, 0.)+v*(h if k.endswith(("_kW", "_kg_s")) else 1.)
                elapsed += h
        except Exception:
            for n, v in snapshot.items():
                setattr(self, n, v)
            self.formation.temperature_K, self.formation.time_s = rock_before, time_before
            raise
        self.command = command
        result = {k: v/dt if k.endswith(("_kW", "_kg_s")) else v for k, v in totals.items()}
        self.exchange = result["water_exchange_kW"]
        a = ammonia_state(self.mass, self.energy, self.volume)
        result.update(time_s=self.time_s, pressure_MPa=a.pressure_MPa,
                      condenser_pressure_MPa=liquid(self.sink_T+self.cfg.condenser_approach_K)[0]/1000,
                      power_end_kW=self.power, cooling_end_kW=self.cooling,
                      source_mass_kg=self.mass, receiver_mass_kg=self.receiver_mass,
                      mass_residual_kg=self.mass+self.receiver_mass-self.total_mass,
                      source_temperature_K=a.temperature_K, bulk_quality=a.bulk_quality,
                      command_power_kg_s=command[0], command_cooling_kg_s=command[1],
                      actual_power_kg_s=self.actual[0], actual_cooling_kg_s=self.actual[1],
                      supply_shortfall_kg_s=max(0., float(command.sum()-self.actual.sum())))
        result.update(self._reported_source(a))
        return result

    def measurement_packet(self, episode_id):
        a = ammonia_state(self.mass, self.energy, self.volume)
        values = dict(power_kW=self.power, cooling_kW=self.cooling,
                      wellhead_pressure_MPa=self._reported_source(a)['pressure_MPa'], measured_exchange_kW=self.exchange,
                      flow_power_kg_s=float(self.actual[0]), flow_cooling_kg_s=float(self.actual[1]),
                      cooling_water_inlet_C=self.sink_T-273.15,
                      condenser_pressure_MPa=liquid(self.sink_T+self.cfg.condenser_approach_K)[0]/1000)
        return MeasurementPacket(episode_id, tuple(Measurement(k, v, self.time_s, self.time_s)
                                                    for k, v in values.items()))
