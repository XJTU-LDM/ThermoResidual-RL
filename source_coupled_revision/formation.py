"""Conservative radial formation reference, not a full heat-pipe reproduction.

Independent axial slices omit axial conduction and pore-fluid advection. Pore
heat capacity is frozen at initial depth-dependent temperature and an explicit
hydrostatic reference pressure. The borehole boundary is prescribed, not an
ammonia boiling correlation or a turbine inlet temperature.
"""
from dataclasses import asdict, dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.linalg import cho_solve_banded, cholesky_banded


@dataclass(frozen=True)
class FormationConfig:
    inner_radius_m: float = .105
    outer_radius_m: float = 150.
    active_depth_start_m: float = 650.
    active_depth_end_m: float = 1800.
    conductivity_W_mK: float = 2.4
    rock_density_kg_m3: float = 2400.
    rock_cp_J_kgK: float = 850.
    porosity: float = .2
    radial_cells: int = 96
    axial_slices: int = 16
    radial_stretch_power: float = 1.
    hydrostatic_reference_density_kg_m3: float = 1000.
    gravity_m_s2: float = 9.80665
    surface_absolute_pressure_Pa: float = 101325.
    # Empty tuples retain the original independent Slovenia reference.
    temperature_profile_depths_m: tuple = ()
    temperature_profile_K: tuple = ()
    wall_areal_resistance_m2_K_W: float = 0.

    def __post_init__(self):
        if not all(np.isfinite(x).all() for x in asdict(self).values()):
            raise ValueError("Finite parameters required")
        z, t = self.temperature_profile_depths_m, self.temperature_profile_K
        if bool(z) != bool(t) or len(z) != len(t):
            raise ValueError("Temperature profile coordinates must match")
        if z and (len(z) < 2 or np.any(np.diff(z) <= 0) or min(t) <= 0 or
                  z[0] > self.active_depth_start_m or z[-1] < self.active_depth_end_m):
            raise ValueError("Temperature profile must cover the active depth")
        if self.wall_areal_resistance_m2_K_W < 0:
            raise ValueError("Negative wall resistance")
        if not (0 < self.inner_radius_m < self.outer_radius_m and
                0 <= self.active_depth_start_m < self.active_depth_end_m):
            raise ValueError("Invalid cylindrical domain")
        if not 0 <= self.porosity <= 1:
            raise ValueError("Invalid porosity")
        if (not isinstance(self.radial_cells, int) or self.radial_cells < 3 or
                not isinstance(self.axial_slices, int) or self.axial_slices < 1):
            raise ValueError("Integer cell counts required")
        for key in ("conductivity_W_mK", "rock_density_kg_m3", "rock_cp_J_kgK",
                    "hydrostatic_reference_density_kg_m3", "gravity_m_s2",
                    "surface_absolute_pressure_Pa", "radial_stretch_power"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")


def initial_temperature_K(depth_m):
    """Gruber et al., Energy 353 (2026), 140961, Table 2, Slovenia case."""
    return 273.15 + 81.13 / (1 + np.exp(-(0.00606*np.asarray(depth_m)-1.99)))


class RadialFormation:
    def __init__(self, config: FormationConfig):
        self.cfg = config
        c = config
        coordinate = np.linspace(0., 1., c.radial_cells+1)**c.radial_stretch_power
        self.faces = c.inner_radius_m*np.exp(np.log(c.outer_radius_m/c.inner_radius_m)*coordinate)
        self.centers = np.sqrt(self.faces[:-1]*self.faces[1:])
        dz = (c.active_depth_end_m-c.active_depth_start_m)/c.axial_slices
        self.depths = c.active_depth_start_m + dz*(np.arange(c.axial_slices)+.5)
        self.far_temperature_K = (np.interp(self.depths, c.temperature_profile_depths_m,
                                            c.temperature_profile_K)
                                  if c.temperature_profile_depths_m else initial_temperature_K(self.depths))
        pressure = (c.surface_absolute_pressure_Pa + c.hydrostatic_reference_density_kg_m3 *
                    c.gravity_m_s2*self.depths)
        pore_capacity = np.array([PropsSI("D", "T", t, "P", p, "Water") *
                                  PropsSI("C", "T", t, "P", p, "Water")
                                  for t, p in zip(self.far_temperature_K, pressure)])
        volumetric = (1-c.porosity)*c.rock_density_kg_m3*c.rock_cp_J_kgK + c.porosity*pore_capacity
        volume = np.pi*(self.faces[1:]**2-self.faces[:-1]**2)*dz
        self.capacity_J_K = volumetric[:, None]*volume[None, :]
        prefactor = 2*np.pi*c.conductivity_W_mK*dz
        self.internal_conductance_W_K = prefactor/np.log(self.centers[1:]/self.centers[:-1])
        self.inner_conductance_W_K = 1/(np.log(self.centers[0]/c.inner_radius_m)/prefactor +
            c.wall_areal_resistance_m2_K_W/(2*np.pi*c.inner_radius_m*dz))
        self.outer_conductance_W_K = prefactor/np.log(c.outer_radius_m/self.centers[-1])
        self.temperature_K = np.repeat(self.far_temperature_K[:, None], c.radial_cells, axis=1)
        self.time_s = 0.
        self._factor_cache = {}

    def _factors(self, dt_s, insulated, wall_multiplier=1.):
        if not np.isfinite(wall_multiplier) or wall_multiplier <= 0:
            raise ValueError("Positive wall conductance multiplier required")
        key = float(dt_s), bool(insulated), float(wall_multiplier)
        if key not in self._factor_cache:
            conductance = self.internal_conductance_W_K
            diagonal = np.zeros(self.cfg.radial_cells)
            diagonal[:-1] += conductance
            diagonal[1:] += conductance
            diagonal[0] += 0 if insulated else self.inner_conductance_W_K*wall_multiplier
            diagonal[-1] += self.outer_conductance_W_K
            factors = []
            for capacity in self.capacity_J_K:
                band = np.zeros((2, self.cfg.radial_cells))
                band[0] = capacity/dt_s + diagonal
                band[1, :-1] = -conductance
                factors.append(cholesky_banded(band, lower=True))
            if len(self._factor_cache) >= 64:
                self._factor_cache.pop(next(iter(self._factor_cache)))
            self._factor_cache[key] = factors
        return self._factor_cache[key]

    def advance(self, wall_temperature_K, dt_s):
        """Backward Euler with identical heat fluxes in state and energy ledger.

        None is an insulated borehole. Otherwise wall temperature is a scalar
        or one value per axial slice. Positive extraction means rock -> wall.
        """
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Positive finite timestep required")
        insulated = wall_temperature_K is None
        wall = np.zeros(self.cfg.axial_slices) if insulated else np.broadcast_to(
            np.asarray(wall_temperature_K, float), (self.cfg.axial_slices,))
        if not np.isfinite(wall).all() or (not insulated and (wall <= 0).any()):
            raise ValueError("Finite positive Kelvin boundary required")
        before = self.temperature_K.copy()
        rhs = self.capacity_J_K/dt_s*before
        if not insulated:
            rhs[:, 0] += self.inner_conductance_W_K*wall
        rhs[:, -1] += self.outer_conductance_W_K*self.far_temperature_K
        after = np.array([cho_solve_banded((f, True), r)
                          for f, r in zip(self._factors(dt_s, insulated), rhs)])
        extracted_W = 0. if insulated else float(np.sum(self.inner_conductance_W_K*(after[:, 0]-wall)))
        far_W = float(np.sum(self.outer_conductance_W_K*(self.far_temperature_K-after[:, -1])))
        storage_J = float(np.sum(self.capacity_J_K*(after-before)))
        residual_J = storage_J - dt_s*(far_W-extracted_W)
        self.temperature_K = after
        self.time_s += dt_s
        return dict(time_s=self.time_s, source_heat_kW=extracted_W/1000,
                    far_boundary_heat_kW=far_W/1000, storage_change_kJ=storage_J/1000,
                    energy_residual_kJ=residual_J/1000,
                    inner_rock_mean_K=float(after[:, 0].mean()))

    def analytic_steady_profile(self, wall_temperature_K):
        wall = np.broadcast_to(np.asarray(wall_temperature_K, float), (self.cfg.axial_slices,))
        c = self.cfg
        contact = c.conductivity_W_mK*c.wall_areal_resistance_m2_K_W/c.inner_radius_m
        ratio = (contact+np.log(self.centers/c.inner_radius_m)) / (contact+np.log(
            c.outer_radius_m/c.inner_radius_m))
        return wall[:, None] + (self.far_temperature_K-wall)[:, None]*ratio

    def analytic_steady_extraction_kW(self, wall_temperature_K):
        c = self.cfg
        length = (c.active_depth_end_m-c.active_depth_start_m)/c.axial_slices
        wall = np.broadcast_to(np.asarray(wall_temperature_K, float), (c.axial_slices,))
        resistance = (np.log(c.outer_radius_m/c.inner_radius_m) +
                      c.conductivity_W_mK*c.wall_areal_resistance_m2_K_W/c.inner_radius_m)
        return float(np.sum(2*np.pi*c.conductivity_W_mK*length*(self.far_temperature_K-wall) /
                            resistance)/1000)
