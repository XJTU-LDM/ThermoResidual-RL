"""Development-identified coupled nominal map, shared by all controllers.

This is a quasi-steady simulation calibration, not a field-identified model
or a prediction of hidden source states. No extrapolation outside its samples.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class JointAllocator:
    def __init__(self, points):
        x = np.asarray(points, float)
        if x.ndim != 2 or x.shape[1] != 4 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError('Finite [flow_power, flow_cooling, power, cooling] map required')
        self.inverse = LinearNDInterpolator(x[:, 2:], x[:, :2])
        self.mapping = LinearNDInterpolator(x[:, :2], x[:, 2:])

    def allocate(self, target, pressure, condenser):
        result = np.asarray(self.inverse(np.asarray(target,float))).reshape(2)
        if not np.isfinite(result).all():
            raise ValueError('Target outside calibrated joint allocation map')
        return result

    def coefficients(self, pressure, condenser):
        # Pressure remains in the common observation. It is not a hidden map input.
        return None

    def forward(self, flow, coefficients=None):
        result = np.asarray(self.mapping(np.asarray(flow,float))).reshape(np.shape(flow))
        if not np.isfinite(result).all():
            raise ValueError('Flow outside calibrated joint allocation map')
        return result
