"""Residual coordinates on the image of the existing actuator projection.

The attainable command set is unchanged. Bounds are evaluated using only the
nominal action and previous command available to every controller.
"""
import numpy as np

from .control import project, residual_bounds, compose_residual_flow


def feasible_residual_interval(nominal, previous, config):
    low, high = residual_bounds(nominal, config)
    nominal = np.asarray(nominal, float)
    lower = project(nominal+low, previous, config)
    upper = project(nominal+high, previous, config)
    center = project(nominal, previous, config)
    return lower, center, upper


def compose_feasible_residual(action, nominal, previous, config):
    action = np.asarray(action, float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action)>1):
        raise ValueError('Finite normalized pair in [-1, 1] required')
    lower, center, upper = feasible_residual_interval(nominal, previous, config)
    return center+action*np.where(action<0, center-lower, upper-center)


def compose_actor_flow(action, context, config, mode, coordinates='raw'):
    if coordinates not in ('raw', 'feasible'):
        raise ValueError('Unknown action coordinates')
    action = np.asarray(action)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action)>1):
        raise ValueError('Finite normalized pair in [-1, 1] required')
    if mode == 'rrl':
        if coordinates == 'feasible':
            return compose_feasible_residual(action, context.nominal_flow, context.previous_command, config)
        return compose_residual_flow(action, context.nominal_flow, config)
    if mode != 'sac':
        raise ValueError('Unknown learning controller')
    action = np.asarray(action)
    if coordinates == 'feasible':
        lower = project(np.zeros(2), context.previous_command, config)
        upper = project(np.full(2, config.flow_max), context.previous_command, config)
        return lower+.5*(action+1)*(upper-lower)
    return .5*(action+1)*config.flow_max
