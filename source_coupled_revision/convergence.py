"""Development-only stopping diagnostic; never a global optimality certificate."""
import numpy as np


def plateau_status(validations, rule):
    result = dict(plateau=False, reason='insufficient_history')
    window = int(rule['window'])
    if len(validations) < window+1:
        return result
    if validations[-1]['interactions'] < rule['minimum_interactions']:
        return dict(plateau=False, reason='minimum_budget_not_reached')
    if not all(all(s['admissible'] for s in v['scores']) for v in validations):
        return dict(plateau=False, reason='inadmissible_development_checkpoint')
    values = np.asarray([v['selection_cost'] for v in validations], float)
    if not np.isfinite(values).all() or np.any(values < 0):
        return dict(plateau=False, reason='invalid_cost')
    previous_best = float(values[:-window].min())
    best = float(values.min())
    recent = values[-window:]
    improvement = (previous_best-best)/max(previous_best, 1e-12)
    spread = float(np.ptp(recent)/max(float(np.mean(recent)), 1e-12))
    best_gap = (float(np.mean(recent))-best)/max(best, 1e-12)
    plateau = (improvement <= rule['max_relative_improvement'] and
               spread <= rule['max_relative_spread'] and
               best_gap <= rule['max_relative_best_gap'])
    return dict(plateau=bool(plateau), reason='stable_development_plateau' if plateau else 'still_changing',
                relative_improvement=float(improvement), relative_spread=spread,
                relative_best_gap=best_gap, window=window)


def screening_plateau_status(validations, rule):
    """A finite-budget local plateau, with guards against dead or stalled learning."""
    result = plateau_status(validations, rule)
    if not result['plateau']:
        return result
    recent = validations[-int(rule['window']):]
    diagnostics = [v.get('training_diagnostics', {}) for v in recent]
    required = ('mean_reward', 'actor_grad_norm', 'actor_first_layer_inactive_fraction')
    if any(any(k not in d or not np.isfinite(d[k]) for k in required) for d in diagnostics):
        return dict(**{**result, 'plateau': False, 'reason': 'missing_health_diagnostics'})
    rewards = np.array([d['mean_reward'] for d in diagnostics])
    spread = float(np.ptp(rewards)/max(abs(float(rewards.mean())), 1e-8))
    healthy = all(d['actor_grad_norm'] > 1e-8 and
                  d['actor_first_layer_inactive_fraction'] < .90 for d in diagnostics)
    result['training_reward_relative_spread'] = spread
    if not healthy:
        result.update(plateau=False, reason='flat_but_network_health_failed')
    elif spread > rule.get('max_training_reward_spread', .15):
        result.update(plateau=False, reason='training_reward_still_changing')
    else:
        result['reason'] = 'local_training_and_development_plateau'
    return result
