import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reproduce import ROOT, load_jobs, verify
from thermoresidual.control import ControlConfig, OBS_DIM, project
from thermoresidual.feasible_actions import compose_actor_flow
from thermoresidual.contracts import Measurement, MeasurementPacket, Observer, reward_components
from thermoresidual.stochastic import ProcessConfig, process_stream


def test_all_release_checksums():
    assert verify() > 50


def test_main_matrix_and_shared_recipe():
    jobs = load_jobs('main')
    assert len(jobs) == 480
    assert len({(j['mode'], j['profile'], j['condition'], j['process_seed']) for j in jobs}) == 480
    assert {j['mode'] for j in jobs} == {'rrl', 'sac', 'pi', 'gspid', 'mpc', 'physics'}
    demo = {j['mode']: j for j in load_jobs('demo')}
    for key in ('model_root', 'learning', 'target_protocol', 'joint_allocation_points',
                'tracking_weights', 'controller_overrides', 'interactions'):
        assert demo['rrl']['plan'][key] == demo['sac']['plan'][key]
    assert demo['rrl']['plan']['interactions'] == 768000
    assert OBS_DIM == 191


def test_full_action_sets_and_rate_projection():
    cfg = ControlConfig(flow_max=1.088)
    assert cfg.residual_power is cfg.residual_cooling is None
    rng = np.random.default_rng(7)
    for _ in range(1000):
        ctx = SimpleNamespace(nominal_flow=rng.uniform(0, cfg.flow_max, 2),
                              previous_command=rng.uniform(0, cfg.flow_max, 2))
        for endpoint in ([-1, -1], [-1, 1], [1, -1], [1, 1]):
            a = project(compose_actor_flow(endpoint, ctx, cfg, 'rrl'), ctx.previous_command, cfg)
            b = project(compose_actor_flow(endpoint, ctx, cfg, 'sac'), ctx.previous_command, cfg)
            np.testing.assert_allclose(a, b, atol=1e-14)
        result = project(rng.normal(size=2), ctx.previous_command, cfg)
        assert np.all((0 <= result) & (result <= cfg.flow_max))
        assert np.max(np.abs(result-ctx.previous_command)) <= cfg.rate_per_step + 1e-14


def test_future_measurement_rejected():
    reading = Measurement('wellhead_pressure_MPa', 3., 0., 10.)
    observer = Observer('test', max_age_s=60., max_step_s=10.)
    with pytest.raises(ValueError, match='Future'):
        observer.update(MeasurementPacket('test', (reading,)), 0.)


def test_shared_objective_arithmetic():
    row = reward_components([14., 15.], [15., 17.], [.6, .2], [.54, .18],
                            power_weight=1., cooling_weight=1.)
    expected = .01 + .04 + .002*(.06**2 + .02**2)
    assert row['total_cost'] == pytest.approx(expected)
    assert row['reward'] == pytest.approx(-expected)


def test_process_prefix_and_positive_multipliers():
    cfg = ProcessConfig(oscillation_weight=.9)
    short, long = process_stream(cfg, 100, 19), process_stream(cfg, 200, 19)
    np.testing.assert_array_equal(short, long[:100])
    assert np.all(short > 0)
    assert not np.array_equal(short, process_stream(cfg, 100, 20))


def test_initial_learners_are_paired_and_empty():
    import torch
    states = [torch.load(ROOT / j['plan']['resume_learner_checkpoint'], weights_only=True, map_location='cpu')
              for j in load_jobs('demo') if j['mode'] in ('rrl', 'sac')]
    for state in states:
        assert state['updates'] == state['learner']['replay_count'] == 0
        assert all(tensor.numel() == 0 for tensor in state['learner']['replay'].values())
    for group in ('actor', 'critics', 'targets'):
        assert all(torch.equal(v, states[1][group][k]) for k, v in states[0][group].items())


def test_current_policy_assets():
    from hashlib import sha256
    import torch
    manifest = json.loads((ROOT / 'ASSETS_MANIFEST.json').read_text())
    assert manifest['raw_field_data_included'] is False
    records = manifest['policies']
    assert len(records) == 6
    assert {(r['controller'], r['weight_ratio']) for r in records} == {
        (mode, ratio) for mode in ('rrl', 'sac') for ratio in (.5, 1., 2.)}
    for record in records:
        assert sha256((ROOT / record['checkpoint']).read_bytes()).hexdigest() == record['checkpoint_sha256']
        assert record['interactions'] <= 768000
        initial = torch.load(ROOT / record['initial_learner'], map_location='cpu', weights_only=True)
        assert initial['updates'] == initial['learner']['replay_count'] == 0
        for key in ('actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            assert not initial['learner'][key]['state']


def test_effective_coefficient_export_excludes_original_measurements():
    from thermoresidual.site import electrical_reference
    reference = electrical_reference()
    assert reference['effective_electric_eta_max'] == 0.36913660916421515
    assert reference['branch_conductance'] == 2.547984534683023
    assert 'electrical_kW' not in reference and 'flow_kg_s' not in reference


def test_pooled_summary_is_not_an_average_of_rmse():
    import pandas as pd
    from summarize_results import summarize
    frame = pd.DataFrame(dict(mode=['rrl', 'rrl'], profile=['standard']*2, condition=[0, 1],
        process_seed=[1, 1], samples=[1, 3], power_RMSE=[1., 3.], cooling_RMSE=[1., 3.],
        mean_cost=[.1, .9], rate_violations=[0, 0], bound_violations=[0, 0]))
    result = summarize(frame).iloc[0]
    assert result.combined_RMSE == pytest.approx(np.sqrt(7.))


def test_no_private_material_in_release():
    from tools.audit_public_release import scan
    assert scan()['passed']


def test_pressure_ensemble_runs_without_raw_records():
    from thermoresidual.pressure import Ensemble
    directory = ROOT / 'assets/pressure'
    model = Ensemble(directory, list(range(20260920, 20260925)))
    mean, std = model.predict(np.full((2, 7), 3.), np.array([360., 420.]))
    assert mean.shape == std.shape == (2,)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()


def test_baseline_grids_do_not_modify_common_constraints():
    from tune_baselines import candidates
    for mode, count in [('pi', 24), ('mpc', 9), ('gspid', 54)]:
        settings = list(candidates(mode))
        assert len(settings) == count
        assert all(not {'flow_max', 'rate_per_step', 'residual_power', 'residual_cooling'} & set(v)
                   for v in settings)


def test_final_only_layout_and_current_sensor_protocol():
    assert not (ROOT / 'experiments').exists()
    assert not (ROOT / 'results').exists()
    assert not (ROOT / 'PUBLIC_EXPORT.json').exists()
    assert (ROOT / 'docs/figures/control_framework.png').is_file()
    for suite, expected in [('stress', 840), ('sensor', 792), ('weights', 320)]:
        assert len(load_jobs(suite)) == expected
    assert all(j.get('pressure_offset', 0.) == 0. for j in load_jobs('sensor'))


def test_all_frozen_asset_paths_resolve():
    for suite in ('demo', 'main', 'stress', 'sensor', 'weights'):
        for job in load_jobs(suite):
            plan = job['plan']
            for key in ('model_root', 'normalization_source', 'resume_learner_checkpoint', 'out'):
                if plan.get(key):
                    assert (ROOT / plan[key]).exists(), (suite, key, plan[key])
            if job.get('checkpoint'):
                assert (ROOT / job['checkpoint']).is_file()
            if job.get('pressure_assets'):
                assert (ROOT / job['pressure_assets']['ensemble_directory']).is_dir()


def test_only_absolute_pressure_conversion_is_supported():
    from thermoresidual.evaluation import PressureFallback
    PressureFallback(offset=0.)
    with pytest.raises(ValueError):
        PressureFallback(offset=.101325)


def test_no_private_record_loaders_or_historical_training_entrypoints():
    from thermoresidual import pressure, site, foundation, simulation
    from thermoresidual.learning import SAC
    assert not hasattr(pressure, 'read_data')
    assert not hasattr(foundation, 'fit_water_exchanger')
    assert not hasattr(site.SiteScenario(), 'surface_version')
    assert not hasattr(simulation, 'main')
    assert not hasattr(SAC, 'warm_start')


@pytest.mark.parametrize('ratio,mode', [(r, m) for r in (.5, 2.) for m in ('rrl', 'sac')])
def test_weight_policy_interface_matches_frozen_protocol(monkeypatch, ratio, mode):
    monkeypatch.chdir(ROOT)
    from thermoresidual.simulation import make_episode
    from thermoresidual.evaluation import controller_for
    job = next(j for j in load_jobs('weights') if j['mode'] == mode and float(j['ratio']) == ratio)
    _, interface, cfg, _ = make_episode(job['plan'], 19, 1, 'weight_asset_check')
    controller = controller_for(job, interface, cfg)
    assert controller is not None


@pytest.mark.slow
def test_short_physical_balance_and_causal_observation(monkeypatch):
    monkeypatch.chdir(ROOT)
    from thermoresidual.simulation import make_episode, targets
    from thermoresidual.control import PhysicsController, evaluate_transition
    plan = load_jobs('demo')[0]['plan']
    plant, interface, _, _ = make_episode(plan, 99, 8, 'balance_test')
    refs = targets('standard', 0, 8, target_protocol=plan['target_protocol'])
    energy, throughput = 0., 0.
    for k in range(8):
        ctx = interface.receive(plant.measurement_packet('balance_test'), plant.time_s, refs[k], refs[k+1])
        assert len(ctx.observation) == 191
        result, _ = evaluate_transition(plant, interface, PhysicsController(), ctx, ctx.target)
        energy += abs(result['energy_residual_kJ'])
        throughput += result['energy_throughput_kJ']
        assert abs(result['mass_residual_kg']) < 1e-6
    assert energy / throughput < .01
