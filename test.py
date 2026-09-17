"""Fast deterministic checks for the independent evaporator implementation."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import inspect
import json
from pathlib import Path
import numpy as np

from .config import ExperimentConfig
from .control import (
    SafeController,
    _rpi_support,
    build_safety_design,
    estimate_hinf_norm,
    point_in_convex_polygon,
    project_qp_2d,
    spectral_radius,
)
from .model import EvaporatorModel
from .multiseed import DEFAULT_SEEDS, T_975_DF2, _moving_average
from .paper2016_adapter import (
    DEFAULT_TUNEMPC_PATH,
    dependency_diagnostics,
    reference_derivative,
    reference_economic_cost,
    reference_intermediate,
)
from .paper2016_compare import (
    PAPER2016_DISTURBANCE_PROTOCOL,
    SCENARIOS as PAPER2016_SCENARIOS,
    apply_state_shock,
    economic_metric_g,
    _rollout,
)
from .disturbance_experiments import (
    disturbance_corner_feasibility,
    run_disturbance_scale_scan,
)
from .sac import ReplayBuffer, SACAgent, SACConfig
from .theta_learning import OnlineThetaLearner
from .train import (
    OBS_DIM,
    ZeroResidualPolicy,
    advance_disturbance,
    balanced_random_paper2016_schedule,
    compute_safe_steady_reference,
    economic_policy_eligible,
    evaluate_paper_nominal_pressure_policy,
    formal_safety_certified,
    hard_safety_passed,
    observation,
    run_episode,
    summarize_training_trend,
)
from . import train as train_module
from . import paper2016_compare as paper2016_compare_module


def main() -> None:
    cfg = ExperimentConfig(disturbance_bound_samples=50)
    assert cfg.benchmark_profile == "default"
    assert np.isclose(cfg.dt_min, 0.20)
    paper_cfg = ExperimentConfig(
        benchmark_profile="zanon2016", disturbance_bound_samples=50
    )
    assert np.isclose(paper_cfg.dt_min, 1.0 / 60.0)
    assert paper_cfg.main_experiment_protocol == "paper2016_original_state_shocks"
    assert np.array_equal(paper_cfg.disturbance_half_range, np.zeros(4))
    assert paper_cfg.paper2016_training_scenarios == (
        "pressure_positive", "pressure_negative", "concentration_positive"
    )
    path_diagnostics = dependency_diagnostics(DEFAULT_TUNEMPC_PATH)
    assert path_diagnostics["reference_files_present"], path_diagnostics
    paper_model = EvaporatorModel(paper_cfg)
    consistency_state = np.array([28.4, 54.2])
    consistency_input = np.array([225.0, 240.0])
    current_flows = paper_model.algebraic(
        consistency_state, consistency_input, paper_cfg.disturbance_nominal
    )
    reference_flows = reference_intermediate(consistency_state, consistency_input)
    assert np.allclose(
        [current_flows.f2, current_flows.f4, current_flows.f5,
         current_flows.f100, current_flows.q100, current_flows.q200],
        [reference_flows["F2"], reference_flows["F4"], reference_flows["F5"],
         reference_flows["F100"], reference_flows["Q100"], reference_flows["Q200"]],
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        paper_model.derivative(
            consistency_state, consistency_input, paper_cfg.disturbance_nominal
        ),
        reference_derivative(consistency_state, consistency_input),
        rtol=1e-12, atol=1e-12,
    )
    assert np.isclose(
        paper_model.economic_cost(
            consistency_state, consistency_input, paper_cfg.disturbance_nominal
        ),
        reference_economic_cost(consistency_state, consistency_input),
        rtol=1e-12, atol=1e-12,
    )
    assert np.max(np.abs(paper_model.derivative(
        paper_cfg.paper2016_steady_state,
        paper_cfg.paper2016_steady_input,
        paper_cfg.disturbance_nominal,
    ))) < 2e-5
    nominal_state = paper_cfg.paper2016_steady_state
    for second in (0, 20, 40):
        assert np.allclose(
            apply_state_shock(nominal_state, "pressure_positive", second),
            nominal_state + [0.0, 1.0],
        )
        assert np.allclose(
            apply_state_shock(nominal_state, "pressure_negative", second),
            nominal_state + [0.0, -1.0],
        )
        assert np.allclose(
            apply_state_shock(nominal_state, "concentration_positive", second),
            nominal_state + [1.0, 0.0],
        )
    assert np.allclose(
        apply_state_shock(nominal_state, "pressure_positive", 1), nominal_state
    )
    assert PAPER2016_DISTURBANCE_PROTOCOL["uses_rho_d_scaling"] is False
    assert PAPER2016_DISTURBANCE_PROTOCOL["external_conditions"] == "nominal"
    assert PAPER2016_DISTURBANCE_PROTOCOL["state_shock_scaling"] == 1.0
    assert PAPER2016_DISTURBANCE_PROTOCOL["pressure_positive_shock_kpa"] == 1.0
    assert PAPER2016_DISTURBANCE_PROTOCOL["pressure_negative_shock_kpa"] == -1.0
    assert (
        PAPER2016_DISTURBANCE_PROTOCOL[
            "concentration_positive_shock_percentage_point"
        ] == 1.0
    )
    assert PAPER2016_DISTURBANCE_PROTOCOL["shock_times_seconds"] == [0, 20, 40]
    rho_one_half_range = 1.0 * paper_cfg.disturbance_full_half_range
    assert np.allclose(
        paper_cfg.disturbance_nominal - rho_one_half_range,
        [9.5, 4.5, 36.0, 20.0],
    )
    assert np.allclose(
        paper_cfg.disturbance_nominal + rho_one_half_range,
        [10.5, 5.5, 44.0, 30.0],
    )
    # rho_d scales only external [F1,X1,T1,T200] uncertainty.  It is not an
    # argument to the paper state-shock function and cannot change amplitude.
    for rho_d in (0.0, 0.158203125, 1.0):
        rho_cfg = replace(
            paper_cfg,
            disturbance_half_range=(
                rho_d * paper_cfg.disturbance_full_half_range
            ),
        )
        assert np.allclose(
            apply_state_shock(
                rho_cfg.paper2016_steady_state, "pressure_positive", 0
            ) - rho_cfg.paper2016_steady_state,
            [0.0, 1.0],
        )
        assert np.allclose(
            apply_state_shock(
                rho_cfg.paper2016_steady_state, "pressure_negative", 0
            ) - rho_cfg.paper2016_steady_state,
            [0.0, -1.0],
        )
        assert np.allclose(
            apply_state_shock(
                rho_cfg.paper2016_steady_state, "concentration_positive", 0
            ) - rho_cfg.paper2016_steady_state,
            [1.0, 0.0],
        )
    paper_compare_source = inspect.getsource(paper2016_compare_module)
    assert "run_disturbance_scale_scan" not in paper_compare_source
    training_source = inspect.getsource(train_module.main)
    assert "run_disturbance_scale_scan" not in training_source
    assert "run_full_disturbance_stress_test" not in training_source
    assert economic_metric_g(100.0, 100.0, 10, 2.0) == 0.0
    assert economic_metric_g(100.0, 110.0, 10, 2.0) < 0.0
    rollout_source = inspect.getsource(_rollout)
    assert ".update(" not in rollout_source
    nominal_evaluation_source = inspect.getsource(
        evaluate_paper_nominal_pressure_policy
    )
    assert "cfg.disturbance_nominal" in nominal_evaluation_source
    assert "second in (0, 20, 40)" in nominal_evaluation_source
    paper_rollout_source = inspect.getsource(_rollout)
    assert "cfg.disturbance_nominal" in paper_rollout_source

    class RecordingPaperModel:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.disturbances: list[np.ndarray] = []

        def economic_cost(self, state, control, disturbance):
            self.disturbances.append(np.asarray(disturbance, dtype=float).copy())
            return self.wrapped.economic_cost(state, control, disturbance)

        def step(self, state, control, disturbance):
            self.disturbances.append(np.asarray(disturbance, dtype=float).copy())
            return self.wrapped.step(state, control, disturbance)

    short_paper_cfg = replace(
        paper_cfg,
        paper2016_simulation_seconds=2,
        disturbance_half_range=paper_cfg.disturbance_full_half_range.copy(),
    )
    recording_model = RecordingPaperModel(EvaporatorModel(short_paper_cfg))
    _rollout(
        short_paper_cfg,
        recording_model,
        "pressure_positive",
        lambda state: (short_paper_cfg.paper2016_steady_input.copy(), {}),
    )
    assert recording_model.disturbances
    assert all(
        np.array_equal(value, short_paper_cfg.disturbance_nominal)
        for value in recording_model.disturbances
    )

    temporary_path = (
        Path(__file__).resolve().parent
        / "evaporation_safe_sac"
        / "test_disturbance_scan_tmp"
    )
    temporary_path.mkdir(parents=True, exist_ok=True)
    if True:
        corner_summary = disturbance_corner_feasibility(
            paper_cfg,
            paper_model,
            temporary_path / "disturbance_corner_feasibility.csv",
        )
        corner_rows = np.genfromtxt(
            temporary_path / "disturbance_corner_feasibility.csv",
            delimiter=",", names=True, dtype=None, encoding="utf-8",
        )
        assert len(corner_rows) == 16
        assert not corner_summary[
            "full_disturbance_all_corners_steady_feasible"
        ]
        assert corner_summary["max_required_F200"] > paper_cfg.input_upper[1]

        scan_cfg = ExperimentConfig(
            benchmark_profile="zanon2016",
            disturbance_scale_scan_grid=(0.0,),
            disturbance_scale_bisection_iterations=0,
            disturbance_scale_random_samples=50,
            disturbance_bound_samples=50,
            robust_region_random_samples=50,
            robust_region_bisection_iterations=1,
            theta_static_outer_iterations=0,
            disturbance_scale_static_outer_iterations=0,
            disturbance_scale_m_angle_max_degrees=0.0,
        )
        scan = run_disturbance_scale_scan(scan_cfg, temporary_path / "scan")
        assert scan.rho_d_max_certified == 0.0
        assert any(
            row["rho_d"] == 0.0
            and row["formal_certification_passed"]
            for row in scan.rows
        )
        assert np.allclose(
            scan.rho_d_max_certified * scan_cfg.disturbance_full_half_range,
            scan.selected_cfg.disturbance_half_range,
        )
        assert scan.selected_design.robust_region_scale > 0.0
        # A full-box failure is metadata for Experiment III and does not alter
        # the nominal paper model or prevent Experiment I state shocks.
        assert np.array_equal(
            scan.selected_model.cfg.disturbance_nominal,
            scan_cfg.disturbance_nominal,
        )
        with (temporary_path / "scan" / "disturbance_scale_summary.json").open(
            encoding="utf-8"
        ) as stream:
            robustness_summary = json.load(stream)
        assert robustness_summary["rho_d_max_certified"] == 0.0
        assert robustness_summary[
            "related_to_paper_state_shock_scaling"
        ] is False
        assert robustness_summary["disturbance_variables"] == [
            "F1", "X1", "T1", "T200"
        ]
        scan_header = (
            temporary_path / "scan" / "disturbance_scale_scan.csv"
        ).read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "rho_d" in scan_header and "alpha" not in scan_header

        paper_episode_cfg = replace(
            scan.selected_cfg,
            steps_per_episode=41,
            paper2016_simulation_seconds=41,
        )
        paper_stat, paper_records, _ = run_episode(
            paper_episode_cfg,
            scan.selected_model,
            SafeController(
                paper_episode_cfg, scan.selected_model, scan.selected_design
            ),
            ZeroResidualPolicy(),
            None,
            np.random.default_rng(2016),
            training=False,
            global_step=0,
            paper_scenario="pressure_positive",
        )
        assert len(paper_records) == 41
        assert np.isfinite(paper_stat["economic_cost_mean"])
        assert all(
            np.array_equal(record["disturbance"], paper_cfg.disturbance_nominal)
            for record in paper_records
        )
        assert all(
            record["paper2016_scenario"] == "pressure_positive"
            for record in paper_records
        )
        shock_seconds = [
            index for index, record in enumerate(paper_records)
            if np.any(record["paper_state_shock"] != 0.0)
        ]
        assert shock_seconds == [0, 20, 40]
        assert all(
            np.array_equal(paper_records[index]["paper_state_shock"], [0.0, 1.0])
            for index in shock_seconds
        )
        # A zero actor residual must pass the actual online verification QP,
        # even though the paper initial state lies outside the local nominal
        # invariant set.  reset() projects only z; the physical state remains
        # unchanged and is handled through the ancillary error feedback.
        assert paper_stat["qp_infeasible_rate"] == 0.0
        assert paper_stat["execution_mask_mean"] == 1.0
        assert paper_stat["residual_applied_norm_mean"] == 0.0
        assert all(record["execution_mask"] == 1.0 for record in paper_records)
        assert scan.selected_design.minimum_residual_authority >= (
            paper_episode_cfg.qp_min_residual_authority - 1e-8
        )
        paper_zero_controller = SafeController(
            paper_episode_cfg, scan.selected_model, scan.selected_design
        )
        paper_zero_controller.reset(paper_episode_cfg.paper2016_steady_state)
        _, zero_info = paper_zero_controller.act(
            paper_episode_cfg.paper2016_steady_state + [0.0, 1.0],
            np.zeros(2),
            action_is_normalized=True,
        )
        assert zero_info["reset_projection_norm"] > 0.0
        assert zero_info["base_qp_feasible"]
        assert zero_info["qp_feasible"]
        assert zero_info["reserved_residual_authority"] >= (
            paper_episode_cfg.qp_min_residual_authority - 1e-8
        )
        assert np.allclose(zero_info["nominal"], zero_info["base"])
        paper_nonzero_controller = SafeController(
            paper_episode_cfg, scan.selected_model, scan.selected_design
        )
        paper_nonzero_controller.reset(
            paper_episode_cfg.paper2016_steady_state
        )
        _, nonzero_info = paper_nonzero_controller.act(
            paper_episode_cfg.paper2016_steady_state + [0.0, 1.0],
            np.array([1.0, -1.0]),
            action_is_normalized=True,
        )
        assert nonzero_info["qp_feasible"]
        assert nonzero_info["feasible_action_mapping_scale"] >= (
            paper_episode_cfg.qp_min_residual_authority - 1e-8
        )
        assert np.linalg.norm(
            nonzero_info["nominal"] - nonzero_info["base"]
        ) > 0.0
    short_trend = summarize_training_trend(
        np.arange(10, dtype=float), np.array([0.0, 0.1])
    )
    assert short_trend["training_trend_status"] == "insufficient_episodes"
    assert short_trend["positive_training_trend"] is None
    assert cfg.episodes == 500
    assert cfg.steps_per_episode == 300
    assert paper_cfg.paper2016_simulation_seconds == 300
    paper_schedule = balanced_random_paper2016_schedule(500, 42)
    assert len(paper_schedule) == 500
    scenario_counts = {
        scenario: paper_schedule.count(scenario)
        for scenario in PAPER2016_SCENARIOS
    }
    assert max(scenario_counts.values()) - min(scenario_counts.values()) <= 1
    for start in range(0, 498, 3):
        assert set(paper_schedule[start:start + 3]) == set(
            PAPER2016_SCENARIOS
        )
    assert len(set(paper_schedule[498:])) == 2
    assert paper_schedule == balanced_random_paper2016_schedule(500, 42)
    assert len(DEFAULT_SEEDS) == 3 and 42 in DEFAULT_SEEDS
    assert np.isclose(T_975_DF2, 4.302652729911275)
    # A deliberately slow diagonal closed loop forces the finite-series path
    # to use its certified eigenbasis tail.  For a symmetric box the exact
    # coordinate support is known analytically.
    slow_acl = np.diag([0.999, 0.8])
    slow_w = np.asarray(list(product((-0.01, 0.01), (-0.02, 0.02))))
    slow_support = _rpi_support(
        slow_acl, slow_w, np.eye(2), max_terms=100
    )
    exact_slow_support = np.array([0.01 / (1.0 - 0.999), 0.02 / (1.0 - 0.8)])
    assert np.all(slow_support >= exact_slow_support - 1e-10)
    assert np.allclose(slow_support, exact_slow_support, rtol=1e-9, atol=1e-9)
    moving = _moving_average(np.arange(25, dtype=float), window=20)
    assert np.all(np.isnan(moving[:19])) and np.all(np.isfinite(moving[19:]))
    assert cfg.replay_capacity == 100000
    assert cfg.training_segment_steps == 200
    assert cfg.experiment_mode == "proposed"
    assert not cfg.restrict_invariant_to_local_window
    assert cfg.sac_final_candidate_count == 0
    assert cfg.disturbance_mode == "piecewise_constant"
    assert cfg.disturbance_hold_steps == 50
    assert cfg.residual_parameterization == "state_dependent_box"
    assert cfg.proposed_nominal_controller == "safe_center_tracking"
    assert not cfg.reset_within_disturbance_identification_window
    assert np.allclose(cfg.residual_action_scale, [0.12, 0.10])
    disturbance_rng = np.random.default_rng(991)
    held = cfg.disturbance_nominal.copy()
    for completed_steps in range(1, cfg.disturbance_hold_steps):
        advanced = advance_disturbance(
            cfg, disturbance_rng, held, completed_steps
        )
        assert np.array_equal(advanced, held)
    changed = advance_disturbance(
        cfg, disturbance_rng, held, cfg.disturbance_hold_steps
    )
    assert not np.array_equal(changed, held)
    assert np.all(changed >= cfg.disturbance_nominal - cfg.disturbance_half_range)
    assert np.all(changed <= cfg.disturbance_nominal + cfg.disturbance_half_range)
    model = EvaporatorModel(cfg)
    next_state = model.step(cfg.linearization_state, cfg.linearization_input, cfg.disturbance_nominal)
    assert np.max(np.abs(next_state - cfg.linearization_state)) < 2e-3, next_state
    safe_u = model.steady_input(cfg.safe_center_state)
    derivative = model.derivative(cfg.safe_center_state, safe_u, cfg.disturbance_nominal)
    assert np.max(np.abs(derivative)) < 1e-9, derivative
    a = np.vstack([np.eye(2), -np.eye(2)])
    b = np.ones(4)
    point, feasible = project_qp_2d(np.array([2.0, -3.0]), a, b)
    assert feasible and np.allclose(point, [1.0, -1.0]), point
    design = build_safety_design(cfg, model, np.random.default_rng(cfg.seed))
    assert design.minimum_residual_authority >= (
        cfg.qp_min_residual_authority - 1e-8
    )
    a_economic, b_economic, affine_economic = model.linearize(
        cfg.linearization_state, cfg.linearization_input
    )
    assert np.allclose(design.a, a_economic)
    assert np.allclose(design.b, b_economic)
    assert np.allclose(design.affine, affine_economic)
    paper_u = model.linearized_steady_input(
        cfg.safe_center_state, design.a, design.b, design.affine
    )
    assert np.allclose(paper_u, [223.76, 221.61], atol=0.1), paper_u
    paper_z = model.normalized_state(cfg.safe_center_state)
    paper_v = model.normalized_input(paper_u)
    assert np.allclose(
        design.a @ paper_z + design.b @ paper_v + design.affine,
        paper_z,
        atol=1e-10,
    )
    controller = SafeController(cfg, model, design)
    controller.reset(cfg.safe_center_state)
    obs = observation(
        model,
        controller,
        cfg.safe_center_state,
        design.v_ref,
    )
    assert OBS_DIM == 19 and obs.shape == (19,)
    assert np.allclose(obs[-2:], 0.0)
    supplied_w_est = np.array([0.25, -0.5]) * design.w_bound
    estimated_obs = observation(
        model, controller, cfg.safe_center_state, design.v_ref, supplied_w_est
    )
    assert np.allclose(estimated_obs[-2:], [0.25, -0.5])
    assert observation.__code__.co_varnames[:5] == (
        "model", "controller", "state", "previous_u", "disturbance_estimate"
    )
    controller.reset(cfg.safe_center_state)
    _, tracking_info = controller.act(
        cfg.safe_center_state, np.zeros(2), action_is_normalized=True
    )
    assert np.allclose(tracking_info["base"], design.v_ref, atol=1e-9)
    acl = design.a + design.b @ design.k
    assert spectral_radius(acl) < 1.0
    assert all(
        point_in_convex_polygon(vertex, design.w_vertices, tol=1e-8)
        for vertex in design.w_data_hull
    )
    assert np.all(
        design.theta_m_matrix @ design.w_vertices.T
        <= design.theta_m_bound[:, None] + 1e-8
    )
    assert design.theta_m_matrix.shape == (4, 2)
    assert design.theta_h.shape == (4,) and design.theta_p.shape == (2,)
    assert not np.allclose(
        np.max(design.w_vertices, axis=0),
        -np.min(design.w_vertices, axis=0),
    )
    assert np.all(design.x_upper_tight > design.x_lower_tight)
    assert np.all(design.invariant_lower >= design.x_lower_tight - 1e-10)
    assert np.all(design.invariant_upper <= design.x_upper_tight + 1e-10)
    invariant_lower_physical = model.physical_state(design.invariant_lower)
    invariant_upper_physical = model.physical_state(design.invariant_upper)
    assert bool(
        np.any(invariant_lower_physical < cfg.safe_center_state - 1.0 - 1e-6)
        or np.any(invariant_upper_physical > cfg.safe_center_state + 1.0 + 1e-6)
    )
    for vertex in product(*zip(design.invariant_lower, design.invariant_upper)):
        z = np.asarray(vertex)
        center_next = design.a @ z + design.affine
        aq = np.vstack([np.eye(2), -np.eye(2), design.b, -design.b])
        bq = np.concatenate([
            design.u_upper_tight,
            -design.u_lower_tight,
            design.invariant_upper - center_next,
            -design.invariant_lower + center_next,
        ])
        _, feasible = project_qp_2d(design.v_ref, aq, bq)
        assert feasible
    test_rng = np.random.default_rng(123)
    for _ in range(50):
        z = test_rng.uniform(design.invariant_lower, design.invariant_upper)
        state = model.physical_state(z)
        controller.reset(state)
        raw_action = test_rng.uniform(-1.0, 1.0, 2)
        _, info = controller.act(
            state, raw_action, action_is_normalized=True
        )
        assert info["qp_feasible"]
        assert 0.0 <= info["feasible_action_mapping_scale"] <= 1.0
        assert info["feasible_action_mapping_gap"] == 0.0
        assert info["projection_gap"] < 1e-7
        assert np.allclose(
            info["candidate"] - info["base"],
            info["feasible_action_mapping_scale"]
            * cfg.residual_action_scale * raw_action,
        )
        assert np.all(
            info["a_q"] @ info["candidate"] <= info["b_q"] + 1e-8
        )
        assert np.all(info["z_next"] >= design.invariant_lower - 1e-8)
        assert np.all(info["z_next"] <= design.invariant_upper + 1e-8)
        emergency_projection, emergency_feasible = project_qp_2d(
            np.array([1e6, -1e6]), info["a_q"], info["b_q"]
        )
        assert emergency_feasible
        assert np.all(
            info["a_q"] @ emergency_projection <= info["b_q"] + 1e-8
        )
    selected_boundary = design.rpi_boundary
    selected_physical = selected_boundary * cfg.state_scale
    selected_area = 0.5 * abs(float(np.sum(
        selected_physical[:, 0] * np.roll(selected_physical[:, 1], -1)
        - selected_physical[:, 1] * np.roll(selected_physical[:, 0], -1)
    )))
    selected_x_minus_z_area = float(np.prod(
        (design.x_upper_tight - design.x_lower_tight) * cfg.state_scale
    ))
    assert selected_area <= design.reference_rpi_area_physical + 1e-9
    # The densely sampled support half-space polygon is a safe outer
    # approximation and can exceed the exact box-zonotope area slightly.
    assert selected_area <= 1.02 * design.box_reference_rpi_area_physical
    assert np.all(
        (design.rpi_support_lower + design.rpi_support_upper) * cfg.state_scale
        <= design.box_reference_rpi_width_physical + 1e-9
    )
    assert (
        selected_x_minus_z_area
        >= design.reference_x_minus_z_area_physical - 1e-9
    )
    assert estimate_hinf_norm(
        design.a, design.b, design.k, cfg.hinf_q, cfg.hinf_r, points=128
    ) < cfg.hinf_gamma
    reference = compute_safe_steady_reference(
        ExperimentConfig(
            disturbance_bound_samples=50,
            safe_reference_grid_points=21,
        ),
        model,
        design,
    )
    assert np.isfinite(reference["cost"])
    assert np.all(reference["state"] >= cfg.state_lower - 1e-10)
    assert np.all(reference["state"] <= cfg.state_upper + 1e-10)
    assert np.all(reference["input"] >= cfg.input_lower - 1e-10)
    assert np.all(reference["input"] <= cfg.input_upper + 1e-10)

    episode_cfg = ExperimentConfig(
        disturbance_bound_samples=20,
        steps_per_episode=4,
    )
    episode_model = EvaporatorModel(episode_cfg)
    episode_design = build_safety_design(
        episode_cfg, episode_model, np.random.default_rng(11)
    )
    episode_stat, episode_records, _ = run_episode(
        episode_cfg,
        episode_model,
        SafeController(episode_cfg, episode_model, episode_design),
        ZeroResidualPolicy(),
        None,
        np.random.default_rng(12),
        training=False,
        global_step=0,
    )
    for reward_field in (
        "return", "return_per_step", "economic_reward_mean",
        "projection_penalty_mean", "mapping_penalty_mean",
        "move_penalty_mean", "safety_penalty_mean",
    ):
        assert reward_field in episode_stat
    assert np.all(np.isfinite(episode_records[0]["w_est"]))
    assert np.allclose(
        episode_records[0]["w_est"],
        (1.0 - episode_cfg.disturbance_estimate_ema)
        * episode_records[0]["w_hat"],
    )
    assert all(
        key in episode_stat for key in (
            "residual_feasible_scale_mean", "residual_feasible_scale_min",
            "residual_requested_norm_mean", "residual_applied_norm_mean",
            "residual_execution_ratio_mean",
        )
    )
    episode_source = inspect.getsource(run_episode)
    assert "(reward_reference_cost - cost) / cfg.reward_cost_scale" in episode_source
    assert "total_return += reward" in episode_source

    safe_stat = {
        "violation_rate": 0.0,
        "robust_operating_region_violation_rate": 0.0,
        "rpi_violation_rate": 0.0,
        "qp_infeasible_rate": 0.0,
        "disturbance_bound_exceedance_rate": 0.0,
        "economic_cost_mean": 9.0,
        "return_per_step": -100.0,
    }
    assert hard_safety_passed(safe_stat)
    assert economic_policy_eligible(safe_stat, 10.0, 1.0)
    eligibility_source = inspect.getsource(economic_policy_eligible)
    assert "return_per_step" not in eligibility_source
    assert "incremental_return" not in eligibility_source
    unsafe_stat = dict(safe_stat, disturbance_bound_exceedance_rate=0.01)
    assert not economic_policy_eligible(unsafe_stat, 10.0, 1.0)
    assert formal_safety_certified(0, safe_stat, safe_stat)
    assert not formal_safety_certified(1, safe_stat, safe_stat)
    main_source = inspect.getsource(train_module.main)
    selection_source = main_source[
        main_source.index("selected_candidate:"):
        main_source.index("if selected_candidate is None:")
    ]
    assert "holdout" not in selection_source
    theta_cfg = ExperimentConfig(
        disturbance_bound_samples=20,
        theta_min_transition_samples=4,
        theta_batch_size=4,
        theta_set_update_every_episodes=999,
        theta_k_update_every_episodes=1,
        theta_static_k_max_iterations=8,
        theta_static_outer_iterations=1,
        theta_m_max_angle_degrees=3.0,
    )
    theta_model = EvaporatorModel(theta_cfg)
    theta_design = build_safety_design(
        theta_cfg, theta_model, np.random.default_rng(theta_cfg.seed)
    )
    static_learner = OnlineThetaLearner(
        theta_cfg, theta_model, theta_design, np.random.default_rng(7)
    )
    optimized_design, _ = static_learner.optimize_static_safety_design(theta_design)
    optimized_rpi_area = 0.5 * abs(float(np.sum(
        (optimized_design.rpi_boundary * theta_cfg.state_scale)[:, 0]
        * np.roll((optimized_design.rpi_boundary * theta_cfg.state_scale)[:, 1], -1)
        - (optimized_design.rpi_boundary * theta_cfg.state_scale)[:, 1]
        * np.roll((optimized_design.rpi_boundary * theta_cfg.state_scale)[:, 0], -1)
    )))
    initial_rpi_area = 0.5 * abs(float(np.sum(
        (theta_design.rpi_boundary * theta_cfg.state_scale)[:, 0]
        * np.roll((theta_design.rpi_boundary * theta_cfg.state_scale)[:, 1], -1)
        - (theta_design.rpi_boundary * theta_cfg.state_scale)[:, 1]
        * np.roll((theta_design.rpi_boundary * theta_cfg.state_scale)[:, 0], -1)
    )))
    assert optimized_rpi_area <= initial_rpi_area + 1e-9
    assert float(np.prod(
        (optimized_design.x_upper_tight - optimized_design.x_lower_tight)
        * theta_cfg.state_scale
    )) >= float(np.prod(
        (theta_design.x_upper_tight - theta_design.x_lower_tight)
        * theta_cfg.state_scale
    )) - 1e-9
    assert spectral_radius(optimized_design.a + optimized_design.b @ optimized_design.k) < 1.0
    assert estimate_hinf_norm(
        optimized_design.a, optimized_design.b, optimized_design.k,
        theta_cfg.hinf_q, theta_cfg.hinf_r, points=128,
    ) < theta_cfg.hinf_gamma
    assert all(
        point_in_convex_polygon(vertex, optimized_design.w_vertices, tol=1e-8)
        for vertex in optimized_design.w_data_hull
    )
    refinement_cfg = ExperimentConfig(
        disturbance_bound_samples=20,
        robust_region_random_samples=30,
        robust_region_max_scale=1.25,
        robust_region_bisection_iterations=2,
        theta_static_k_max_iterations=2,
        theta_static_outer_iterations=1,
        theta_m_max_angle_degrees=0.0,
    )
    nonlinear_reference_model = EvaporatorModel(refinement_cfg)
    linear_a, linear_b, linear_affine = nonlinear_reference_model.linearize(
        refinement_cfg.linearization_state, refinement_cfg.linearization_input
    )

    class ExactLinearTestModel(EvaporatorModel):
        def linearize(self, state, control):
            del state, control
            return linear_a.copy(), linear_b.copy(), linear_affine.copy()

        def step(self, state, control, disturbance):
            del disturbance
            next_normalized = (
                linear_a @ self.normalized_state(state)
                + linear_b @ self.normalized_input(control)
                + linear_affine
            )
            return self.physical_state(next_normalized)

    refinement_model = ExactLinearTestModel(refinement_cfg)
    refinement_initial = build_safety_design(
        refinement_cfg,
        refinement_model,
        np.random.default_rng(refinement_cfg.seed),
    )
    refinement_learner = OnlineThetaLearner(
        refinement_cfg,
        refinement_model,
        refinement_initial,
        np.random.default_rng(808),
    )
    refinement_design, refinement_metrics = (
        refinement_learner.build_certified_robust_operating_design(
            refinement_initial
        )
    )
    assert refinement_metrics["robust_refinement_uncovered_count"] == 0.0
    assert refinement_design.robust_region_scale >= 1.0
    tol = refinement_cfg.robust_region_membership_tolerance
    assert np.all(
        refinement_design.invariant_lower
        >= refinement_design.x_lower_tight - tol
    )
    assert np.all(
        refinement_design.invariant_upper
        <= refinement_design.x_upper_tight + tol
    )
    assert np.all(
        refinement_design.invariant_lower
        - refinement_design.rpi_support_lower
        >= refinement_design.robust_state_lower - tol
    )
    assert np.all(
        refinement_design.invariant_upper
        + refinement_design.rpi_support_upper
        <= refinement_design.robust_state_upper + tol
    )
    assert np.allclose(
        refinement_design.u_lower_tight,
        refinement_design.robust_input_lower
        + refinement_design.input_rpi_support_lower,
    )
    assert np.allclose(
        refinement_design.u_upper_tight,
        refinement_design.robust_input_upper
        - refinement_design.input_rpi_support_upper,
    )
    region_rng = np.random.default_rng(1203)
    for _ in range(50):
        z = region_rng.uniform(
            refinement_design.invariant_lower,
            refinement_design.invariant_upper,
        )
        e = (
            region_rng.uniform(0.0, 1.0)
            * refinement_design.rpi_boundary[
                region_rng.integers(len(refinement_design.rpi_boundary))
            ]
        )
        actual_state = z + e
        assert np.all(
            actual_state >= refinement_design.robust_state_lower - tol
        )
        assert np.all(
            actual_state <= refinement_design.robust_state_upper + tol
        )
        nominal_input = region_rng.uniform(
            refinement_design.u_lower_tight,
            refinement_design.u_upper_tight,
        )
        actual_input = nominal_input + refinement_design.k @ e
        assert np.all(
            actual_input >= refinement_design.robust_input_lower - tol
        )
        assert np.all(
            actual_input <= refinement_design.robust_input_upper + tol
        )
    certified_residuals = refinement_learner._sample_robust_region_residuals(
        refinement_design,
        refinement_design.robust_state_lower,
        refinement_design.robust_state_upper,
        refinement_design.robust_input_lower,
        refinement_design.robust_input_upper,
        999,
    )
    assert all(
        point_in_convex_polygon(point, refinement_design.w_vertices, tol=1e-8)
        for point in certified_residuals
    )
    assert np.all(
        model.physical_state(refinement_design.robust_state_lower)
        >= refinement_cfg.state_lower - tol
    )
    assert np.all(
        model.physical_state(refinement_design.robust_state_upper)
        <= refinement_cfg.state_upper + tol
    )
    assert np.all(
        model.physical_input(refinement_design.robust_input_lower)
        >= refinement_cfg.input_lower - tol
    )
    assert np.all(
        model.physical_input(refinement_design.robust_input_upper)
        <= refinement_cfg.input_upper + tol
    )

    class FeasibleIntervalLearner(OnlineThetaLearner):
        def _candidate_region_diagnostics(self, scale, seed_design, attempt_index):
            feasible = 1.30 <= float(scale) <= 1.80
            diagnostic = {
                "attempt": int(attempt_index),
                "scale": float(scale),
                "state_region": [],
                "input_region": [],
                "w_hull_vertex_count": len(seed_design.w_data_hull),
                "max_residual_norm": 0.0,
                "hinf_feasible": feasible,
                "rpi_feasible": feasible,
                "x_tightening_feasible": feasible,
                "u_tightening_feasible": feasible,
                "invariant_feasible": feasible,
                "s_plus_z_contained": feasible,
                "input_tube_contained": feasible,
                "residual_membership_passed": feasible,
                "safe_center_feasible": feasible,
                "v_ref_feasible": feasible,
                "feasible": feasible,
                "failure_reason": "" if feasible else "outside synthetic interval",
            }
            return (
                replace(seed_design, robust_region_scale=float(scale))
                if feasible else None,
                diagnostic,
            )

    fallback_cfg = replace(
        refinement_cfg,
        robust_region_growth_factor=1.25,
        robust_region_max_scale=2.5,
        robust_region_bisection_iterations=4,
    )
    fallback_learner = FeasibleIntervalLearner(
        fallback_cfg,
        refinement_model,
        refinement_initial,
        np.random.default_rng(900),
    )
    fallback_design, _ = fallback_learner.build_certified_robust_operating_design(
        refinement_initial
    )
    assert fallback_learner.first_feasible_scale > 1.0
    assert np.isclose(fallback_learner.first_feasible_scale, 1.5625)
    assert np.isclose(fallback_learner.last_lower_infeasible_scale, 1.25)
    assert np.isclose(fallback_learner.first_upper_infeasible_scale, 1.953125)
    assert 1.30 <= fallback_learner.minimum_certified_scale <= 1.32
    assert 1.78 <= fallback_learner.maximum_certified_scale <= 1.80
    assert np.isclose(
        fallback_design.robust_region_scale,
        fallback_learner.maximum_certified_scale,
    )
    selected_row = [
        row for row in fallback_learner.robust_region_search_diagnostics
        if np.isclose(row["scale"], fallback_design.robust_region_scale)
    ][-1]
    assert selected_row["feasible"]
    assert np.all(
        fallback_design.invariant_lower
        - fallback_design.rpi_support_lower
        >= fallback_design.robust_state_lower - tol
    )
    assert np.all(
        fallback_design.invariant_upper
        + fallback_design.rpi_support_upper
        <= fallback_design.robust_state_upper + tol
    )
    assert np.all(
        fallback_design.u_lower_tight
        - fallback_design.input_rpi_support_lower
        >= fallback_design.robust_input_lower - tol
    )
    assert np.all(
        fallback_design.u_upper_tight
        + fallback_design.input_rpi_support_upper
        <= fallback_design.robust_input_upper + tol
    )

    class AllScalesInfeasibleLearner(FeasibleIntervalLearner):
        def _candidate_region_diagnostics(self, scale, seed_design, attempt_index):
            _, diagnostic = super()._candidate_region_diagnostics(
                1.0, seed_design, attempt_index
            )
            diagnostic["scale"] = float(scale)
            return None, diagnostic

    failed_as_required = False
    try:
        AllScalesInfeasibleLearner(
            fallback_cfg,
            refinement_model,
            refinement_initial,
            np.random.default_rng(901),
        ).build_certified_robust_operating_design(refinement_initial)
    except RuntimeError as exc:
        message = str(exc)
        failed_as_required = (
            "No feasible robust operating region" in message
            and "scale=1" in message
            and "scale=2.5" in message
        )
    assert failed_as_required
    learner = OnlineThetaLearner(
        theta_cfg, theta_model, theta_design, np.random.default_rng(7)
    )
    theta_state = theta_cfg.safe_center_state.copy()
    theta_input = theta_model.linearized_steady_input(
        theta_state, theta_design.a, theta_design.b, theta_design.affine
    )
    for _ in range(4):
        theta_next = theta_model.step(
            theta_state, theta_input, theta_cfg.disturbance_nominal
        )
        learner.observe_transition(
            theta_state,
            theta_input,
            theta_next,
            theta_model.economic_cost(
                theta_next, theta_input, theta_cfg.disturbance_nominal
            ),
            theta_design,
        )
        theta_state = theta_next
    learned_design, theta_metrics = learner.update_after_episode(
        101,
        theta_design,
        design_evaluator=lambda candidate: float(candidate.k[0, 0]),
    )
    assert theta_metrics["theta_cost_updates"] == 1.0
    assert np.all(np.isfinite(learned_design.theta_h))
    assert np.all(np.isfinite(learned_design.theta_p))
    assert theta_metrics["theta_K_update_attempts"] == 1.0
    assert theta_metrics["theta_K_update_accepts"] == 1.0
    assert not np.allclose(learned_design.k, theta_design.k)

    sac_agent = SACAgent(
        obs_dim=3,
        action_dim=2,
        cfg=SACConfig(
            hidden_dim=16,
            auto_entropy_tuning=True,
            entropy_tuning_warmup_updates=0,
        ),
        device="cpu",
    )
    sac_replay = ReplayBuffer(3, 2, 64, sac_agent.device)
    sac_rng = np.random.default_rng(99)
    for _ in range(32):
        sac_replay.add(
            sac_rng.normal(size=3),
            np.tanh(sac_rng.normal(size=2)),
            float(sac_rng.normal()),
            sac_rng.normal(size=3),
            False,
            execution_mask=1.0,
        )
    q1_before = [parameter.detach().clone() for parameter in sac_agent.q1.parameters()]
    q2_before = [parameter.detach().clone() for parameter in sac_agent.q2.parameters()]
    actor_before = [parameter.detach().clone() for parameter in sac_agent.actor.parameters()]
    alpha_before = sac_agent.alpha
    losses = sac_agent.update(sac_replay, 16)
    assert all(np.isfinite(losses[key]) for key in ("q1_loss", "q2_loss", "actor_loss", "alpha"))
    assert any(
        not np.allclose(before.numpy(), after.detach().numpy())
        for before, after in zip(q1_before, sac_agent.q1.parameters())
    )
    assert any(
        not np.allclose(before.numpy(), after.detach().numpy())
        for before, after in zip(q2_before, sac_agent.q2.parameters())
    )
    assert any(
        not np.allclose(before.numpy(), after.detach().numpy())
        for before, after in zip(actor_before, sac_agent.actor.parameters())
    )
    assert not np.isclose(sac_agent.alpha, alpha_before)
    rpi_area = 0.5 * abs(float(np.sum(
        (refinement_design.rpi_boundary * refinement_cfg.state_scale)[:, 0]
        * np.roll(
            (refinement_design.rpi_boundary * refinement_cfg.state_scale)[:, 1],
            -1,
        )
        - (refinement_design.rpi_boundary * refinement_cfg.state_scale)[:, 1]
        * np.roll(
            (refinement_design.rpi_boundary * refinement_cfg.state_scale)[:, 0],
            -1,
        )
    )))
    print("offline robust design diagnostic")
    print(f"  robust region scale: {refinement_design.robust_region_scale:.6g}")
    print(
        "  X_R:",
        refinement_model.physical_state(refinement_design.robust_state_lower),
        refinement_model.physical_state(refinement_design.robust_state_upper),
    )
    print(
        "  U_R:",
        refinement_model.physical_input(refinement_design.robust_input_lower),
        refinement_model.physical_input(refinement_design.robust_input_upper),
    )
    print(f"  W hull vertices: {len(refinement_design.w_data_hull)}")
    print(f"  RPI area: {rpi_area:.6g}")
    print(
        "  X_R-minus-Z area:",
        float(np.prod(
            (refinement_design.x_upper_tight - refinement_design.x_lower_tight)
            * refinement_cfg.state_scale
        )),
    )
    print(
        "  invariant area:",
        float(np.prod(
            (refinement_design.invariant_upper - refinement_design.invariant_lower)
            * refinement_cfg.state_scale
        )),
    )
    print(f"  first feasible scale: {refinement_learner.first_feasible_scale:.6g}")
    print(
        "  minimum certified scale: "
        f"{refinement_learner.minimum_certified_scale:.6g}"
    )
    print(
        "  maximum certified scale: "
        f"{refinement_learner.maximum_certified_scale:.6g}"
    )
    print(
        "  last lower infeasible scale: "
        f"{refinement_learner.last_lower_infeasible_scale}"
    )
    print(
        "  first upper infeasible scale: "
        f"{refinement_learner.first_upper_infeasible_scale}"
    )
    print("evaporation_safe_sac tests passed")


if __name__ == "__main__":
    main()
