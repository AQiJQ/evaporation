"""Fast deterministic checks for the independent evaporator implementation."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import inspect
import numpy as np

from .config import ExperimentConfig
from .control import (
    SafeController,
    build_safety_design,
    estimate_hinf_norm,
    point_in_convex_polygon,
    project_qp_2d,
    spectral_radius,
)
from .model import EvaporatorModel
from .multiseed import DEFAULT_SEEDS, T_975_DF2, _moving_average
from .sac import ReplayBuffer, SACAgent, SACConfig
from .theta_learning import OnlineThetaLearner
from .train import (
    OBS_DIM,
    ZeroResidualPolicy,
    advance_disturbance,
    compute_safe_steady_reference,
    economic_policy_eligible,
    formal_safety_certified,
    hard_safety_passed,
    observation,
    run_episode,
)
from . import train as train_module


def main() -> None:
    cfg = ExperimentConfig(disturbance_bound_samples=50)
    assert cfg.episodes == 300
    assert len(DEFAULT_SEEDS) == 3 and 42 in DEFAULT_SEEDS
    assert np.isclose(T_975_DF2, 4.302652729911275)
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

    class ExpansionFallbackLearner(OnlineThetaLearner):
        def _candidate_region_diagnostics(self, scale, seed_design, attempt_index):
            feasible = float(scale) <= 1.10
            diagnostic = {
                "attempt": int(attempt_index),
                "scale": float(scale),
                "state_region": [],
                "input_region": [],
                "w_hull_vertex_count": len(seed_design.w_data_hull),
                "max_residual_norm": 0.0,
                "s_plus_z_contained": feasible,
                "feasible": feasible,
                "failure_reason": "" if feasible else "synthetic limit",
            }
            return (
                replace(seed_design, robust_region_scale=float(scale))
                if feasible else None,
                diagnostic,
            )

    fallback_cfg = replace(
        refinement_cfg,
        robust_region_growth_factor=1.25,
        robust_region_max_scale=1.25,
        robust_region_bisection_iterations=4,
    )
    fallback_learner = ExpansionFallbackLearner(
        fallback_cfg,
        refinement_model,
        refinement_initial,
        np.random.default_rng(900),
    )
    fallback_design, _ = fallback_learner.build_certified_robust_operating_design(
        refinement_initial
    )
    assert 1.0 <= fallback_design.robust_region_scale <= 1.10
    assert np.isclose(fallback_learner.first_infeasible_scale, 1.25)

    class InitialFailureLearner(ExpansionFallbackLearner):
        def _candidate_region_diagnostics(self, scale, seed_design, attempt_index):
            _, diagnostic = super()._candidate_region_diagnostics(
                1.25, seed_design, attempt_index
            )
            diagnostic["scale"] = float(scale)
            return None, diagnostic

    failed_as_required = False
    try:
        InitialFailureLearner(
            fallback_cfg,
            refinement_model,
            refinement_initial,
            np.random.default_rng(901),
        ).build_certified_robust_operating_design(refinement_initial)
    except RuntimeError as exc:
        failed_as_required = "scale=1.0" in str(exc)
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
    print(f"  last feasible scale: {refinement_learner.last_feasible_scale:.6g}")
    print(f"  first infeasible scale: {refinement_learner.first_infeasible_scale}")
    print("evaporation_safe_sac tests passed")


if __name__ == "__main__":
    main()
