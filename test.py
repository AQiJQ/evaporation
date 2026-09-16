"""Fast deterministic checks for the independent evaporator implementation."""
from __future__ import annotations

from itertools import product
import numpy as np

from .config import ExperimentConfig
from .control import (
    SafeController,
    build_safety_design,
    point_in_convex_polygon,
    project_qp_2d,
    spectral_radius,
)
from .model import EvaporatorModel
from .multiseed import DEFAULT_SEEDS, T_975_DF2
from .theta_learning import OnlineThetaLearner
from .train import OBS_DIM, observation


def main() -> None:
    cfg = ExperimentConfig(disturbance_bound_samples=50)
    assert cfg.episodes == 300
    assert len(DEFAULT_SEEDS) == 3 and 42 in DEFAULT_SEEDS
    assert np.isclose(T_975_DF2, 4.302652729911275)
    assert cfg.replay_capacity == 100000
    assert cfg.training_segment_steps == 200
    assert np.allclose(cfg.residual_action_scale, [0.12, 0.10])
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
    assert OBS_DIM == 17 and obs.shape == (17,)
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
        requested_residual = test_rng.uniform(-0.25, 0.25, 2)
        _, info = controller.act(state, requested_residual)
        assert info["qp_feasible"]
        assert 0.0 <= info["feasible_action_mapping_scale"] <= 1.0
        assert info["projection_gap"] < 1e-7
        assert np.allclose(
            info["candidate"] - info["base"],
            info["feasible_action_mapping_scale"] * requested_residual,
        )
        assert np.all(info["z_next"] >= design.invariant_lower - 1e-8)
        assert np.all(info["z_next"] <= design.invariant_upper + 1e-8)
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
    theta_cfg = ExperimentConfig(
        disturbance_bound_samples=20,
        theta_min_transition_samples=4,
        theta_batch_size=4,
        theta_set_update_every_episodes=999,
        theta_k_update_every_episodes=1,
    )
    theta_model = EvaporatorModel(theta_cfg)
    theta_design = build_safety_design(
        theta_cfg, theta_model, np.random.default_rng(theta_cfg.seed)
    )
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
    print("evaporation_safe_sac tests passed")


if __name__ == "__main__":
    main()
