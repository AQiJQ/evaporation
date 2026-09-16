"""Configuration for the independent evaporator H-infinity/RPI/QP/SAC study."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class ExperimentConfig:
    # Reproducibility and requested training length.
    seed: int = 42
    episodes: int = 300
    steps_per_episode: int = 2000
    dt_min: float = 0.20

    # Wang-Cameron / Zanon-Gros benchmark constants.
    liquid_holdup: float = 20.0
    pressure_capacitance: float = 4.0
    recirculation_f3: float = 50.0
    cp: float = 0.07
    latent_evaporation: float = 38.5
    latent_steam: float = 36.6
    ua2: float = 6.84
    ua1_factor: float = 0.16
    t2_p_coeff: float = 0.5616
    t2_x_coeff: float = 0.3126
    t2_offset: float = 48.43
    t3_p_coeff: float = 0.507
    t3_offset: float = 55.0
    t100_p_coeff: float = 0.1538
    t100_offset: float = 90.0

    # d = [F1, X1, T1, T200], with the uncertainty ranges from safe-rpi2020.
    disturbance_nominal: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 5.0, 40.0, 25.0], dtype=float)
    )
    disturbance_half_range: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 4.0, 5.0], dtype=float)
    )

    # Paper constraints: (25,40) <= (X2,P2) <= (100,80), 100 <= u <= 400.
    state_lower: np.ndarray = field(
        default_factory=lambda: np.array([25.0, 40.0], dtype=float)
    )
    state_upper: np.ndarray = field(
        default_factory=lambda: np.array([100.0, 80.0], dtype=float)
    )
    input_lower: np.ndarray = field(
        default_factory=lambda: np.array([100.0, 100.0], dtype=float)
    )
    input_upper: np.ndarray = field(
        default_factory=lambda: np.array([400.0, 400.0], dtype=float)
    )

    # Nominal economic operating point, used for normalization and cost reporting.
    linearization_state: np.ndarray = field(
        default_factory=lambda: np.array([25.0, 49.74], dtype=float)
    )
    linearization_input: np.ndarray = field(
        default_factory=lambda: np.array([191.71, 215.89], dtype=float)
    )

    # Zanon-Gros (2020) evaporation terminal/safety center.  Its corresponding
    # input is computed as a steady input of the linearized nominal model (not
    # as a nonlinear plant equilibrium), matching the definition in the paper.
    safe_center_state: np.ndarray = field(
        default_factory=lambda: np.array([29.0, 53.57], dtype=float)
    )

    state_scale: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 20.0], dtype=float)
    )
    input_scale: np.ndarray = field(
        default_factory=lambda: np.array([100.0, 100.0], dtype=float)
    )

    # H-infinity dynamic-game weights. A larger second input penalty avoids an
    # unrealistically large F200 ancillary correction for this ill-conditioned plant.
    hinf_q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))
    hinf_r: np.ndarray = field(default_factory=lambda: np.diag([0.002, 0.25]))
    hinf_gamma: float = 500.0
    # These grids are retained only for diagnostics/backward compatibility.
    # The active design learns every entry of K continuously and checks the
    # resulting gain against the H-infinity/RPI/QP safety conditions.
    hinf_state_weight_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    hinf_input_weight_scales: tuple[float, ...] = (
        0.50, 0.75, 1.0, 1.50, 2.0, 3.0, 4.0,
    )
    rpi_inflation: float = 1.20
    disturbance_bound_samples: int = 8000
    invariant_set_margin: float = 0.995
    safety_design_state_half_range: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 3.0], dtype=float)
    )
    safety_design_input_half_range: np.ndarray = field(
        default_factory=lambda: np.array([35.0, 30.0], dtype=float)
    )
    invariant_nominal_half_range_physical: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=float)
    )
    theta_initial_safe_state_half_range: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5], dtype=float)
    )
    theta_initial_safe_input_half_range: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0], dtype=float)
    )

    # Paper-inspired slow safe-parameter learner.  W_theta has four facets and
    # is represented as {w | M w <= m}, with m fixed to one and M adapted.
    # h and p are the linear stage/terminal terms of a positive-definite
    # finite-horizon quadratic nominal value model.  SAC remains the fast
    # residual-action learner, while every theta proposal is safety-gated.
    theta_learning_enabled: bool = True
    theta_cost_learning_rate: float = 1e-2
    theta_cost_scale: float = 600.0
    theta_cost_update_every_episodes: int = 1
    theta_set_update_every_episodes: int = 5
    theta_min_transition_samples: int = 256
    theta_batch_size: int = 256
    theta_replay_capacity: int = 20000
    theta_parameter_clip: float = 2.0
    theta_m_angle_step_degrees: float = 2.0
    theta_m_max_angle_degrees: float = 35.0
    theta_w_online_inflation: float = 1.05
    # Feasible continuous H-infinity seed for the paper center under the full
    # disturbance envelope.  Online K updates are not restricted to a gain
    # grid: cyclic two-sided pattern search changes each of the four entries.
    theta_k_initial_gain: np.ndarray = field(
        default_factory=lambda: np.array([
            [-8.02154403, -1.67958159],
            [-0.30078081, 1.75387715],
        ], dtype=float)
    )
    theta_k_update_every_episodes: int = 5
    theta_k_initial_step: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.20, 0.08],
            [0.03, 0.05],
        ], dtype=float)
    )
    theta_k_min_step: float = 1e-3
    theta_k_max_step: float = 0.50
    theta_k_step_growth: float = 1.10
    theta_k_step_shrink: float = 0.70
    theta_k_acceptance_tolerance: float = 1e-5
    theta_k_evaluation_steps: int = 250
    theta_k_evaluation_seed_count: int = 2
    theta_rpi_area_weight: float = 1.0
    theta_x_minus_z_area_weight: float = 2e-3
    theta_invariant_area_weight: float = 1e-3
    nominal_mpc_horizon: int = 10
    nominal_mpc_stage_hessian: np.ndarray = field(
        default_factory=lambda: np.diag([1.0, 0.6, 0.02, 0.02])
    )
    nominal_mpc_terminal_hessian: np.ndarray = field(
        default_factory=lambda: np.diag([4.0, 2.0])
    )

    # Continuous Gaussian SAC settings aligned with the original CSTR-SAC.
    # Actor output a in [-1,1]^2 is mapped to the normalized residual below.
    residual_action_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.12, 0.10], dtype=float)
    )
    # The evaporation comparison in Zanon-Gros (2020) uses gamma=0.99.
    gamma_rl: float = 0.99
    tau: float = 0.005
    alpha_initial: float = 0.10
    alpha_min: float = 0.005
    entropy_tuning_warmup_updates: int = 2000
    auto_entropy_tuning: bool = True
    target_entropy: float = -2.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 256
    # A recent-window replay limits stale transitions produced under much older
    # online theta/K values while retaining 50 full episodes at the default run.
    replay_capacity: int = 100000
    warmup_steps: int = 5000
    updates_per_step: int = 1
    evaluation_every: int = 5
    evaluation_seed_count: int = 3
    holdout_seed_count: int = 20
    save_every: int = 25
    actor_ema_decay: float = 0.0

    # Periodic model selection is paired against the zero-residual controller
    # under the same final certified theta.  A one-sided scale search prevents
    # an over-aggressive residual actor from being deployed unchanged.
    sac_selection_scales: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    sac_min_incremental_return_per_step: float = 0.0
    # Re-certify the ten strongest common-seed periodic actors under final W/theta;
    # the zero-residual fallback and raw last actor are always added separately.
    sac_final_candidate_count: int = 10

    # Cover most of the certified nominal controlled-invariant set during training. A
    # smaller evaluation fraction keeps all policy comparisons repeatable and
    # representative of normal operation.
    training_initial_radius_fraction: float = 0.90
    evaluation_initial_radius_fraction: float = 0.75
    training_boundary_start_probability: float = 0.30
    # Long 2000-step episodes otherwise spend almost all samples near one
    # steady state.  Treat every segment as a replay-terminal subtrajectory and
    # safely resample inside S, while keeping the requested episode accounting.
    training_segment_steps: int = 200

    # Negative-square penalties for quantities whose desired value is zero.
    # The event terms remain separate because squaring a Boolean changes nothing.
    projection_penalty_weight: float = 5.0
    feasible_action_mapping_penalty_weight: float = 0.20
    input_move_penalty_weight: float = 0.010
    state_violation_event_penalty: float = 100.0
    input_violation_event_penalty: float = 100.0
    rpi_violation_event_penalty: float = 20.0
    qp_infeasible_event_penalty: float = 5.0
    state_excess_square_weight: float = 1000.0
    input_excess_square_weight: float = 1000.0
    rpi_excess_square_weight: float = 200.0
    qp_intervention_tolerance: float = 1e-3

    # The reward baseline is the fixed nominal economic point above, independent
    # of the terminal center, so paper-center and ablation returns are comparable.
    reward_cost_scale: float = 200.0

    # Keep hidden-disturbance results separate from earlier preview experiments.
    output_dir: Path = Path(
        "evaporation_safe_sac/outputs_state_dependent_safe_sac_300x2000/seed_42"
    )
