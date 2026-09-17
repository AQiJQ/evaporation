"""Configuration for the independent evaporator H-infinity/RPI/QP/SAC study."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class ExperimentConfig:
    # Reproducibility and requested training length.
    seed: int = 42
    episodes: int = 500
    steps_per_episode: int = 300
    benchmark_profile: str = "default"
    dt_min: float = 0.20
    # ``proposed`` freezes an offline-optimized safety design during SAC.
    # ``joint_theta`` retains the original online h/p/M/K learner for ablation.
    experiment_mode: str = "proposed"

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
    # Immutable reference envelope used by the independent certification scan.
    # ``disturbance_half_range`` is replaced by rho_d_max times this vector for
    # the formal proposed training distribution.  rho_d scales only exogenous
    # [F1, X1, T1, T200] uncertainty, never the 2016 plant-state shocks.
    disturbance_full_half_range: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 4.0, 5.0], dtype=float)
    )
    disturbance_scale_scan_enabled: bool = True
    disturbance_scale_scan_grid: tuple[float, ...] = (
        0.0, 0.25, 0.5, 0.75, 1.0,
    )
    disturbance_scale_bisection_iterations: int = 8
    disturbance_scale_scan_seed: int = 420016
    disturbance_scale_random_samples: int = 2000
    # One independent offline M/K design pass is run for every rho_d before
    # robust-region certification.  The resulting gain is never warm-started
    # from a neighbouring rho_d or from an earlier scan artifact.
    disturbance_scale_static_outer_iterations: int = 1
    disturbance_scale_k_max_iterations: int = 8
    disturbance_scale_m_angle_max_degrees: float = 8.0
    disturbance_scale_m_angle_step_degrees: float = 4.0
    full_disturbance_stress_steps: int = 2000
    disturbance_mode: str = "piecewise_constant"
    disturbance_hold_steps: int = 50
    disturbance_estimate_ema: float = 0.8
    # The formal Zanon2016 main experiment uses only nominal exogenous
    # conditions and the paper's unscaled instantaneous plant-state shocks.
    # rho_d-scaled exogenous uncertainty is a separate disturbance_scan CLI.
    main_experiment_protocol: str = "default_exogenous_uncertainty"
    paper2016_training_scenarios: tuple[str, ...] = (
        "pressure_positive", "pressure_negative", "concentration_positive",
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
    rpi_series_max_terms: int = 5000
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
    restrict_invariant_to_local_window: bool = False
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
    theta_x_minus_z_area_weight: float = 0.5
    theta_invariant_area_weight: float = 0.2
    theta_static_k_max_iterations: int = 100
    theta_static_outer_iterations: int = 2
    robust_set_refinement_samples: int = 10000
    robust_set_refinement_max_iterations: int = 6
    robust_set_refinement_tolerance: float = 1e-3
    robust_region_initial_state_half_range: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 3.0], dtype=float)
    )
    robust_region_initial_input_half_range: np.ndarray = field(
        default_factory=lambda: np.array([35.0, 30.0], dtype=float)
    )
    robust_region_growth_factor: float = 1.25
    robust_region_max_scale: float = 4.0
    robust_region_bisection_iterations: int = 8
    robust_region_random_samples: int = 10000
    robust_region_membership_tolerance: float = 1e-8
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
    # The safe projection reserves this fraction of the declared normalized
    # residual box before the actor command is applied.  This prevents a
    # feasible v_base projection from landing on a QP facet and silently
    # collapsing all residual authority to zero.
    qp_min_residual_authority: float = 0.01
    residual_parameterization: str = "state_dependent_box"
    proposed_nominal_controller: str = "safe_center_tracking"
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
    sac_min_economic_improvement_percent: float = 1e-3
    # Zero means re-certify every periodic checkpoint under the same fixed design.
    sac_final_candidate_count: int = 0

    # Evaluation-only nominal safe steady-state reference. It never enters the
    # SAC reward or replay buffer.
    safe_reference_grid_points: int = 81
    safe_reference_derivative_tolerance: float = 1e-7

    # Cover most of the certified nominal controlled-invariant set during training. A
    # smaller evaluation fraction keeps all policy comparisons repeatable and
    # representative of normal operation.
    training_initial_radius_fraction: float = 0.90
    evaluation_initial_radius_fraction: float = 0.75
    # After self-consistent S-plus-Z refinement, proposed resets can use the
    # configured fractions of the complete certified invariant set.
    reset_within_disturbance_identification_window: bool = False
    training_boundary_start_probability: float = 0.30
    # Non-paper long rollouts can otherwise spend most samples near one steady
    # state. Treat every segment as a replay-terminal subtrajectory and safely
    # resample inside S; the Paper2016 main protocol does not use this reset.
    training_segment_steps: int = 200
    disturbance_adaptation_steps: int = 2000
    disturbance_adaptation_switch_steps: int = 250

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
        "evaporation_safe_sac/outputs_state_dependent_safe_sac_500x300/seed_42"
    )

    # Zanon, Gros & Diehl (2016) formal-comparison protocol.  The TuneMPC
    # example uses N=20 only as a reference/smoke-test setting; paper results
    # use a one-second sample, N=200 and a 300-second closed loop.
    paper2016_steady_state: np.ndarray = field(
        default_factory=lambda: np.array([25.0, 49.743], dtype=float)
    )
    paper2016_steady_input: np.ndarray = field(
        default_factory=lambda: np.array([191.713, 215.888], dtype=float)
    )
    paper2016_prediction_horizon: int = 200
    paper2016_smoke_horizon: int = 20
    paper2016_simulation_seconds: int = 300

    def __post_init__(self) -> None:
        if self.benchmark_profile == "zanon2016":
            # Model derivatives are expressed per minute, so 1 s = 1/60 min.
            self.dt_min = 1.0 / 60.0
            self.main_experiment_protocol = "paper2016_original_state_shocks"
            self.disturbance_half_range = np.zeros(4, dtype=float)
            # The certified Zanon2016 geometry supports a materially useful
            # residual box at every checked invariant vertex (the diagnosed
            # optimized design has >15% maximum authority).  Reserve 5% for
            # exploration instead of using the more conservative generic 1%.
            self.qp_min_residual_authority = 0.05
            # The robust tube is local to the interior safety anchor.  Keep the
            # paper economic steady state separately for Experiment I, while
            # eliminating a persistent affine mismatch caused by linearizing
            # the safety model at a different operating point.
            self.linearization_state = self.safe_center_state.copy()
            self.linearization_input = np.array(
                [222.62265879, 215.15014494], dtype=float
            )
            # The one-second discrete closed loop contains a deliberately slow
            # mode.  Its RPI support series needs more terms than the 12-second
            # default; control.py encloses its omitted infinite RPI tail with a
            # certified eigenbasis box bound.  Its unweighted discrete-time l2
            # gain also scales with the finer sampling grid.  Both quantities
            # are re-certified rather than reusing the default certificate.
            self.rpi_series_max_terms = 5000
            self.hinf_gamma = 3000.0
        elif self.benchmark_profile != "default":
            raise ValueError(
                "benchmark_profile must be 'default' or 'zanon2016'"
            )
