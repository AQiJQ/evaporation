"""Slow, safety-gated learning of theta=(h, p, M, K).

The fast learner remains SAC.  This module follows the data path of Zanon and
Gros (2020): state transitions produce affine-model residuals, their convex
hull is the sample-based SDC, and W_theta={w | M w <= m} must contain every
retained residual.  Here m is fixed to one, while the four rows of M rotate
and rescale.  Candidate M/K updates are accepted only when a complete RPI/QP
safety design can be rebuilt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import product
from typing import Callable, Deque

import numpy as np

from .config import ExperimentConfig
from .control import (
    SafetyDesign,
    _convex_hull,
    build_safety_design,
    finite_horizon_nominal_policy,
    point_in_convex_polygon,
    spectral_radius,
    synthesize_hinf,
)
from .model import EvaporatorModel


def _polygon_area(vertices: np.ndarray, scale: np.ndarray) -> float:
    physical = np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)
    return 0.5 * abs(float(np.sum(
        physical[:, 0] * np.roll(physical[:, 1], -1)
        - physical[:, 1] * np.roll(physical[:, 0], -1)
    )))


class OnlineThetaLearner:
    """Two-time-scale parameter learner with an explicit safe acceptance gate."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: EvaporatorModel,
        initial_design: SafetyDesign,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.rng = rng
        self.transitions: Deque[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = deque(
            maxlen=int(cfg.theta_replay_capacity)
        )
        self.data_hull = np.asarray(initial_design.w_data_hull, dtype=float).copy()
        self.initial_k = np.asarray(initial_design.k, dtype=float).copy()
        self.initial_rpi_area = _polygon_area(
            initial_design.rpi_boundary, cfg.state_scale
        )
        self.initial_x_minus_z_area = float(np.prod(
            (initial_design.x_upper_tight - initial_design.x_lower_tight)
            * cfg.state_scale
        ))
        self.initial_invariant_area = float(np.prod(
            (initial_design.invariant_upper - initial_design.invariant_lower)
            * cfg.state_scale
        ))
        self.set_update_attempts = 0
        self.set_update_accepts = 0
        self.k_update_attempts = 0
        self.k_update_accepts = 0
        self.k_steps = np.asarray(cfg.theta_k_initial_step, dtype=float).copy()
        self.last_k_score = float("nan")
        self.last_k_coordinate = -1
        self.cost_updates = 0
        self.membership_exceedances = 0
        self.last_td_error = float("nan")
        self.refinement_iterations = 0
        self.refinement_uncovered_count = 0
        self.refinement_max_excess = 0.0
        self.refinement_relative_rpi_change = float("nan")
        self.robust_region_search_diagnostics: list[dict[str, object]] = []
        self.first_feasible_scale = float("nan")
        self.minimum_certified_scale = float("nan")
        self.maximum_certified_scale = float("nan")
        self.last_lower_infeasible_scale = float("nan")
        self.first_upper_infeasible_scale = float("nan")
        # Backward-compatible aliases retained in existing metrics.
        self.last_feasible_scale = float("nan")
        self.first_infeasible_scale = float("nan")
        self.robust_region_max_sample_residual_norm = float("nan")
        self.self_consistency_s_plus_z_passed = False

    def observe_transition(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        economic_cost: float,
        design: SafetyDesign,
    ) -> None:
        x = self.model.normalized_state(state)
        u = self.model.normalized_input(control)
        x_next = self.model.normalized_state(next_state)
        residual = x_next - (design.a @ x + design.b @ u + design.affine)
        self.transitions.append((x.copy(), u.copy(), x_next.copy(), float(economic_cost)))
        if not point_in_convex_polygon(residual, self.data_hull, tol=1e-10):
            self.membership_exceedances += 1
            self.data_hull = _convex_hull(np.vstack([
                self.data_hull, residual[None, :], np.zeros((1, 2))
            ]))

    def _q_value(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        h: np.ndarray,
        p: np.ndarray,
    ) -> float:
        y = np.concatenate([x, u])
        stage = 0.5 * float(y @ self.cfg.nominal_mpc_stage_hessian @ y) + float(h @ y)
        terminal = 0.5 * float(
            x_next @ self.cfg.nominal_mpc_terminal_hessian @ x_next
        ) + float(p @ x_next)
        return stage + self.cfg.gamma_rl * terminal

    def _update_h_p(self, design: SafetyDesign) -> SafetyDesign:
        if len(self.transitions) < int(self.cfg.theta_min_transition_samples):
            return design
        batch_size = min(int(self.cfg.theta_batch_size), len(self.transitions))
        indices = self.rng.integers(0, len(self.transitions), size=batch_size)
        samples = list(self.transitions)
        gradient_h = np.zeros(4, dtype=float)
        gradient_p = np.zeros(2, dtype=float)
        td_errors = []
        reference_cost = self.model.economic_cost(
            self.cfg.linearization_state,
            self.cfg.linearization_input,
            self.cfg.disturbance_nominal,
        )
        for index in indices:
            x, u, x_next, cost = samples[int(index)]
            next_u = design.nominal_policy_gain @ x_next + design.nominal_policy_offset
            predicted_after_next = (
                design.a @ x_next + design.b @ next_u + design.affine
            )
            q_current = self._q_value(
                x, u, x_next, design.theta_h, design.theta_p
            )
            q_next = self._q_value(
                x_next,
                next_u,
                predicted_after_next,
                design.theta_h,
                design.theta_p,
            )
            normalized_cost = (cost - reference_cost) / self.cfg.theta_cost_scale
            td_error = normalized_cost + self.cfg.gamma_rl * q_next - q_current
            # Semi-gradient Q-learning: the bootstrap target is held fixed.
            gradient_h += td_error * np.concatenate([x, u])
            gradient_p += td_error * self.cfg.gamma_rl * x_next
            td_errors.append(td_error)
        rate = float(self.cfg.theta_cost_learning_rate) / batch_size
        clip = float(self.cfg.theta_parameter_clip)
        h = np.clip(design.theta_h + rate * gradient_h, -clip, clip)
        p = np.clip(design.theta_p + rate * gradient_p, -clip, clip)
        nominal_gain, nominal_offset = finite_horizon_nominal_policy(
            self.cfg, design.a, design.b, design.affine, h, p
        )
        self.cost_updates += 1
        self.last_td_error = float(np.mean(np.abs(td_errors)))
        return replace(
            design,
            theta_h=h,
            theta_p=p,
            nominal_policy_gain=nominal_gain,
            nominal_policy_offset=nominal_offset,
        )

    def _design_score(self, design: SafetyDesign) -> float:
        rpi_area = _polygon_area(design.rpi_boundary, self.cfg.state_scale)
        x_minus_z_area = float(np.prod(
            (design.x_upper_tight - design.x_lower_tight) * self.cfg.state_scale
        ))
        invariant_area = float(np.prod(
            (design.invariant_upper - design.invariant_lower) * self.cfg.state_scale
        ))
        return (
            self.cfg.theta_rpi_area_weight
            * rpi_area / max(self.initial_rpi_area, 1e-12)
            - self.cfg.theta_x_minus_z_area_weight
            * x_minus_z_area / max(self.initial_x_minus_z_area, 1e-12)
            - self.cfg.theta_invariant_area_weight
            * invariant_area / max(self.initial_invariant_area, 1e-12)
        )

    @staticmethod
    def _region_kwargs(design: SafetyDesign) -> dict[str, object]:
        """Propagate the certified X_R/U_R through every M/K rebuild."""
        return {
            "robust_state_lower": design.robust_state_lower,
            "robust_state_upper": design.robust_state_upper,
            "robust_input_lower": design.robust_input_lower,
            "robust_input_upper": design.robust_input_upper,
            "robust_region_scale": design.robust_region_scale,
        }

    def _static_candidate_admissible(self, design: SafetyDesign) -> bool:
        """Keep the optimized geometry Pareto-safe relative to its seed."""
        rpi_area = _polygon_area(design.rpi_boundary, self.cfg.state_scale)
        x_minus_z_area = float(np.prod(
            (design.x_upper_tight - design.x_lower_tight) * self.cfg.state_scale
        ))
        return bool(
            rpi_area <= self.initial_rpi_area + 1e-10
            and x_minus_z_area >= self.initial_x_minus_z_area - 1e-10
        )

    def _static_angle_search(self, design: SafetyDesign, pass_index: int) -> SafetyDesign:
        """Exhaustively scan the configured four-facet orientation range."""
        limit = float(self.cfg.theta_m_max_angle_degrees)
        step = float(self.cfg.theta_m_angle_step_degrees)
        angles = np.deg2rad(np.arange(-limit, limit + 0.5 * step, step))
        best = design
        best_score = self._design_score(design)
        for index, angle in enumerate(angles):
            self.set_update_attempts += 1
            try:
                candidate = build_safety_design(
                    self.cfg,
                    self.model,
                    np.random.default_rng(
                        self.cfg.seed + 61000 + 1000 * pass_index + index
                    ),
                    w_data_hull=self.data_hull,
                    theta_m_angle=float(angle),
                    theta_h=design.theta_h,
                    theta_p=design.theta_p,
                    theta_k=design.k,
                    w_inflation=self.cfg.rpi_inflation,
                    **self._region_kwargs(design),
                )
            except RuntimeError:
                continue
            if not all(
                point_in_convex_polygon(point, candidate.w_vertices, tol=1e-8)
                for point in self.data_hull
            ):
                continue
            if not self._static_candidate_admissible(candidate):
                continue
            score = self._design_score(candidate)
            if score < best_score - float(self.cfg.theta_k_acceptance_tolerance):
                best, best_score = candidate, score
        if best is not design:
            self.set_update_accepts += 1
        return best

    def _static_k_search(self, design: SafetyDesign, pass_index: int) -> SafetyDesign:
        """Safety-gated two-sided coordinate search using geometry only."""
        steps = np.asarray(self.cfg.theta_k_initial_step, dtype=float).copy()
        best = design
        max_iterations = int(self.cfg.theta_static_k_max_iterations)
        tolerance = float(self.cfg.theta_k_acceptance_tolerance)
        for iteration in range(max_iterations):
            if np.all(steps <= float(self.cfg.theta_k_min_step) + 1e-15):
                break
            coordinate = iteration % best.k.size
            self.k_update_attempts += 1
            self.last_k_coordinate = coordinate
            current_score = self._design_score(best)
            proposals: list[tuple[float, SafetyDesign]] = []
            for sign_index, sign in enumerate((-1.0, 1.0)):
                proposed_k = np.asarray(best.k, dtype=float).copy()
                proposed_k.flat[coordinate] += sign * float(steps.flat[coordinate])
                try:
                    candidate = build_safety_design(
                        self.cfg,
                        self.model,
                        np.random.default_rng(
                            self.cfg.seed + 71000 + 10000 * pass_index
                            + 10 * iteration + sign_index
                        ),
                        w_data_hull=self.data_hull,
                        theta_m_angle=best.theta_m_angle,
                        theta_h=best.theta_h,
                        theta_p=best.theta_p,
                        theta_k=proposed_k,
                        w_inflation=self.cfg.rpi_inflation,
                        **self._region_kwargs(best),
                    )
                except RuntimeError:
                    continue
                if all(
                    point_in_convex_polygon(point, candidate.w_vertices, tol=1e-8)
                    for point in self.data_hull
                ) and self._static_candidate_admissible(candidate):
                    proposals.append((self._design_score(candidate), candidate))
            score, candidate = min(
                proposals, key=lambda item: item[0],
                default=(float("inf"), best),
            )
            if score < current_score - tolerance:
                best = candidate
                self.k_update_accepts += 1
                self.last_k_score = -score
                steps.flat[coordinate] = min(
                    float(self.cfg.theta_k_max_step),
                    float(steps.flat[coordinate])
                    * float(self.cfg.theta_k_step_growth),
                )
            else:
                self.last_k_score = -current_score
                steps.flat[coordinate] = max(
                    float(self.cfg.theta_k_min_step),
                    float(steps.flat[coordinate])
                    * float(self.cfg.theta_k_step_shrink),
                )
        self.k_steps = steps
        return best

    def optimize_static_safety_design(
        self, initial_design: SafetyDesign
    ) -> tuple[SafetyDesign, dict[str, float]]:
        """Optimize M and K before SAC without using rollout return."""
        design = initial_design
        for pass_index in range(int(self.cfg.theta_static_outer_iterations)):
            design = self._static_angle_search(design, pass_index)
            design = self._static_k_search(design, pass_index)
        return design, self.metrics(design)

    def _robust_region_bounds(
        self, design: SafetyDesign, scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized X_R/U_R bounds clipped by physical constraints."""
        safe_input = self.model.physical_input(design.v_ref)
        state_lower = np.maximum(
            self.cfg.state_lower,
            self.cfg.safe_center_state
            - float(scale) * self.cfg.robust_region_initial_state_half_range,
        )
        state_upper = np.minimum(
            self.cfg.state_upper,
            self.cfg.safe_center_state
            + float(scale) * self.cfg.robust_region_initial_state_half_range,
        )
        input_lower = np.maximum(
            self.cfg.input_lower,
            safe_input
            - float(scale) * self.cfg.robust_region_initial_input_half_range,
        )
        input_upper = np.minimum(
            self.cfg.input_upper,
            safe_input
            + float(scale) * self.cfg.robust_region_initial_input_half_range,
        )
        return (
            self.model.normalized_state(state_lower),
            self.model.normalized_state(state_upper),
            self.model.normalized_input(input_lower),
            self.model.normalized_input(input_upper),
        )

    def _sample_robust_region_residuals(
        self,
        design: SafetyDesign,
        state_lower: np.ndarray,
        state_upper: np.ndarray,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
        attempt_index: int,
    ) -> np.ndarray:
        """Certify mismatch only on the declared X_R x U_R x D domain."""
        disturbance_lower = (
            self.cfg.disturbance_nominal - self.cfg.disturbance_half_range
        )
        disturbance_upper = (
            self.cfg.disturbance_nominal + self.cfg.disturbance_half_range
        )
        residuals: list[np.ndarray] = []

        def observe(x: np.ndarray, u: np.ndarray, disturbance: np.ndarray) -> None:
            next_state = self.model.normalized_state(self.model.step(
                self.model.physical_state(x),
                self.model.physical_input(u),
                disturbance,
            ))
            residuals.append(
                next_state - (design.a @ x + design.b @ u + design.affine)
            )

        for state in product(*zip(state_lower, state_upper)):
            for control in product(*zip(input_lower, input_upper)):
                for disturbance in product(
                    *zip(disturbance_lower, disturbance_upper)
                ):
                    observe(
                        np.asarray(state, dtype=float),
                        np.asarray(control, dtype=float),
                        np.asarray(disturbance, dtype=float),
                    )

        rng = np.random.default_rng(
            self.cfg.seed + 81000 + int(attempt_index)
        )
        for _ in range(int(self.cfg.robust_region_random_samples)):
            observe(
                rng.uniform(state_lower, state_upper),
                rng.uniform(input_lower, input_upper),
                rng.uniform(disturbance_lower, disturbance_upper),
            )
        return np.asarray(residuals, dtype=float)

    def _candidate_region_diagnostics(
        self,
        scale: float,
        seed_design: SafetyDesign,
        attempt_index: int,
    ) -> tuple[SafetyDesign | None, dict[str, object]]:
        """Build and fully self-check one candidate robust operating region."""
        tolerance = float(self.cfg.robust_region_membership_tolerance)
        state_lower, state_upper, input_lower, input_upper = (
            self._robust_region_bounds(seed_design, scale)
        )
        residuals = self._sample_robust_region_residuals(
            seed_design,
            state_lower,
            state_upper,
            input_lower,
            input_upper,
            attempt_index,
        )
        data_hull = _convex_hull(np.vstack([
            np.zeros((1, 2)), residuals
        ]))
        diagnostic: dict[str, object] = {
            "attempt": int(attempt_index),
            "scale": float(scale),
            "state_region": self.model.physical_state(
                np.vstack([state_lower, state_upper])
            ).tolist(),
            "input_region": self.model.physical_input(
                np.vstack([input_lower, input_upper])
            ).tolist(),
            "w_hull_vertex_count": int(len(data_hull)),
            "max_residual_norm": float(np.max(np.linalg.norm(residuals, axis=1))),
            "hinf_feasible": False,
            "rpi_feasible": False,
            "x_tightening_feasible": False,
            "u_tightening_feasible": False,
            "invariant_feasible": False,
            "s_plus_z_contained": False,
            "input_tube_contained": False,
            "residual_membership_passed": False,
            "safe_center_feasible": False,
            "v_ref_feasible": False,
            "verification_qp_feasible": False,
            "residual_authority_feasible": False,
            "minimum_residual_authority": 0.0,
            "feasible": False,
            "failure_reason": "no feasible M/K seed",
        }
        angle_limit = float(self.cfg.theta_m_max_angle_degrees)
        angle_step = float(self.cfg.theta_m_angle_step_degrees)
        angles = [float(seed_design.theta_m_angle)] + [
            float(value) for value in np.deg2rad(np.arange(
                -angle_limit, angle_limit + 0.5 * angle_step, angle_step
            ))
            if abs(float(value) - float(seed_design.theta_m_angle)) > 1e-12
        ]
        gain_seeds = [np.asarray(seed_design.k, dtype=float)]
        configured_gain = np.asarray(self.cfg.theta_k_initial_gain, dtype=float)
        if not np.allclose(configured_gain, gain_seeds[0]):
            gain_seeds.append(configured_gain)
        try:
            synthesized_gain, _, _ = synthesize_hinf(
                self.cfg, seed_design.a, seed_design.b
            )
        except RuntimeError:
            synthesized_gain = None
        if synthesized_gain is not None and not any(
            np.allclose(synthesized_gain, gain) for gain in gain_seeds
        ):
            gain_seeds.append(synthesized_gain)

        rebuilt: SafetyDesign | None = None
        for gain_index, gain in enumerate(gain_seeds):
            for angle_index, angle in enumerate(angles):
                gate: dict[str, object] = {}
                try:
                    candidate = build_safety_design(
                        self.cfg,
                        self.model,
                        np.random.default_rng(
                            self.cfg.seed + 82000 + 1000 * attempt_index
                            + 100 * gain_index + angle_index
                        ),
                        w_data_hull=data_hull,
                        theta_m_angle=angle,
                        theta_h=seed_design.theta_h,
                        theta_p=seed_design.theta_p,
                        theta_k=gain,
                        w_inflation=self.cfg.rpi_inflation,
                        robust_state_lower=state_lower,
                        robust_state_upper=state_upper,
                        robust_input_lower=input_lower,
                        robust_input_upper=input_upper,
                        robust_region_scale=float(scale),
                        diagnostics=gate,
                    )
                except RuntimeError:
                    for key in (
                        "hinf_feasible", "rpi_feasible",
                        "x_tightening_feasible", "u_tightening_feasible",
                        "invariant_feasible", "safe_center_feasible",
                        "v_ref_feasible", "verification_qp_feasible",
                        "residual_authority_feasible",
                    ):
                        diagnostic[key] = bool(
                            diagnostic[key] or gate.get(key, False)
                        )
                    diagnostic["minimum_residual_authority"] = max(
                        float(diagnostic["minimum_residual_authority"]),
                        float(gate.get("minimum_residual_authority", 0.0)),
                    )
                    continue
                rebuilt = candidate
                diagnostic.update(gate)
                break
            if rebuilt is not None:
                break
        if rebuilt is None:
            failed = [
                key for key in (
                    "hinf_feasible", "rpi_feasible",
                    "x_tightening_feasible", "u_tightening_feasible",
                    "invariant_feasible", "safe_center_feasible",
                    "v_ref_feasible", "verification_qp_feasible",
                    "residual_authority_feasible",
                )
                if not bool(diagnostic[key])
            ]
            diagnostic["failure_reason"] = ",".join(failed)
            return None, diagnostic

        candidate_learner = OnlineThetaLearner(
            self.cfg,
            self.model,
            rebuilt,
            np.random.default_rng(self.cfg.seed + 83000 + attempt_index),
        )
        refined, _ = candidate_learner.optimize_static_safety_design(rebuilt)
        self.set_update_attempts += candidate_learner.set_update_attempts
        self.set_update_accepts += candidate_learner.set_update_accepts
        self.k_update_attempts += candidate_learner.k_update_attempts
        self.k_update_accepts += candidate_learner.k_update_accepts

        s_subset_tightened = bool(
            np.all(refined.invariant_lower >= refined.x_lower_tight - tolerance)
            and np.all(refined.invariant_upper <= refined.x_upper_tight + tolerance)
        )
        s_plus_z_contained = bool(
            np.all(
                refined.invariant_lower - refined.rpi_support_lower
                >= refined.robust_state_lower - tolerance
            )
            and np.all(
                refined.invariant_upper + refined.rpi_support_upper
                <= refined.robust_state_upper + tolerance
            )
        )
        input_tube_contained = bool(
            np.all(
                refined.u_lower_tight - refined.input_rpi_support_lower
                >= refined.robust_input_lower - tolerance
            )
            and np.all(
                refined.u_upper_tight + refined.input_rpi_support_upper
                <= refined.robust_input_upper + tolerance
            )
        )
        residual_membership = bool(all(
            point_in_convex_polygon(point, refined.w_vertices, tol=tolerance)
            for point in residuals
        ))
        safe_center_feasible = bool(
            np.all(refined.z_ref > refined.robust_state_lower + tolerance)
            and np.all(refined.z_ref < refined.robust_state_upper - tolerance)
        )
        v_ref_feasible = bool(
            np.all(refined.v_ref >= refined.u_lower_tight - tolerance)
            and np.all(refined.v_ref <= refined.u_upper_tight + tolerance)
        )
        verification_qp_feasible = bool(
            refined.minimum_residual_authority
            >= float(self.cfg.qp_min_residual_authority) - tolerance
        )
        feasible = bool(
            s_subset_tightened
            and s_plus_z_contained
            and input_tube_contained
            and residual_membership
            and safe_center_feasible
            and v_ref_feasible
            and verification_qp_feasible
        )
        diagnostic.update({
            "s_subset_tightened": s_subset_tightened,
            "s_plus_z_contained": s_plus_z_contained,
            "input_tube_contained": input_tube_contained,
            "residual_membership_passed": residual_membership,
            "safe_center_feasible": safe_center_feasible,
            "v_ref_feasible": v_ref_feasible,
            "verification_qp_feasible": verification_qp_feasible,
            "residual_authority_feasible": verification_qp_feasible,
            "minimum_residual_authority": float(
                refined.minimum_residual_authority
            ),
            "rpi_area": _polygon_area(
                refined.rpi_boundary, self.cfg.state_scale
            ),
            "x_minus_z_area": float(np.prod(
                (refined.x_upper_tight - refined.x_lower_tight)
                * self.cfg.state_scale
            )),
            "invariant_area": float(np.prod(
                (refined.invariant_upper - refined.invariant_lower)
                * self.cfg.state_scale
            )),
            "feasible": feasible,
            "failure_reason": "" if feasible else "self-consistency check",
        })
        return (refined if feasible else None), diagnostic

    def build_certified_robust_operating_design(
        self,
        initial_design: SafetyDesign,
    ) -> tuple[SafetyDesign, dict[str, float]]:
        """Find the feasible scale interval and return its largest design."""
        growth = float(self.cfg.robust_region_growth_factor)
        maximum = float(self.cfg.robust_region_max_scale)
        if growth <= 1.0 or maximum < 1.0:
            raise ValueError(
                "robust region growth_factor must exceed one and max_scale "
                "must be at least one"
            )
        self.robust_region_search_diagnostics = []
        self.first_feasible_scale = float("nan")
        self.minimum_certified_scale = float("nan")
        self.maximum_certified_scale = float("nan")
        self.last_lower_infeasible_scale = float("nan")
        self.first_upper_infeasible_scale = float("nan")
        attempt_index = 0

        def attempt(
            scale: float, seed: SafetyDesign
        ) -> SafetyDesign | None:
            nonlocal attempt_index
            attempt_index += 1
            candidate, diagnostic = self._candidate_region_diagnostics(
                scale, seed, attempt_index
            )
            self.robust_region_search_diagnostics.append(diagnostic)
            if candidate is None or not bool(diagnostic.get("feasible", False)):
                return None
            return candidate

        # Phase A: scale=1 may be too narrow to contain Z/KZ.  Continue the
        # geometric scan until the first independently certified scale appears.
        scale = 1.0
        first_feasible_design: SafetyDesign | None = None
        first_feasible_scale = float("nan")
        last_lower_infeasible_scale = float("nan")
        while scale <= maximum + 1e-12:
            candidate = attempt(scale, initial_design)
            if candidate is not None:
                first_feasible_design = candidate
                first_feasible_scale = scale
                break
            last_lower_infeasible_scale = scale
            if scale >= maximum - 1e-12:
                break
            scale = min(maximum, scale * growth)

        if first_feasible_design is None:
            gates = []
            for row in self.robust_region_search_diagnostics:
                gates.append(
                    "scale={scale:.9g}: hinf={hinf}, rpi={rpi}, "
                    "x_tight={x_tight}, u_tight={u_tight}, "
                    "invariant={invariant}, v_ref={v_ref}, "
                    "verification_qp={verification_qp}, "
                    "residual_authority={authority:.6g}, "
                    "max_residual_norm={residual:.6g}".format(
                        scale=float(row["scale"]),
                        hinf=bool(row.get("hinf_feasible", False)),
                        rpi=bool(row.get("rpi_feasible", False)),
                        x_tight=bool(row.get("x_tightening_feasible", False)),
                        u_tight=bool(row.get("u_tightening_feasible", False)),
                        invariant=bool(row.get("invariant_feasible", False)),
                        v_ref=bool(row.get("v_ref_feasible", False)),
                        verification_qp=bool(
                            row.get("verification_qp_feasible", False)
                        ),
                        authority=float(
                            row.get("minimum_residual_authority", 0.0)
                        ),
                        residual=float(row["max_residual_norm"]),
                    )
                )
            raise RuntimeError(
                "No feasible robust operating region was found on the "
                f"configured scale range [1, {maximum:.9g}].\n  "
                + "\n  ".join(gates)
            )

        # Phase B: grow from the first feasible coarse point.  Every attempt
        # reconstructs its own X_R/U_R samples and W inside _candidate_... .
        last_feasible_design = first_feasible_design
        last_feasible_scale = first_feasible_scale
        first_upper_infeasible_scale = float("nan")
        while last_feasible_scale < maximum - 1e-12:
            candidate_scale = min(maximum, last_feasible_scale * growth)
            candidate = attempt(candidate_scale, last_feasible_design)
            if candidate is None:
                first_upper_infeasible_scale = candidate_scale
                break
            last_feasible_design = candidate
            last_feasible_scale = candidate_scale
            if candidate_scale >= maximum - 1e-12:
                break

        # Phase C (lower bracket): only bisect the adjacent observed
        # infeasible->feasible pair.  The returned controller does not use this
        # boundary; it is retained as a useful feasibility diagnostic.
        minimum_certified_scale = first_feasible_scale
        if np.isfinite(last_lower_infeasible_scale):
            lower = last_lower_infeasible_scale
            upper = first_feasible_scale
            upper_design = first_feasible_design
            for _ in range(int(self.cfg.robust_region_bisection_iterations)):
                middle = 0.5 * (lower + upper)
                candidate = attempt(middle, upper_design)
                if candidate is None:
                    lower = middle
                else:
                    upper = middle
                    upper_design = candidate
            minimum_certified_scale = upper

        # Phase C (upper bracket): likewise bisect only the adjacent observed
        # feasible->infeasible pair, and keep the largest certified design.
        if np.isfinite(first_upper_infeasible_scale):
            lower = last_feasible_scale
            upper = first_upper_infeasible_scale
            for _ in range(int(self.cfg.robust_region_bisection_iterations)):
                middle = 0.5 * (lower + upper)
                candidate = attempt(middle, last_feasible_design)
                if candidate is None:
                    upper = middle
                else:
                    lower = middle
                    last_feasible_scale = middle
                    last_feasible_design = candidate

        self.first_feasible_scale = float(first_feasible_scale)
        self.minimum_certified_scale = float(minimum_certified_scale)
        self.maximum_certified_scale = float(last_feasible_scale)
        self.last_lower_infeasible_scale = float(last_lower_infeasible_scale)
        self.first_upper_infeasible_scale = float(
            first_upper_infeasible_scale
        )
        self.last_feasible_scale = self.maximum_certified_scale
        self.first_infeasible_scale = self.first_upper_infeasible_scale
        self.refinement_iterations = int(attempt_index)
        self.refinement_uncovered_count = 0
        self.refinement_max_excess = 0.0
        self.refinement_relative_rpi_change = float("nan")
        self.data_hull = np.asarray(
            last_feasible_design.w_data_hull, dtype=float
        ).copy()
        feasible_rows = [
            row for row in self.robust_region_search_diagnostics
            if bool(row["feasible"])
        ]
        self.robust_region_max_sample_residual_norm = float(
            feasible_rows[-1]["max_residual_norm"]
        )
        self.self_consistency_s_plus_z_passed = bool(
            feasible_rows[-1]["s_plus_z_contained"]
        )
        return last_feasible_design, self.metrics(last_feasible_design)

    def build_self_consistent_proposed_design(
        self,
        initial_design: SafetyDesign,
    ) -> tuple[SafetyDesign, dict[str, float]]:
        """Backward-compatible alias for robust operating-region certification."""
        return self.build_certified_robust_operating_design(initial_design)

    def _safe_set_update(
        self,
        design: SafetyDesign,
        episode: int,
    ) -> SafetyDesign:
        if len(self.transitions) < int(self.cfg.theta_min_transition_samples):
            return design
        self.set_update_attempts += 1
        step = np.deg2rad(float(self.cfg.theta_m_angle_step_degrees))
        limit = np.deg2rad(float(self.cfg.theta_m_max_angle_degrees))
        # Alternate directions so rejected local proposals do not bias rotation.
        direction = 1.0 if self.set_update_attempts % 2 else -1.0
        proposal_angle = float(np.clip(
            design.theta_m_angle + direction * step, -limit, limit
        ))
        proposals = [design]
        current_contains_all_data = all(
            point_in_convex_polygon(point, design.w_vertices, tol=1e-8)
            for point in self.data_hull
        )
        # Paper Eq. (40) analogue: a newly observed exterior residual forces an
        # enclosing-set rebuild even when the proposed facet rotation fails.
        if not current_contains_all_data:
            try:
                proposals.append(build_safety_design(
                    self.cfg,
                    self.model,
                    np.random.default_rng(self.cfg.seed + 29000 + episode),
                    w_data_hull=self.data_hull,
                    theta_m_angle=design.theta_m_angle,
                    theta_h=design.theta_h,
                    theta_p=design.theta_p,
                    theta_k=design.k,
                    **self._region_kwargs(design),
                ))
            except RuntimeError:
                pass
        try:
            proposals.append(build_safety_design(
                self.cfg,
                self.model,
                np.random.default_rng(self.cfg.seed + 30000 + episode),
                w_data_hull=self.data_hull,
                theta_m_angle=proposal_angle,
                theta_h=design.theta_h,
                theta_p=design.theta_p,
                theta_k=design.k,
                **self._region_kwargs(design),
            ))
        except RuntimeError:
            pass
        feasible = []
        for candidate in proposals:
            contains_data = all(
                point_in_convex_polygon(point, candidate.w_vertices, tol=1e-8)
                for point in self.data_hull
            )
            stable = spectral_radius(
                candidate.a + candidate.b @ candidate.k
            ) < 1.0
            nonempty = bool(
                np.all(candidate.x_upper_tight > candidate.x_lower_tight)
                and np.all(candidate.u_upper_tight > candidate.u_lower_tight)
                and np.all(candidate.invariant_upper > candidate.invariant_lower)
            )
            if contains_data and stable and nonempty:
                feasible.append(candidate)
        selected = min(feasible, key=self._design_score) if feasible else design
        changed = (
            abs(selected.theta_m_angle - design.theta_m_angle) > 1e-12
            or np.linalg.norm(selected.k - design.k) > 1e-10
            or len(selected.w_data_hull) != len(design.w_data_hull)
        )
        if changed:
            self.set_update_accepts += 1
        return selected

    def _safe_k_update(
        self,
        design: SafetyDesign,
        episode: int,
        evaluator: Callable[[SafetyDesign], float] | None,
    ) -> SafetyDesign:
        """Continuous, two-sided policy search over all four entries of K.

        Unlike the former finite Q/R grid, the proposal is made directly in
        R^(2x2).  A proposal is considered only after a complete rebuild has
        certified the H-infinity bound, RPI convergence, tightened state/input
        constraints and the controlled-invariant QP set.  The current SAC
        policy then supplies a common-random-number evaluation score.
        """
        if len(self.transitions) < int(self.cfg.theta_min_transition_samples):
            return design
        self.k_update_attempts += 1
        coordinate = (self.k_update_attempts - 1) % design.k.size
        self.last_k_coordinate = coordinate
        step = float(self.k_steps.flat[coordinate])

        candidates: list[SafetyDesign] = []
        for sign in (-1.0, 1.0):
            proposed_k = np.asarray(design.k, dtype=float).copy()
            proposed_k.flat[coordinate] += sign * step
            try:
                candidate = build_safety_design(
                    self.cfg,
                    self.model,
                    np.random.default_rng(
                        self.cfg.seed + 41000 + 10 * episode + int(sign > 0.0)
                    ),
                    w_data_hull=self.data_hull,
                    theta_m_angle=design.theta_m_angle,
                    theta_h=design.theta_h,
                    theta_p=design.theta_p,
                    theta_k=proposed_k,
                    **self._region_kwargs(design),
                )
            except RuntimeError:
                continue
            if all(
                point_in_convex_polygon(point, candidate.w_vertices, tol=1e-8)
                for point in self.data_hull
            ):
                candidates.append(candidate)

        def score(candidate: SafetyDesign) -> float:
            if evaluator is not None:
                value = float(evaluator(candidate))
                return value if np.isfinite(value) else -np.inf
            return -self._design_score(candidate)

        current_score = score(design)
        scored = [(score(candidate), candidate) for candidate in candidates]
        best_score, selected = max(
            scored,
            key=lambda item: item[0],
            default=(-np.inf, design),
        )
        tolerance = float(self.cfg.theta_k_acceptance_tolerance)
        if best_score > current_score + tolerance:
            self.k_update_accepts += 1
            self.last_k_score = best_score
            self.k_steps.flat[coordinate] = min(
                float(self.cfg.theta_k_max_step),
                step * float(self.cfg.theta_k_step_growth),
            )
            return selected

        self.last_k_score = current_score
        self.k_steps.flat[coordinate] = max(
            float(self.cfg.theta_k_min_step),
            step * float(self.cfg.theta_k_step_shrink),
        )
        return design

    def update_after_episode(
        self,
        episode: int,
        design: SafetyDesign,
        design_evaluator: Callable[[SafetyDesign], float] | None = None,
    ) -> tuple[SafetyDesign, dict[str, float]]:
        if not self.cfg.theta_learning_enabled:
            return design, self.metrics(design)
        if episode % int(self.cfg.theta_cost_update_every_episodes) == 0:
            design = self._update_h_p(design)
        if episode % int(self.cfg.theta_set_update_every_episodes) == 0:
            design = self._safe_set_update(design, episode)
        if episode % int(self.cfg.theta_k_update_every_episodes) == 0:
            design = self._safe_k_update(design, episode, design_evaluator)
        return design, self.metrics(design)

    def metrics(self, design: SafetyDesign) -> dict[str, float]:
        rpi_area = _polygon_area(design.rpi_boundary, self.cfg.state_scale)
        pending_uncovered = sum(
            not point_in_convex_polygon(point, design.w_vertices, tol=1e-8)
            for point in self.data_hull
        )
        return {
            "theta_cost_updates": float(self.cost_updates),
            "theta_set_update_attempts": float(self.set_update_attempts),
            "theta_set_update_accepts": float(self.set_update_accepts),
            "theta_K_update_attempts": float(self.k_update_attempts),
            "theta_K_update_accepts": float(self.k_update_accepts),
            "theta_K_last_coordinate": float(self.last_k_coordinate),
            "theta_K_step_norm": float(np.linalg.norm(self.k_steps)),
            "theta_K_last_policy_score": float(self.last_k_score),
            "theta_transition_samples": float(len(self.transitions)),
            "theta_data_hull_vertices": float(len(self.data_hull)),
            "theta_membership_exceedances": float(self.membership_exceedances),
            "theta_pending_uncovered_hull_vertices": float(pending_uncovered),
            "theta_h_norm": float(np.linalg.norm(design.theta_h)),
            "theta_p_norm": float(np.linalg.norm(design.theta_p)),
            "theta_M_norm": float(np.linalg.norm(design.theta_m_matrix)),
            "theta_K_change_norm": float(np.linalg.norm(design.k - self.initial_k)),
            "theta_M_angle_deg": float(np.rad2deg(design.theta_m_angle)),
            "theta_td_error_abs_mean": float(self.last_td_error),
            "theta_rpi_area": float(rpi_area),
            "theta_minimum_residual_authority": float(
                design.minimum_residual_authority
            ),
            "theta_rpi_area_change_percent": float(
                100.0 * (rpi_area - self.initial_rpi_area)
                / max(self.initial_rpi_area, 1e-12)
            ),
            "robust_refinement_iterations": float(self.refinement_iterations),
            "robust_refinement_uncovered_count": float(
                self.refinement_uncovered_count
            ),
            "robust_refinement_max_excess": float(self.refinement_max_excess),
            "robust_refinement_relative_rpi_change": float(
                self.refinement_relative_rpi_change
            ),
            "robust_region_search_attempts": float(
                len(self.robust_region_search_diagnostics)
            ),
            "last_feasible_scale": float(self.last_feasible_scale),
            "first_infeasible_scale": float(self.first_infeasible_scale),
            "first_feasible_scale": float(self.first_feasible_scale),
            "minimum_certified_scale": float(self.minimum_certified_scale),
            "maximum_certified_scale": float(self.maximum_certified_scale),
            "last_lower_infeasible_scale": float(
                self.last_lower_infeasible_scale
            ),
            "first_upper_infeasible_scale": float(
                self.first_upper_infeasible_scale
            ),
            "robust_region_residual_hull_vertices": float(
                len(design.w_data_hull)
            ),
            "robust_region_max_sample_residual_norm": float(
                self.robust_region_max_sample_residual_norm
            ),
            "self_consistency_S_plus_Z_passed": float(
                self.self_consistency_s_plus_z_passed
            ),
        }
