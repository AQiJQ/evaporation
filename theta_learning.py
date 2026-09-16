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
            "theta_rpi_area_change_percent": float(
                100.0 * (rpi_area - self.initial_rpi_area)
                / max(self.initial_rpi_area, 1e-12)
            ),
        }
