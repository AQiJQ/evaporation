"""Selectable actor mapping: radius means fraction of feasible ray authority.

The Omega-SAC training entrypoint can select this mapping or the legacy one.
Both retain the certified Z/Ω polytopes and final verification QPs unchanged.
"""
from __future__ import annotations

import numpy as np

from .control import project_qp_2d
from .controlled_invariant_error_set import contains
from .omega_online_closed_loop import OmegaSafeOnlineController
from .omega_safe_action_filter_diagnosis import qp_rows


def ray_authority(center, direction, rows, bounds):
    direction = np.asarray(direction, dtype=float)
    if np.max(np.abs(direction)) <= 1e-15:
        return 0.0
    slack = np.asarray(bounds) - np.asarray(rows) @ np.asarray(center)
    outward = np.asarray(rows) @ direction
    active = outward > 1e-12
    if not np.any(active):
        raise RuntimeError("Feasible action polytope unexpectedly unbounded")
    return max(0.0, float(np.min(slack[active] / outward[active])))


def action_for_feasible_point(center, desired, rows, bounds, input_scale=None):
    """Radial inverse after physical-input Euclidean projection."""
    scale = (np.ones(2) if input_scale is None else
             np.asarray(input_scale, dtype=float))
    physical_point, feasible = project_qp_2d(
        np.asarray(desired) * scale,
        np.asarray(rows) / scale[np.newaxis, :], bounds,
    )
    if not feasible:
        raise RuntimeError("Offline feasible-action polytope is empty")
    point = physical_point / scale
    delta = point - center
    radius = float(np.max(np.abs(delta)))
    if radius <= 1e-13:
        return np.zeros(2), point
    direction = delta / radius
    tau = ray_authority(center, direction, rows, bounds)
    if tau <= 1e-13:
        raise RuntimeError("Projected feasible point has zero ray authority")
    action = np.clip(delta / tau, -1.0, 1.0)
    return action, point


class FeasibleSetNormalizedController(OmegaSafeOnlineController):
    """Mode-specific full feasible ray map; keeps baseline and final QPs."""

    def act(self, state, actor_action, *, action_is_normalized=True):
        if not action_is_normalized:
            raise ValueError("Expected normalized actor action in [-1,1]^2")
        action = np.asarray(actor_action, dtype=float)
        if action.shape != (2,) or np.any(np.abs(action) > 1 + 1e-9):
            raise ValueError("Actor action must lie in [-1,1]^2")
        x = self.model.normalized_state(state)
        rpi_error = x - self.inner.z
        omega_error = x - self.d.z_ref
        in_z = contains(self.d.rpi_boundary, rpi_error)
        in_omega = contains(self.omega, omega_error)
        omega_exit_event = bool(self.was_in_omega and not in_omega)
        self.was_in_omega = in_omega

        fallback_control, info = self.inner.act(
            state, np.zeros(2), action_is_normalized=True
        )
        q_base = np.asarray(info["base"]) + np.asarray(info["ancillary"]) - self.d.v_ref
        rho = float(np.max(np.abs(action)))
        direction = action / rho if rho > 1e-15 else np.zeros(2)
        tau = 0.0
        center_was_projected = False
        final_gap = 0.0
        mode = "outside_certified_domain"
        control = fallback_control
        safe_rows = np.empty((0, 2))
        safe_bounds = np.empty(0)
        safe_center = np.zeros(2)
        candidate = safe_center.copy()
        final_coordinate = safe_center.copy()
        omega_qp_feasible = None
        qp_feasible = False

        if in_z:
            mode = "Z_mode_existing_controller"
            safe_rows, safe_bounds = np.asarray(info["a_q"]), np.asarray(info["b_q"])
            raw_center = np.asarray(info["nominal"])
            safe_center, center_ok = project_qp_2d(
                raw_center, safe_rows, safe_bounds
            )
            center_was_projected = bool(np.linalg.norm(safe_center - raw_center) > 1e-9)
            if center_ok:
                tau = ray_authority(safe_center, direction, safe_rows, safe_bounds)
                candidate = safe_center + rho * tau * direction
                final_coordinate, final_ok = project_qp_2d(
                    candidate, safe_rows, safe_bounds
                )
                final_gap = float(np.linalg.norm(final_coordinate - candidate))
                qp_feasible = bool(info["qp_feasible"] and final_ok and
                                   np.max(safe_rows @ final_coordinate - safe_bounds) <= 2e-7)
                if qp_feasible:
                    actual_unclipped = final_coordinate + np.asarray(info["ancillary"])
                    actual_n = np.clip(actual_unclipped, self.d.robust_input_lower,
                                       self.d.robust_input_upper)
                    control = self.model.physical_input(actual_n)
                    # Unlike the Ω path, the Z nominal input must advance with
                    # the mapped action, exactly as in SafeController.act.
                    self.inner.z = (self.d.a @ np.asarray(info["z"]) +
                                    self.d.b @ final_coordinate + self.d.affine)
                    info["z_next"] = self.inner.z.copy()
                    info["nominal"] = final_coordinate.copy()
                    info["candidate"] = candidate.copy()
                    info["parameterized_residual"] = final_coordinate - safe_center
                    info["actual_norm"] = actual_n.copy()
                    info["physical_clip_gap"] = float(np.linalg.norm(
                        actual_n - actual_unclipped
                    ))
        elif in_omega:
            mode = "Omega_safe_one_step_QP"
            safe_rows, safe_bounds = qp_rows(
                omega_error, self.omega, self.d, self.domain
            )
            safe_center, center_ok = project_qp_2d(
                q_base, safe_rows, safe_bounds
            )
            center_was_projected = bool(np.linalg.norm(safe_center - q_base) > 1e-9)
            if center_ok:
                tau = ray_authority(safe_center, direction, safe_rows, safe_bounds)
                candidate = safe_center + rho * tau * direction
                final_coordinate, final_ok = project_qp_2d(
                    candidate, safe_rows, safe_bounds
                )
                final_gap = float(np.linalg.norm(final_coordinate - candidate))
                omega_qp_feasible = bool(final_ok and
                    np.max(safe_rows @ final_coordinate - safe_bounds) <= 2e-7)
                qp_feasible = omega_qp_feasible
                if qp_feasible:
                    control = self.model.physical_input(
                        self.d.v_ref + final_coordinate
                    )
                    info["actual_norm"] = self.model.normalized_input(control)
            if not qp_feasible:
                mode = "Omega_QP_infeasible_fallback"

        actual_n = self.model.normalized_input(control)
        if in_z:
            physical_center_n = safe_center + np.asarray(info["ancillary"])
            physical_candidate_n = candidate + np.asarray(info["ancillary"])
        elif in_omega:
            physical_center_n = self.d.v_ref + safe_center
            physical_candidate_n = self.d.v_ref + candidate
        else:
            physical_center_n = self.model.normalized_input(fallback_control)
            physical_candidate_n = physical_center_n.copy()
        applied = actual_n - physical_center_n
        merged = dict(info)
        merged.update({
            "mode": mode, "in_Z": bool(in_z), "in_Omega": bool(in_omega),
            "omega_exit_event": omega_exit_event,
            "rpi_error": rpi_error.copy(), "omega_error": omega_error.copy(),
            "reachable_rank_before": self.rank_of(omega_error),
            "q_base": q_base.copy(),
            "q_candidate": physical_candidate_n - self.d.v_ref,
            "requested_residual": physical_candidate_n - physical_center_n,
            "applied_residual": applied.copy(),
            "parameterized_residual": applied.copy(),
            "requested_candidate": (
                np.asarray(info["base"]) +
                (physical_candidate_n - physical_center_n)
            ),
            "candidate": (
                np.asarray(info["base"]) +
                (physical_candidate_n - physical_center_n)
            ),
            "omega_qp_feasible": omega_qp_feasible,
            "base_verification_qp_feasible": bool(info["qp_feasible"]),
            "z_before": np.asarray(info["z"]).copy(),
            "z_next": self.inner.z.copy(),
            "actual_norm": actual_n.copy(),
            "qp_feasible": qp_feasible,
            "execution_mask": float(qp_feasible),
            "projection_gap": final_gap,
            "feasible_action_mapping_scale": rho,
            "feasible_action_mapping_gap": final_gap,
            "action_center_coordinate": safe_center.copy(),
            "action_center_actual_norm": physical_center_n.copy(),
            "action_safe_rows": safe_rows.copy(),
            "action_safe_bounds": safe_bounds.copy(),
            "action_direction": direction.copy(),
            "action_rho": rho,
            "action_tau_max": tau,
            "action_candidate_coordinate": candidate.copy(),
            "action_final_coordinate": final_coordinate.copy(),
            "action_center_was_projected": center_was_projected,
            "final_verification_gap": final_gap,
            "final_physical_clip_gap": float(np.linalg.norm(
                actual_n - physical_candidate_n
            )),
            "action_polytope_max_excess": (
                float(np.max(safe_rows @ final_coordinate - safe_bounds))
                if len(safe_rows) else float("nan")
            ),
        })
        return control, merged
