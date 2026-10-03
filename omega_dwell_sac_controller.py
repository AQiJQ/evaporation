"""Interior-anchor candidate followed by fixed-G_m/B_j one-step supervisor."""
from __future__ import annotations

import numpy as np

from .controlled_invariant_error_set import contains, hull
from .omega_dwell_event_supervisor import (
    D, EventDrivenDwellSupervisor, make_rank_sets, validate_source_summary,
)
from .omega_interior_anchor import InteriorAnchorController


class DwellSupervisedInteriorController:
    def __init__(self, cfg, model, design, omega, domain, source):
        self.cfg, self.model, self.d = cfg, model, design
        self.omega, self.domain, self.source = omega, domain, source
        validate_source_summary(source, omega)
        self.candidate = InteriorAnchorController(
            cfg, model, design, omega, domain
        )
        self.cached_sets = {}
        self.supervisor = None
        self.last_context = None
        self.last_info = None
        self.step_index = 0

    @property
    def z(self):
        return self.candidate.z

    def rank_of(self, omega_error):
        return self.candidate.rank_of(omega_error)

    def reset(self, state):
        self.candidate.reset(state)
        self.supervisor = None
        self.last_context = None
        self.last_info = None
        self.step_index = 0

    def set_paper_scenario(self, scenario):
        if scenario not in self.cached_sets:
            self.cached_sets[scenario] = make_rank_sets(
                scenario, self.omega, self.d, self.domain, self.source
            )
        self.supervisor = EventDrivenDwellSupervisor(
            self.cfg, self.model, self.d, self.omega, self.domain,
            scenario, self.source,
            precomputed=self.cached_sets[scenario],
            shock_scale_lower=self.cfg.paper2016_training_shock_scale_lower,
        )
        self.step_index = 0

    def act(self, state, actor_action, *, action_is_normalized=True):
        if self.supervisor is None:
            raise RuntimeError("Dwell supervisor has no Paper2016 scenario")
        context = self.supervisor.observe(state, self.step_index)
        if context["certificate_failure"]:
            raise RuntimeError(
                f"Dwell supervisor certificate_failure before action: "
                f"scenario={self.supervisor.scenario}, step={self.step_index}, "
                f"reason={context['failure_reason']}"
            )
        candidate_input, info = self.candidate.act(
            state, actor_action, action_is_normalized=action_is_normalized
        )
        filtered = self.supervisor.filter_candidate(
            state, candidate_input, context
        )
        if filtered["certificate_failure"]:
            raise RuntimeError(
                f"Dwell supervisor certificate_failure at QP: "
                f"scenario={self.supervisor.scenario}, step={self.step_index}, "
                f"reason={filtered['failure_reason']}"
            )
        final_control = np.asarray(filtered["control"])
        final_norm = self.model.normalized_input(final_control)
        candidate_norm = self.model.normalized_input(candidate_input)
        if info["mode"] == "Z_mode_existing_controller":
            ancillary = np.asarray(info["ancillary"])
            final_nominal = final_norm - ancillary
            self.candidate.inner.z = (
                self.d.a @ np.asarray(info["z_before"])
                + self.d.b @ final_nominal + self.d.affine
            )
            info["nominal"] = final_nominal.copy()
            info["z_next"] = self.candidate.inner.z.copy()
        center_norm = np.asarray(info["action_center_actual_norm"])
        applied = final_norm - center_norm
        new_info = dict(info)
        new_info.update({
            "actual_norm": final_norm.copy(),
            "applied_residual": applied.copy(),
            "parameterized_residual": applied.copy(),
            "qp_feasible": bool(info["qp_feasible"]),
            "execution_mask": float(bool(info["qp_feasible"])),
            # Keep the existing reward's projection term unchanged. The
            # supervisor-specific displacement is logged separately.
            "projection_gap": float(info["projection_gap"]),
            "final_verification_gap": float(np.linalg.norm(
                final_norm - candidate_norm)),
            "supervisor_mode": context["mode"],
            "supervisor_remaining_shocks": context["remaining_shocks"],
            "supervisor_rank_before": context["rank"],
            "supervisor_target_rank": context["target_rank"],
            "supervisor_event_detected": context["event_detected"],
            "supervisor_inferred_jump_scale": context["inferred_jump_scale"],
            "supervisor_q_area": filtered["q_area_normalized"],
            "supervisor_q_radius": filtered["q_chebyshev_radius_normalized"],
            "supervisor_low_authority": filtered["low_authority"],
            "supervisor_projection_distance_physical": filtered[
                "projection_distance_physical"],
            "supervisor_q_min": filtered["q_min_normalized"].copy(),
            "supervisor_q_max": filtered["q_max_normalized"].copy(),
            "supervisor_candidate_input": np.asarray(candidate_input).copy(),
            "supervisor_Gm_exit": False,
            "supervisor_Bk_recovery_failure": False,
            "supervisor_omega_exit": False,
            "supervisor_qp_infeasible": False,
            "supervisor_nonlinear_W_exceedance": False,
            "supervisor_recovery_completed_steps": None,
        })
        self.last_context = context
        self.last_info = new_info
        self.last_q = filtered["q_safe"].copy()
        return final_control, new_info

    def audit_next_state(self, next_state):
        """Called by the generic rollout immediately after the plant step."""
        if self.last_context is None or self.last_info is None:
            raise RuntimeError("Dwell supervisor missing preceding action")
        xi_next = self.model.normalized_state(next_state) - self.d.z_ref
        linear_next = (self.d.a @ self.last_context["xi"]
                       + self.d.b @ self.last_q)
        mismatch = xi_next - linear_next
        w_ok = contains(hull(self.d.w_vertices), mismatch, tol=2e-7)
        self.last_info["supervisor_nonlinear_W_exceedance"] = not w_ok
        self.supervisor.predicted_xi = linear_next
        completed = self.supervisor.complete_step(xi_next, self.last_context)
        self.last_info["supervisor_Gm_exit"] = bool(
            self.last_context["mode"] == "wait_Gm"
            and not contains(self.supervisor.sets[
                self.last_context["remaining_shocks"]], xi_next)
        )
        self.last_info["supervisor_Bk_recovery_failure"] = bool(
            self.last_context["mode"] == "recover_Bj"
            and completed["certificate_failure"]
        )
        self.last_info["supervisor_omega_exit"] = not contains(
            self.omega, xi_next
        )
        if not completed["certificate_failure"]:
            self.last_info["supervisor_rank_next"] = completed["next_rank"]
            if (self.last_context["event_detected"]
                    and self.last_context["rank"] == 0):
                self.last_info["supervisor_recovery_completed_steps"] = 0
            elif (self.last_context["rank"] > 0
                  and completed["next_rank"] == 0):
                self.last_info["supervisor_recovery_completed_steps"] = (
                    self.supervisor.recovery_steps
                )
        self.step_index += 1
        if not w_ok:
            raise RuntimeError("Dwell supervisor certificate_failure: nonlinear mismatch outside W")
        if completed["certificate_failure"]:
            raise RuntimeError(
                "Dwell supervisor certificate_failure after normal step: "
                f"{completed['failure_reason']}"
            )
        if self.last_info["supervisor_omega_exit"]:
            raise RuntimeError("Dwell supervisor certificate_failure: Omega exit")
        return self.last_info
