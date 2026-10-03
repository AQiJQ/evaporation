"""Observation-only 23D ablation for the unchanged dwell supervisor."""
from __future__ import annotations

from copy import copy

import numpy as np

from .omega_dwell_event_supervisor import D, feasible_q_audit
from .omega_dwell_sac_controller import DwellSupervisedInteriorController


def supervisor_observation_extra(context, design, domain):
    audit = feasible_q_audit(
        context["xi"], context["target"], design, domain
    )
    if audit is None:
        raise RuntimeError("Empty admissible-q set in actor observation")
    q_half_width = 0.5 * float(np.min(
        domain["q_upper"] - domain["q_lower"]
    ))
    if q_half_width <= 0:
        raise RuntimeError("Admissible-q domain has no positive width")
    return np.array([
        context["remaining_shocks"] / 3.0,
        float(context["mode"] == "recover_Bj"),
        context["rank"] / float(D),
        min(1.0, max(0.0, audit["radius"] / q_half_width)),
    ], dtype=np.float32)


class DwellObservationController(DwellSupervisedInteriorController):
    """Expose measured supervisor state without advancing its real automaton."""

    def actor_observation_extra(self, state):
        if self.supervisor is None:
            raise RuntimeError("Supervisor scenario must be set before observation")
        preview = copy(self.supervisor)
        preview.events = list(self.supervisor.events)
        context = preview.observe(state, self.step_index)
        if context["certificate_failure"]:
            raise RuntimeError(
                "Uncertified state while building actor observation: "
                + str(context["failure_reason"])
            )
        return supervisor_observation_extra(context, self.d, self.domain)
