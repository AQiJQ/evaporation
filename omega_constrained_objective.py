"""Paired-baseline constraints for the frozen balanced-B dwell20 experiment.

The five constraints have the same sign convention: g <= 0 is feasible.
Recovery and input-TV tolerances reuse omega_sac_train.assess_checkpoint's
10% rule (including +1 physical unit when baseline TV is zero). The former
checkpoint rule has no boundary-occupancy threshold, so this first audit
extends its 10% relative tolerance to baseline occupancy steps explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KEYS = ("x2", "p2", "tv_p100", "tv_f200", "boundary")
RECOVERY_START_STEP = 40
RELATIVE_TOLERANCE = 0.10


def allowed_extra_tv(baseline_tv: float) -> float:
    baseline_tv = float(baseline_tv)
    return 1.0 if baseline_tv < 1e-12 else RELATIVE_TOLERANCE * baseline_tv


@dataclass(frozen=True)
class PairedLimits:
    baseline_x2_iae: float
    baseline_p2_iae: float
    baseline_p100_tv: float
    baseline_f200_tv: float
    baseline_boundary_steps: int
    steps: int
    robust_p100_range: float
    robust_f200_range: float

    def allowed(self) -> np.ndarray:
        return np.array([
            1.0 + RELATIVE_TOLERANCE,
            1.0 + RELATIVE_TOLERANCE,
            allowed_extra_tv(self.baseline_p100_tv)
            / (self.steps * self.robust_p100_range),
            allowed_extra_tv(self.baseline_f200_tv)
            / (self.steps * self.robust_f200_range),
            RELATIVE_TOLERANCE * self.baseline_boundary_steps / self.steps,
        ], dtype=float)


def physical_robust_ranges(cfg, design) -> np.ndarray:
    ranges = np.asarray(cfg.input_scale, dtype=float) * (
        np.asarray(design.robust_input_upper, dtype=float)
        - np.asarray(design.robust_input_lower, dtype=float)
    )
    if np.any(ranges <= 0):
        raise ValueError("Robust physical input ranges must be positive")
    return ranges


def limits_from_fixed_row(row: dict, robust_ranges: np.ndarray,
                          steps: int = 300) -> PairedLimits:
    return PairedLimits(
        baseline_x2_iae=float(row["baseline_X2_IAE"]),
        baseline_p2_iae=float(row["baseline_P2_IAE"]),
        baseline_p100_tv=float(row["baseline_P100_TV"]),
        baseline_f200_tv=float(row["baseline_F200_TV"]),
        baseline_boundary_steps=int(float(row["baseline_robust_input_saturation_steps"])),
        steps=steps,
        robust_p100_range=float(robust_ranges[0]),
        robust_f200_range=float(robust_ranges[1]),
    )


def violations_from_fixed_row(row: dict, limits: PairedLimits) -> np.ndarray:
    actual = np.array([
        float(row["X2_IAE"]) / max(limits.baseline_x2_iae, 1e-12),
        float(row["P2_IAE"]) / max(limits.baseline_p2_iae, 1e-12),
        (float(row["P100_TV"]) - limits.baseline_p100_tv)
        / (limits.steps * limits.robust_p100_range),
        (float(row["F200_TV"]) - limits.baseline_f200_tv)
        / (limits.steps * limits.robust_f200_range),
        (float(row["robust_input_saturation_steps"])
         - limits.baseline_boundary_steps) / limits.steps,
    ], dtype=float)
    return actual - limits.allowed()


def boundary_occupied(control: np.ndarray, robust_lower: np.ndarray,
                      robust_upper: np.ndarray) -> bool:
    mid = 0.5 * (robust_lower + robust_upper)
    half = 0.5 * (robust_upper - robust_lower)
    return bool(np.any(np.abs(np.asarray(control) - mid) / half > 0.9))


def limits_from_baseline_records(records: list[dict], cfg, model,
                                 design) -> PairedLimits:
    reference = np.asarray(cfg.robust_economic_reference_state, dtype=float)
    states = np.asarray([r["state"] for r in records], dtype=float)
    controls = np.asarray([r["control"] for r in records], dtype=float)
    robust_lower = model.physical_input(design.robust_input_lower)
    robust_upper = model.physical_input(design.robust_input_upper)
    ranges = physical_robust_ranges(cfg, design)
    return PairedLimits(
        baseline_x2_iae=float(np.sum(np.abs(
            states[RECOVERY_START_STEP:, 0] - reference[0]))),
        baseline_p2_iae=float(np.sum(np.abs(
            states[RECOVERY_START_STEP:, 1] - reference[1]))),
        baseline_p100_tv=float(np.sum(np.abs(np.diff(controls[:, 0])))),
        baseline_f200_tv=float(np.sum(np.abs(np.diff(controls[:, 1])))),
        baseline_boundary_steps=sum(boundary_occupied(
            u, robust_lower, robust_upper) for u in controls),
        steps=len(records),
        robust_p100_range=float(ranges[0]),
        robust_f200_range=float(ranges[1]),
    )


def paired_step_components(step: int, state: np.ndarray, control: np.ndarray,
                           economic_cost: float, previous_control: np.ndarray | None,
                           baseline_records: list[dict], limits: PairedLimits,
                           cfg, model, design) -> np.ndarray:
    """Component sum equals episode-level normalized performance quantities."""
    base = baseline_records[step]
    reference = np.asarray(cfg.robust_economic_reference_state, dtype=float)
    robust_lower = model.physical_input(design.robust_input_lower)
    robust_upper = model.physical_input(design.robust_input_upper)
    recovery = np.zeros(2, dtype=float)
    if step >= RECOVERY_START_STEP:
        recovery = np.abs(np.asarray(state) - reference) / np.maximum([
            limits.baseline_x2_iae, limits.baseline_p2_iae
        ], 1e-12)
    tv = np.zeros(2, dtype=float)
    if step > 0:
        base_previous = np.asarray(baseline_records[step - 1]["control"])
        base_control = np.asarray(base["control"])
        tv = (np.abs(np.asarray(control) - np.asarray(previous_control))
              - np.abs(base_control - base_previous)) / (
                  limits.steps * np.array([
                      limits.robust_p100_range, limits.robust_f200_range
                  ])
              )
    boundary = (
        float(boundary_occupied(control, robust_lower, robust_upper))
        - float(boundary_occupied(base["control"], robust_lower, robust_upper))
    ) / limits.steps
    return np.array([
        (float(base["economic_cost"]) - float(economic_cost))
        / float(cfg.reward_cost_scale),
        recovery[0], recovery[1], tv[0], tv[1], boundary,
    ], dtype=np.float32)


def episode_violations(component_sum: np.ndarray,
                       limits: PairedLimits) -> np.ndarray:
    return np.asarray(component_sum[1:], dtype=float) - limits.allowed()
