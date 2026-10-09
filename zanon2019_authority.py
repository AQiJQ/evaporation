"""Authority-specific experiment adapter, not a change to the safety mapping.

Implementation choice for reproduction: validation 420000..420009, reserved
test 430000..430049, economic comparability and hierarchical selection are
research protocol choices, not specified in the paper. Empirical G is not Hinf.
"""
from pathlib import Path

import numpy as np

from .zanon2019_benchmark import StochasticInterior23

VALIDATION_SEEDS = list(range(420000, 420010))
RESERVED_TEST_SEEDS = list(range(430000, 430050))
PILOT_ALPHAS = (.10, .20, .30, .50)
AUTHORITY_RULES = {
    "hard_gate": "all validation physical/input/QP/Omega counts zero; W diagnostic only",
    "economics": "every validation seed J_policy<=1.005 J_base (1% also reported)",
    "hierarchy": "post-warmup, empirical safety, 0.5% comparability, min/p1/p5 margin and near-boundary fraction, disturbance rejection, control quality, economics tie-break",
    "selection_ties": "exact lexicographic tuple, no scalar weighted sum; Pareto comparison tolerance 1e-10",
    "boundary_occupancy": "reported, not an invented absolute rejection threshold",
    "sweet_spot": "review full validation Pareto front; never select smallest alpha automatically",
    "scope": "implementation choice for reproduction; not specified in the paper",
}


class AuthorityController(StochasticInterior23):
    """Scale environment command; critic/replay/entropy still use raw action.

    alpha=1 calls the original mapping with the identical action object.
    The no-jump benchmark retains the original inactive Gm/Bj behavior.
    """
    def act(self, state, actor_action, *, action_is_normalized=True):
        alpha = float(getattr(self.cfg, "stochastic_residual_scale", 1.0))
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("stochastic_residual_scale must be finite in [0,1]")
        if alpha != 1 and self.cfg.disturbance_mode != "zanon2019_stochastic":
            raise ValueError("Residual authority scaling is only for zanon2019_stochastic")
        if not action_is_normalized and alpha != 1:
            raise ValueError("Authority scaling requires normalized actor actions")
        scaled = actor_action if alpha == 1 else alpha * np.asarray(actor_action)
        control, info = super().act(state, scaled, action_is_normalized=action_is_normalized)
        info["stochastic_residual_scale"] = alpha
        info["scaled_actor_action"] = np.asarray(scaled).copy()
        return control, info


def authority_assessment(rows, status):
    """Keep original diagnostic levels; add authority-specific hard selection."""
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    result = dict(status)
    result.update({
        "empirical_safe": status["empirical_safe"] and status["Omega_exit"] == 0,
        "economic_degradation_pct": -status["economic_improvement_pct"],
        "worst_economic_degradation_pct": -status["worst_economic_improvement_pct"],
        "economic_comparable_0p5pct": all(r["economic_comparable_0p5pct"] for r in rows),
        "economic_comparable_1pct": all(r["economic_comparable_1pct"] for r in rows),
        "minimum_X2_margin": min(r["SAC_X2_margin_min"] for r in rows),
        "X2_margin_p1": min(r["SAC_X2_margin_p1"] for r in rows),
        "X2_margin_p5": min(r["SAC_X2_margin_p5"] for r in rows),
        "margin_quantile_aggregation": "worst per-seed quantile, not pooled quantile",
        "stochastic_residual_scale": rows[0]["stochastic_residual_scale"],
        "validation_seed_count": len(rows),
        "physical_state_violation": sum(r["SAC_physical_state_violation_steps"] for r in rows),
        "input_violation": sum(r["SAC_physical_input_violation_count"] for r in rows),
        "baseline_minimum_X2_margin": min(r["baseline_X2_margin_min"] for r in rows),
        "baseline_X2_margin_p1": min(r["baseline_X2_margin_p1"] for r in rows),
        "baseline_X2_margin_p5": min(r["baseline_X2_margin_p5"] for r in rows),
    })
    for key in ("X2_margin_mean", "X2_margin_p50", "X2_margin_fraction_lt_0p05",
                "X2_margin_fraction_lt_0p10", "X2_margin_fraction_lt_0p20",
                "scaled_action_norm_mean", "raw_actor_mean_0", "raw_actor_mean_1",
                "raw_actor_std_0", "raw_actor_std_1"):
        result[key] = mean("SAC_" + key)
    ratios = ("X2_IAE_ratio", "P2_IAE_ratio", "X2_ISE_ratio", "P2_ISE_ratio",
              "G_X2_ratio", "G_P2_ratio", "X2_std_ratio", "P2_std_ratio",
              "P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio")
    result["validation_quality_within_1p10"] = all(r[k] <= 1.10 for r in rows for k in ratios)
    # Full per-seed metrics remain in fixed_evaluation.csv. Here absolute
    # occupancy/control values are mean counts per realization, not ratios.
    for name in ("P100", "F200"):
        for suffix in ("TV", "delta_RMS", "input_sign_changes", "lower_bound_steps",
                       "upper_bound_steps", "outer_10pct_steps"):
            for label in ("baseline", "SAC"):
                result[f"{label}_{name}_{suffix}"] = mean(f"{label}_{name}_{suffix}")
        result[f"raw_actor_saturation_fraction_{0 if name == 'P100' else 1}"] = mean(
            f"SAC_raw_actor_saturation_fraction_{0 if name == 'P100' else 1}")
    result["authority_joint_candidate"] = (result["post_warmup"] and result["empirical_safe"]
        and result["economic_comparable_0p5pct"] and result["validation_quality_within_1p10"])
    result["eligible_empirical_joint_checkpoint"] = (result["eligible_empirical_joint_checkpoint"]
                                                    and result["empirical_safe"])
    return result


def hierarchy_key(s):
    return (-s["minimum_X2_margin"], -s["X2_margin_p1"], -s["X2_margin_p5"],
            s["X2_margin_fraction_lt_0p05"], s["X2_margin_fraction_lt_0p10"],
            s["X2_margin_fraction_lt_0p20"], s["mean_IAE_ratio"], s["mean_ISE_ratio"],
            .5 * (s["G_X2_ratio"] + s["G_P2_ratio"]),
            .5 * (s["X2_std_ratio"] + s["P2_std_ratio"]),
            s["mean_TV_ratio"], s["mean_RMS_du_ratio"], s["any_outer_10pct_fraction"],
            -s["economic_improvement_pct"])


class AuthoritySelection:
    def __init__(self, model_dir):
        self.model_dir, self.best = Path(model_dir), {}

    def consider(self, agent, s):
        if not s["post_warmup"] or not s["empirical_safe"] or s["Omega_exit"]:
            return
        keys = {"best_economic_actor": (-s["economic_improvement_pct"],)}
        if s["economic_comparable_0p5pct"]:
            keys.update({
                "best_safety_economic_actor": hierarchy_key(s),
                "best_disturbance_rejection_actor": (s["mean_IAE_ratio"], s["mean_ISE_ratio"],
                    .5 * (s["G_X2_ratio"] + s["G_P2_ratio"]), *hierarchy_key(s)),
                "best_margin_actor": (-s["minimum_X2_margin"], -s["X2_margin_p1"],
                                      -s["X2_margin_p5"], *hierarchy_key(s)),
                "best_low_activity_actor": (s["mean_TV_ratio"], s["mean_RMS_du_ratio"],
                                             s["any_outer_10pct_fraction"], *hierarchy_key(s)),
            })
        for name, key in keys.items():
            prev = self.best.get(name)
            if prev is None or tuple(key) < tuple(prev["selection_score"]):
                agent.save_actor(self.model_dir / f"{name}.pth")
                self.best[name] = {**s, "selection_score": list(key),
                    "checkpoint": str(self.model_dir / f"{name}.pth")}
