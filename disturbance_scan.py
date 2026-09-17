"""CLI for rho_d-scaled external-uncertainty applicability analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .disturbance_experiments import (
    run_disturbance_scale_scan,
    run_full_disturbance_stress_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-profile", choices=("default", "zanon2016"), default="zanon2016")
    parser.add_argument("--output-dir", type=Path, default=Path("evaporation_safe_sac/disturbance_scale_scan_zanon2016"))
    parser.add_argument("--stress-steps", type=int, default=None)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "run a fast non-formal rho_d=0 pipeline check; rho_d always scales "
            "only [F1,X1,T1,T200], never paper2016 state shocks"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(benchmark_profile=args.benchmark_profile)
    if args.smoke_test:
        cfg.disturbance_scale_scan_grid = (0.0,)
        cfg.disturbance_scale_bisection_iterations = 0
        cfg.disturbance_scale_random_samples = 50
        cfg.robust_region_random_samples = 50
        cfg.robust_region_bisection_iterations = 1
        cfg.disturbance_scale_static_outer_iterations = 0
        cfg.disturbance_scale_m_angle_max_degrees = 0.0
        cfg.full_disturbance_stress_steps = 50
    if args.stress_steps is not None:
        cfg.full_disturbance_stress_steps = int(args.stress_steps)
    result = run_disturbance_scale_scan(cfg, args.output_dir)
    stress = None
    if not result.full_disturbance_formal_certified:
        stress = run_full_disturbance_stress_test(
            result.selected_cfg,
            result.selected_model,
            result.selected_design,
            args.output_dir,
        )
    payload = {
        "experiment_type": "external_uncertainty_applicability_analysis",
        "formal_result": not bool(args.smoke_test),
        "disturbance_variables": ["F1", "X1", "T1", "T200"],
        "full_half_range": cfg.disturbance_full_half_range.tolist(),
        "rho_d_max_certified": result.rho_d_max_certified,
        "alpha_max_certified_legacy": result.rho_d_max_certified,
        "related_to_paper_state_shock_scaling": False,
        "full_disturbance_formal_certified": result.full_disturbance_formal_certified,
        **result.corner_summary,
        "full_disturbance_stress_test": stress,
    }
    with (args.output_dir / "disturbance_experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
