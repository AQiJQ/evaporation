"""CLI for the independent disturbance-envelope certification experiment."""
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(benchmark_profile=args.benchmark_profile)
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
        "alpha_max_certified": result.alpha_max_certified,
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
