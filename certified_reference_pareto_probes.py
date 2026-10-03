"""Targeted full-gate reference probes for the margin-aware Pareto study."""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .certified_reference_search import FIELDS, evaluate_candidate
from .config import ExperimentConfig
from .model import EvaporatorModel


STATES = (
    (25.38000000, 50.11250000),
    (25.39000000, 50.11875000),
    (25.40000000, 50.50000000),
    (25.40000000, 51.00000000),
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "evaporation_safe_sac"
    / "outputs_certified_reference_pareto_probes"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design_dir = OUTPUT / "candidate_designs"
    design_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "reference_search_candidates.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    done = {(round(float(r["X2"]), 8), round(float(r["P2"]), 8)) for r in rows}
    remaining = [state for state in STATES if state not in done]
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    paper_input = model.steady_input(
        cfg.paper2016_steady_state, cfg.disturbance_nominal
    )
    paper_cost = float(model.economic_cost(
        cfg.paper2016_steady_state, paper_input, cfg.disturbance_nominal
    ))
    print(f"{len(remaining)} full-gate reference probes to run", flush=True)
    if remaining:
        with ProcessPoolExecutor(max_workers=min(4, len(remaining))) as pool:
            futures = {
                pool.submit(
                    evaluate_candidate, np.asarray(state, dtype=float),
                    "pareto_probe", 42, paper_cost, design_dir,
                ): state
                for state in remaining
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                with csv_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(FIELDS) + ["design_file"],
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"state={[row['X2'], row['P2']]} "
                    f"certified={row['fully_certified']} "
                    f"PoR={row['cost_increase_percent']:.9g}% "
                    f"authority={row['minimum_residual_authority']:.9g} "
                    f"reason={row['failure_reason'] or 'all_passed'}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
