"""Common-state mapping intervention: was expanded authority actually used?

For each archived geometry state/nominal context, query the original online
mapping with the SAME raw action. Membership is tested against the original
alpha-contracted action image, not against misleading reference-only ranges.
This is a local mapping intervention, not a dynamic causal cost attribution.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import numpy as np
import torch
from .. import study

ENV = None


def init():
    global ENV
    torch.set_num_threads(1)
    ENV = study.s.env()


def trial(task):
    name, seed = task
    output = study.ROOT / "authority_interventions" / name / f"seed_{seed}.csv"
    if output.exists():
        return name, seed
    c = next(c for c in study.s.candidates() if c["candidate"] == name)
    env = study.s.independent(ENV, c)
    records = study.read_records(study.s.ROOT / "counterfactual" / name / f"seed_{seed}.csv")
    ctrl = study.io.PairedEvidenceController(*env[:5], np.array([r["disturbance"] for r in records]))
    ctrl.reset(records[0]["state"])
    rows = []
    for k, r in enumerate(records):
        context = dict(state=r["state"], z=ctrl.inner.z.copy())
        raw = np.asarray(r["raw_action"], dtype=np.float32)
        control, current = ctrl.act(r["state"], raw)
        np.testing.assert_allclose(control, r["control"], rtol=0, atol=1e-7)
        original_control, original = study.s.old.point_control(ENV, context, raw)
        center = np.asarray(original["action_center_coordinate"])
        physical_center = np.asarray(original["action_center_actual_norm"])
        actual = ENV[1].normalized_input(control)
        # An input in alpha Q + (1-alpha)c has preimage c+(u-u_center)/alpha.
        preimage = center+(actual-physical_center)/.1
        h, b = original["action_safe_rows"], original["action_safe_bounds"]
        excess = float(np.max(h @ preimage-b)) if len(h) else None
        rows.append(dict(geometry=name, validation_seed=seed, step=k, mode=current["mode"],
            original_QP_feasible=bool(original["qp_feasible"]), original_contracted_image_excess=excess,
            used_outside_original_action_image=bool(excess is not None and excess > 1e-8),
            original_same_raw_P100=float(original_control[0]), original_same_raw_F200=float(original_control[1]),
            current_P100=float(control[0]), current_F200=float(control[1]),
            mapping_only_P100_change=float(control[0]-original_control[0]),
            mapping_only_F200_change=float(control[1]-original_control[1]),
            interpretation="same state/z/raw action; local mapping effect only, no plant propagation"))
    study.write_csv(output, rows)
    return name, seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 workers")
    study.locked()
    tasks = [(name, seed) for name in study.GEOMETRIES for seed in study.DEV]
    if args.workers == 1:
        init()
        for task in tasks:
            print(trial(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
            for future in as_completed([pool.submit(trial, task) for task in tasks]):
                print(future.result(), flush=True)
    summary = []
    for name in study.GEOMETRIES:
        rows = []
        for seed in study.DEV:
            rows += study.io.read_csv(study.ROOT / "authority_interventions" / name / f"seed_{seed}.csv")
        summary.append(dict(geometry=name, steps=len(rows),
            fraction_outside_original_action_image=float(np.mean([r["used_outside_original_action_image"] == "True" for r in rows])),
            mean_abs_mapping_P100_change=float(np.mean([abs(float(r["mapping_only_P100_change"])) for r in rows])),
            mean_abs_mapping_F200_change=float(np.mean([abs(float(r["mapping_only_F200_change"])) for r in rows])),
            original_QP_infeasible_count=sum(r["original_QP_feasible"] != "True" for r in rows)))
    study.write_csv(study.ROOT / "new_authority_actual_usage.csv", summary)
    study.save(study.ROOT / "authority_intervention_definition.json", dict(
        source_sha256=study.io.digest(__file__), alpha=.1, constraint_membership_tolerance=1e-8,
        definition="u_actual belongs to original alpha-contracted Q(s) iff c+(u_actual-u_center)/alpha belongs to original Q(s)",
        fixed_intervention="same state, nominal z and raw actor output, original versus candidate geometry",
        caveat="No plant propagated by intervention; not a dynamic causal attribution or new controller"))
    study.locked()


if __name__ == "__main__":
    main()
