"""Cost attribution -> five certified P100 candidates -> finite witness search.

Opt-in offline phases, no SAC training. Resume uses immutable preregistration.
CPU workers call the original controller/QP/nonlinear plant, not an oracle q.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import inspect
import json
import sys
import time
import numpy as np
import torch

from . import core
from .. import zanon2019_ur_certificate_sensitivity as s
from .. import zanon2019_paired_v2 as v2
from .. import zanon2019_paired_experiment as io
from .. import zanon2019_economic_recovery as recovery
from ..zanon2019_benchmark import write_csv, make_agent, load_actor
from ..train import ZeroResidualPolicy

ROOT = recovery.ROOT / "economic_attribution_v1_replayfix"
GEOMETRIES = ("baseline", "C2_8", "A3", "C2", "C2_3")
FACTORS = (1., 1.025, 1.05, 1.075, 1.10)
DEV = s.DEV
ACTOR = recovery.run_dir(recovery.ROOT, "strong", 42) / "models/episode_0100_actor.pth"
ENV = PATHS = ACTOR_POLICY = None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.save(path, value)


def immutable(path, value):
    if path.exists():
        raise RuntimeError(f"Preregistered evidence already exists: {path}")
    save(path, value)


def protected():
    files = list(io.REPO.glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    files += [s.DEFAULT_DESIGN, s.DEFAULT_OMEGA, ACTOR,
              v2.ROOT / "smoothness_regularization_config.json", v2.ROOT / "reward_scale_audit.json"]
    return {str(p): io.digest(p) for p in sorted(files)}


def p100_candidates():
    return [dict(candidate=f"P100_{int(round(f*1000))}", family="P100_only_preregistered",
                 factor=f, hP=35.*f, hF=30., rhoP=s.RHO, rhoF=s.RHO) for f in FACTORS]


def prepare():
    env = s.env()
    paths, sources = s.development_paths()
    protocol = dict(schema="economic_attribution_v1", alpha=.1, development_seeds=list(DEV), steps=1000,
        actor_path=ACTOR, actor_sha256=io.digest(ACTOR), source_and_production_hashes=protected(),
        geometry_hashes=s.geometry_hashes(env), path_sources=sources, geometries=list(GEOMETRIES),
        p100_candidates=p100_candidates(), coverage_random_samples=10000, coverage_tolerance=env[0].robust_region_membership_tolerance,
        state_witness_budget=512, witness_seed=1902019, witness_family="state-only clipped affine actor output",
        witness_observation="obs[:2]*state_scale; no time, disturbance or future path",
        witness_parameters=core.witness_parameters(), witness_status="finite_state_dependent_witness_search; not oracle upper bound",
        control_quality="existing 1.10 per-path TV/RMS band, diagnostic; IAE_B separately reported, no new weighted score",
        return_not_economic_cost=True, no_training=True, final_seeds_used=False,
        certification_scope="unchanged bounded-W numerical protocol + finite sampling; Gaussian rollouts empirical only")
    immutable(ROOT / "protocol.json", protocol)
    definition = core.cost_definition(env[1])
    immutable(ROOT / "economic_cost_definition.json", definition)
    text = ("# Runtime economic cost\n\n" + definition["formula"] + "\n\n"
        + f"Source: {definition['file']}:{definition['line']}, {definition['function']}.\n\n"
        + "Four additive components: 10.09 F2, 10.09 F3, 600 F100, 0.6 F200. F3=50 is fixed.\n\n"
        + "P100 is not directly priced: P100 -> T100 -> Q100 -> F100 and F4 -> F2.\n"
        + "At fixed x,d, increasing P100 raises steam cost and reduces F2 cost.\n"
        + "F200 has direct cooling cost; its indirect effects appear through subsequent state evolution.\n\n"
        + definition["unit"] + ". Do not assign an unverified currency/time unit.\n\n"
        + definition["timing"] + ". J is the existing undiscounted stage-cost sum, with no added dt factor.\n\n"
        + "Training reward remains separate: paired cost gap/200 minus unchanged fixed excess-move penalty and safety dual.\n")
    (ROOT / "ECONOMIC_COST_DEFINITION.md").write_text(text, encoding="utf-8")
    print("Preregistered 5 P100 geometries and exactly 512 state-only candidates. NO TRAINING.", flush=True)


def locked():
    p = v2.load(ROOT / "protocol.json")
    if p["source_and_production_hashes"] != protected():
        raise RuntimeError("Preregistered source, frozen actor, reward or geometry changed")
    if p["development_seeds"] != list(DEV) or p["alpha"] != .1 or p["state_witness_budget"] != 512:
        raise RuntimeError("Immutable development protocol changed")
    for src in p["path_sources"].values():
        if io.digest(src["path"]) != src["sha256"]:
            raise RuntimeError("Development disturbance archive changed")
    return p


def worker_init():
    global ENV, PATHS, ACTOR_POLICY
    torch.set_num_threads(1)
    ENV = s.env()
    PATHS, _ = s.development_paths()
    ACTOR_POLICY = make_agent(ENV[0], "cpu")
    load_actor(ACTOR_POLICY, ACTOR)


def read_records(path):
    return v2.archived_pair(path)[1]


def component_rows(records, env, geometry, policy, seed):
    parts, nxt = core.economic_arrays(records, env)
    total = parts.sum(1)
    sums = np.cumsum(parts, axis=0)
    rows = []
    for k, r in enumerate(records):
        row = dict(geometry=geometry, policy=policy, validation_seed=seed, step=k,
            X2=float(r["state"][0]), P2=float(r["state"][1]), next_X2=float(nxt[k, 0]), next_P2=float(nxt[k, 1]),
            P100=float(r["control"][0]), F200=float(r["control"][1]), total_economic_stage_cost=float(total[k]),
            cumulative_total_cost=float(sums[k].sum()))
        for j, name in enumerate(core.COMPONENTS):
            row[name] = float(parts[k, j]); row["cumulative_"+name] = float(sums[k, j])
        rows.append(row)
    return rows, parts.sum(0)


def worker_path(task):
    kind, candidate, seed = task
    path, base_prod, _ = PATHS[seed]
    env = s.independent(ENV, candidate)
    name = candidate["candidate"]
    folder = ROOT / ("geometry_pairs" if kind == "audit" else "p100_pairs") / name
    own_path = folder / f"seed_{seed}_own_baseline.csv"
    receipt = folder / f"seed_{seed}_result.json"
    if receipt.exists():
        return v2.load(receipt)
    start = time.perf_counter()
    try:
        if own_path.exists():
            own = read_records(own_path)
        elif name == "baseline" or (kind == "p100" and candidate["factor"] == 1.):
            own = base_prod
        else:
            with torch.no_grad():
                own = io.rollout(env, path, seed, ZeroResidualPolicy())[0]
            metric, _, series = recovery.full_metrics(own, own, env, 0., np.zeros(5))
            write_csv(own_path, recovery.trace(own, own, series))
        archive = s.ROOT / "counterfactual" / name / f"seed_{seed}.csv" if kind == "audit" else None
        if archive is not None and archive.exists():
            sac = read_records(archive)
        else:
            with torch.no_grad():
                sac = io.rollout(env, path, seed, ACTOR_POLICY)[0]
            metric, _, series = recovery.full_metrics(base_prod, sac, env, 0., np.zeros(5))
            write_csv(folder / f"seed_{seed}_frozen_actor.csv", recovery.trace(base_prod, sac, series))
        for records in (own, sac):
            np.testing.assert_array_equal(np.array([r["disturbance"] for r in records]), path)
            np.testing.assert_array_equal(records[0]["state"], base_prod[0]["state"])
        metrics, _ = v2.paired_metrics(sac, own, env, 0., np.zeros(5))
        totals = {}
        for role, records in (("production_baseline", base_prod), ("own_baseline", own), ("frozen_SAC", sac)):
            rows, values = component_rows(records, env, name, role, seed)
            write_csv(folder / f"seed_{seed}_{role}_cost_components.csv", rows)
            totals[role] = values
        components = []
        for j, component in enumerate(core.COMPONENTS):
            p, b, a = (totals[role][j] for role in ("production_baseline", "own_baseline", "frozen_SAC"))
            split = core.effect_split(totals["production_baseline"].sum(), totals["own_baseline"].sum(), totals["frozen_SAC"].sum())
            gain = float(p-a)
            components.append(dict(geometry=name, policy="frozen_SAC", validation_seed=seed, component=component,
                baseline_cost=float(p), own_baseline_cost=float(b), SAC_cost=float(a),
                absolute_change=float(a-p), economic_gain_component=gain,
                geometry_baseline_component_change=float(b-p), SAC_given_geometry_component_change=float(a-b),
                contribution_to_total_gain=gain/(split["J_base_prod"]-split["J_SAC_geometry"])
                    if abs(split["J_base_prod"]-split["J_SAC_geometry"]) > 1e-12 else None))
        np.testing.assert_allclose(sum(r["absolute_change"] for r in components), split["DeltaJ_total"], rtol=0, atol=1e-7)
        attr = core.replay_attribution(env, sac, base_prod, name, seed)
        write_csv(folder / f"seed_{seed}_actor_attribution.csv", attr)
        econ = s.economic_metrics(base_prod, sac)
        result = dict(geometry=name, validation_seed=seed, status="COMPLETE", **split, **{**econ, **metrics},
            production_economic_improvement_pct=econ["economic_improvement_pct"],
            components=components, duration_seconds=time.perf_counter()-start,
            disturbance_sha256=s.array_hash(path), certification_scope="Gaussian empirical only")
    except Exception as exc:
        if getattr(exc, "evidence", None):
            write_csv(folder / f"seed_{seed}_failure_evidence.csv", exc.evidence)
        result = dict(geometry=name, validation_seed=seed, status="FAILED", error=repr(exc))
    save(receipt, result)
    return result


def execute(tasks, function, workers):
    if workers < 1 or workers > 4:
        raise ValueError("Use 1..4 CPU workers; no GPU learning or hidden process spawning")
    if workers == 1:
        worker_init()
        for task in tasks:
            yield function(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            pending = [pool.submit(function, task) for task in tasks]
            for future in as_completed(pending):
                yield future.result()


def audit(workers):
    locked()
    original = {c["candidate"]: c for c in s.candidates()}
    tasks = [("audit", original[name], seed) for name in GEOMETRIES for seed in DEV]
    for index, result in enumerate(execute(tasks, worker_path, workers)):
        print(f"cost-attribution {index+1}/50: {result['geometry']} / {result['validation_seed']} {result['status']}", flush=True)
    locked()
    summarize_attribution()


def summarize_attribution():
    components, splits, pooled, behavior = [], [], [], []
    missing = []
    for name in GEOMETRIES:
        folder = ROOT / "geometry_pairs" / name
        attrs = []
        for seed in DEV:
            path = folder / f"seed_{seed}_result.json"
            if not path.exists() or v2.load(path)["status"] != "COMPLETE":
                missing.append((name, seed)); continue
            r = v2.load(path)
            components.extend(r["components"])
            splits.append({k: v for k, v in r.items() if k != "components"})
            attrs += io.read_csv(folder / f"seed_{seed}_actor_attribution.csv")
        pooled += attrs
        if not attrs:
            continue
        gain = np.array([float(r["g_t"]) for r in attrs])
        states = np.array([[float(r["X2"]), float(r["P2"])] for r in attrs])
        cuts = [np.quantile(states[:, j], [1/3, 2/3]) for j in range(2)]
        bins = np.column_stack([np.digitize(states[:, j], cuts[j]) for j in range(2)])
        for j, channel in enumerate(("P100", "F200")):
            residual = np.array([float(r[f"applied_residual_{channel}"]) for r in attrs])
            paired = np.array([float(r[f"paired_input_difference_{channel}"]) for r in attrs])
            groups = [("all", np.ones(len(attrs), bool)), ("economic_positive", gain > 0),
                      ("economic_negative", gain < 0)]
            groups += [(f"X2_{i}_P2_{k}", (bins[:, 0] == i)&(bins[:, 1] == k)) for i in range(3) for k in range(3)]
            for group, mask in groups:
                row = dict(geometry=name, channel=channel, group=group, **core.distribution(residual[mask]),
                    paired_closed_loop_difference_mean=float(paired[mask].mean()) if mask.any() else None,
                    mean_g=float(gain[mask].mean()) if mask.any() else None,
                    mean_X2=float(states[mask, 0].mean()) if mask.any() else None,
                    mean_P2=float(states[mask, 1].mean()) if mask.any() else None,
                    Pearson=core.correlation(residual[mask], gain[mask]),
                    Spearman=core.correlation(residual[mask], gain[mask], True),
                    Pearson_abs=core.correlation(abs(residual[mask]), gain[mask]),
                    Spearman_abs=core.correlation(abs(residual[mask]), gain[mask], True))
                selected = [attrs[i] for i in np.flatnonzero(mask)]
                utilization = [float(r[f"authority_utilization_{channel}"]) for r in selected
                               if r[f"authority_utilization_{channel}"] not in (None, "")]
                row.update(mean_authority_utilization=float(np.mean(utilization)) if utilization else None,
                    positive_authority_utilization_mean=float(np.mean([float(r[f"authority_utilization_{channel}"]) for r in selected
                        if float(r[f"applied_residual_{channel}"]) > 1e-6 and r[f"authority_utilization_{channel}"] not in (None, "")]))
                        if any(float(r[f"applied_residual_{channel}"]) > 1e-6 for r in selected) else None,
                    near_action_limit_fraction=float(np.mean([float(r["raw_actor_"+channel])**2 >= .99**2 for r in selected])) if selected else None,
                    QP_modification_count=sum(r["QP_modification"] == "True" for r in selected))
                behavior.append(row)
    write_csv(ROOT / "economic_component_decomposition.csv", components)
    write_csv(ROOT / "geometry_vs_sac_decomposition.csv", splits)
    write_csv(ROOT / "actor_residual_economic_attribution.csv", pooled)
    write_csv(ROOT / "actor_behavior_summary.csv", behavior)
    save(ROOT / "attribution_status.json", dict(status="COMPLETE" if not missing else "PARTIAL", missing=missing,
        correlation_caveat="correlation is not causation; local residual and paired closed-loop difference are distinct",
        state_bins="per-geometry pooled X2/P2 tertiles; labels 0=low,1=mid,2=high; not safety thresholds"))


def p100_scan(workers):
    protocol = locked()
    status = v2.load(ROOT / "attribution_status.json")
    if status["status"] != "COMPLETE":
        raise RuntimeError("Complete attribution first")
    env = s.env(); common = s.fixed_gates(env)
    contexts = s.snapshots(env)[0]
    rows, failures, authority_rows, tasks = [], [], [], []
    for c in protocol["p100_candidates"]:
        e = s.independent(env, c)
        np.testing.assert_array_equal(e[2].robust_input_lower[1:], env[2].robust_input_lower[1:])
        np.testing.assert_array_equal(e[2].robust_input_upper[1:], env[2].robust_input_upper[1:])
        report_file = ROOT / "p100_certificates" / (c["candidate"]+".json")
        if report_file.exists():
            report = v2.load(report_file)
            errors = report["failures"]
        else:
            report, errors = s.evaluate(e, c, common)
            report["failures"] = errors
            save(report_file, report)
        row = s.geometry_row(e, c)
        row.update(certificate=report["certificate"], failed_gates=";".join(r["failed_gate"] for r in errors),
            minimum_residual_authority=report["minimum_residual_authority"], W_max_facet_excess=report["coverage"]["max_normalized_facet_excess"])
        failures += errors
        if report["certificate"] == "CERTIFIED_PASS":
            authority = s.authority(e, contexts, c["candidate"])
            authority_rows += authority
            row.update({k: authority[0][k] for k in ("P100_applied_residual_min", "P100_applied_residual_max",
                "F200_applied_residual_min", "F200_applied_residual_max")})
            tasks += [("p100", c, seed) for seed in DEV]
        rows.append(row)
        write_csv(ROOT / "p100_certificate_sensitivity.csv", rows)
        write_csv(ROOT / "p100_certificate_failure_reasons.csv", failures)
        print(c["candidate"], report["certificate"], flush=True)
    write_csv(ROOT / "p100_state_authority.csv", authority_rows)
    for index, result in enumerate(execute(tasks, worker_path, workers)):
        print(f"P100 frozen counterfactual {index+1}/{len(tasks)} {result['geometry']} {result['validation_seed']} {result['status']}", flush=True)
    complete = [v2.load(p) for p in sorted((ROOT / "p100_pairs").glob("*/seed_*_result.json"))]
    write_csv(ROOT / "p100_frozen_policy_counterfactual.csv", [{k: v for k, v in r.items() if k != "components"} for r in complete])
    locked()


def worker_witness(parameter):
    index = parameter["candidate"]
    folder = ROOT / "witness_candidates" / f"candidate_{index:04d}"
    receipt = folder / "result.json"
    if receipt.exists():
        return v2.load(receipt)
    policy = core.StateAffineWitness(parameter["bias"], parameter["gain"], ENV[0].state_scale)
    per_path, failure = [], None
    try:
        for seed in DEV:
            path, base, _ = PATHS[seed]
            with torch.no_grad():
                records, _, _ = io.rollout(ENV, path, seed, policy)
            metric, _, _ = recovery.full_metrics(base, records, ENV, 0., np.zeros(5))
            parts, _ = core.economic_arrays(records, ENV)
            metrics = dict(validation_seed=seed, **{**metric, **s.economic_metrics(base, records)},
                component_costs=dict(zip(core.COMPONENTS, parts.sum(0).tolist())))
            per_path.append(metrics)
            # Compressed complete trajectory evidence, not thousands of large CSVs.
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"seed_{seed}.npz",
                state=np.array([r["state"] for r in records]), control=np.array([r["control"] for r in records]),
                raw_actor=np.array([r["raw_action"] for r in records]),
                mapped_residual=np.array([r["residual"] for r in records]),
                applied_residual=np.array([r["applied_residual"] for r in records]),
                disturbance=path, stage_components=parts, economic_cost=parts.sum(1),
                mode=np.array([r["safety_mode"] for r in records]), w_hat=np.array([r["w_hat"] for r in records]))
    except Exception as exc:
        failure = repr(exc)
        if getattr(exc, "evidence", None):
            write_csv(folder / "failure_evidence.csv", exc.evidence)
    safe = len(per_path) == len(DEV) and failure is None
    result = dict(candidate=index, parameters=parameter, safety_passed=safe, failure=failure,
        completed_paths=len(per_path), per_path=per_path, search="finite_state_dependent_witness_search")
    if safe:
        result.update(mean_improvement_pct=float(np.mean([r["economic_improvement_pct"] for r in per_path])),
            worst_path_improvement_pct=min(r["economic_improvement_pct"] for r in per_path),
            all_validation_positive=all(r["economic_improvement_pct"] > 0 for r in per_path),
            minimum_X2_margin=min(r["minimum_X2_margin"] for r in per_path),
            control_quality_pass=all(all(r[k] <= 1.10 for k in ("P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio")) for r in per_path))
        for key in ("economic_win_fraction", "J_econ", "P100_TV", "F200_TV", "P100_RMS_du", "F200_RMS_du",
                    "P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio", "X2_std", "centered_X2_MAE",
                    "X2_IAE_ratio", "P2_IAE_ratio", "X2_ISE_ratio", "P2_ISE_ratio", "W_exceedance_rate",
                    "P100_sign_change_ratio", "F200_sign_change_ratio"):
            result[key] = float(np.mean([r[key] for r in per_path]))
        for key in io.SAFETY:
            result[key] = int(sum(r[key] for r in per_path))
    save(receipt, result)
    return result


def witness_search(workers):
    p = locked()
    rows = io.read_csv(ROOT / "p100_certificate_sensitivity.csv")
    if len(rows) != 5:
        raise RuntimeError("Finish the preregistered P100 scan before witness search")
    expected = sum(r["certificate"] == "CERTIFIED_PASS" for r in rows)*10
    done = io.read_csv(ROOT / "p100_frozen_policy_counterfactual.csv")
    if len(done) != expected or any(r["status"] != "COMPLETE" for r in done):
        raise RuntimeError("Finish all P100 counterfactuals first")
    for index, result in enumerate(execute(p["witness_parameters"], worker_witness, workers)):
        print(f"state-only witness {index+1}/512: candidate={result['candidate']} safe={result['safety_passed']} "
              f"gain={result.get('mean_improvement_pct')} quality={result.get('control_quality_pass')}", flush=True)
        summarize_witness()
    locked()


def summarize_witness():
    rows = [v2.load(p) for p in sorted((ROOT / "witness_candidates").glob("*/result.json"))]
    flat = [{k: v for k, v in r.items() if k not in ("parameters", "per_path")} for r in rows]
    front = core.pareto_front(flat)
    write_csv(ROOT / "state_dependent_witness_results.csv", flat)
    write_csv(ROOT / "state_dependent_witness_pareto.csv", front)
    safe = [r for r in flat if r["safety_passed"]]
    quality = [r for r in safe if r["control_quality_pass"]]
    best = max(safe, key=lambda r: r["mean_improvement_pct"]) if safe else None
    best_quality = max(quality, key=lambda r: r["mean_improvement_pct"]) if quality else None
    save(ROOT / "state_dependent_witness_summary.json", dict(status="COMPLETE" if len(rows) == 512 else "PARTIAL",
        preregistered_budget=512, completed_candidates=len(rows), safe_candidates=len(safe),
        positive_mean_candidates=sum(r["mean_improvement_pct"] > 0 for r in safe),
        all_validation_positive_candidates=sum(r["all_validation_positive"] for r in safe),
        best_safe=best, best_control_quality=best_quality, certification_scope="Gaussian empirical only",
        claim="best found in finite state-only family, NOT global optimum or oracle upper bound",
        frozen_SAC_reference_gain_pct=.08650902042768663))
    # Save human-readable witness trajectories for both economy and quality representatives.
    for role, r in (("best_safe", best), ("best_control_quality", best_quality)):
        if r is None:
            continue
        index = r["candidate"]
        source = ROOT / "witness_candidates" / f"candidate_{index:04d}" / "seed_420000.npz"
        with np.load(source) as data:
            records = [dict(step=k, X2=data["state"][k, 0], P2=data["state"][k, 1],
                P100=data["control"][k, 0], F200=data["control"][k, 1],
                raw_P100=data["raw_actor"][k, 0], raw_F200=data["raw_actor"][k, 1],
                economic_cost=data["economic_cost"][k], mode=data["mode"][k]) for k in range(len(data["state"]))]
        write_csv(ROOT / "representative_witnesses" / f"{role}_candidate_{index:04d}_seed_420000.csv", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "audit", "p100", "witness", "report", "status"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "audit":
        audit(args.workers)
    elif args.phase == "p100":
        p100_scan(args.workers)
    elif args.phase == "witness":
        witness_search(args.workers)
    elif args.phase == "report":
        from .report import report
        report()
    else:
        for name in ("attribution_status.json", "state_dependent_witness_summary.json", "next_stage_decision_summary.json"):
            p = ROOT / name
            print(name, json.dumps(v2.load(p)) if p.exists() else "NOT_RUN")


if __name__ == "__main__":
    main()
