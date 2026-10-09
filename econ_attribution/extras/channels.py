"""Exact affine-in-state/input algebraic cost-channel identity.

This is NOT an affine plant approximation. The runtime algebraic cost itself
is affine in post-state/P100/F200 at fixed current disturbance. It distinguishes
direct input-cost changes from post-state changes without interpreting a
residual/economic correlation as causal proof of closed-loop mechanisms.
"""
from pathlib import Path
import numpy as np
from .. import study


def coefficients(cfg, disturbance):
    f1 = float(disturbance[0])
    steam_minus_flow = 600./cfg.latent_steam - 10.09/cfg.latent_evaporation
    heat = steam_minus_flow*cfg.ua1_factor*(f1+cfg.recirculation_f3)
    state_term = -heat+10.09*f1*cfg.cp/cfg.latent_evaporation
    return np.array([state_term*cfg.t2_x_coeff, state_term*cfg.t2_p_coeff,
                     heat*cfg.t100_p_coeff, .6])


def main():
    study.locked()
    env = study.s.env(); cfg, model = env[:2]
    paths, _ = study.s.development_paths()
    names = ("post_X2", "post_P2", "direct_P100", "direct_F200")
    summary = []
    for geometry in study.GEOMETRIES:
        folder = study.ROOT / "geometry_pairs" / geometry
        for seed in study.DEV:
            files = {role: folder / f"seed_{seed}_{role}_cost_components.csv"
                     for role in ("production_baseline", "own_baseline", "frozen_SAC")}
            if not all(p.exists() for p in files.values()):
                continue
            records = {role: study.io.read_csv(path) for role, path in files.items()}
            for effect, baseline_role, candidate_role in (
                ("geometry", "production_baseline", "own_baseline"),
                ("SAC_given_geometry", "own_baseline", "frozen_SAC"),
                ("total", "production_baseline", "frozen_SAC")):
                components = []
                for k, (b, s) in enumerate(zip(records[baseline_role], records[candidate_role])):
                    vb, vs = (np.array([float(row[key]) for key in ("next_X2", "next_P2", "P100", "F200")]) for row in (b, s))
                    # Test the actual nonlinear algebraic cost identity at each stage.
                    delta = coefficients(cfg, paths[seed][0][k])*(vs-vb)
                    actual = float(s["total_economic_stage_cost"])-float(b["total_economic_stage_cost"])
                    np.testing.assert_allclose(delta.sum(), actual, rtol=0, atol=1e-8)
                    components.append(delta)
                totals = np.array(components).sum(0)
                for j, name in enumerate(names):
                    summary.append(dict(geometry=geometry, validation_seed=seed, effect=effect, channel=name,
                        absolute_cost_change=float(totals[j]), gain_component=-float(totals[j]),
                        interpretation="exact additive fixed-disturbance algebraic identity, not full dynamic causal attribution"))
    study.write_csv(study.ROOT / "economic_channel_decomposition.csv", summary)
    study.save(study.ROOT / "economic_channel_definition.json", dict(
        identity="Delta ell = c_X2(d)*Delta next_X2 + c_P2(d)*Delta next_P2 + c_P100(d)*Delta P100 + 0.6*Delta F200",
        c_Q="600/latent_steam - 10.09/latent_evaporation",
        c_heat="c_Q * ua1_factor*(F1+F3)", c_state="-c_heat + 10.09*F1*cp/latent_evaporation",
        c_X2="c_state*t2_x_coeff", c_P2="c_state*t2_p_coeff", c_P100="c_heat*t100_p_coeff",
        reference_coefficients=coefficients(cfg, cfg.disturbance_nominal),
        runtime_model_sha256=study.io.digest(Path(study.core.inspect.getsourcefile(type(model)))),
        diagnostic_source_sha256=study.io.digest(Path(__file__)),
        no_plant_linearization=True, per_stage_identity_tolerance=1e-8, physical_cost_not_reward=True))
    print("Exact direct-input/post-state economic channel identity verified", len(summary), "rows")


if __name__ == "__main__":
    main()
