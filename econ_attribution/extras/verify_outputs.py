"""Read-only numerical reconciliation of completed offline evidence."""
import numpy as np
from .. import study


def main():
    protocol = study.locked()
    root = study.ROOT
    split = study.io.read_csv(root / "geometry_vs_sac_decomposition.csv")
    parts = study.io.read_csv(root / "economic_component_decomposition.csv")
    assert len(split) == 50 and len(parts) == 200
    for r in split:
        k = (r["geometry"], int(r["validation_seed"]))
        assert k[1] in protocol["development_seeds"]
        subset = [c for c in parts if (c["geometry"], int(c["validation_seed"])) == k]
        np.testing.assert_allclose(sum(float(c["absolute_change"]) for c in subset), float(r["DeltaJ_total"]), rtol=0, atol=1e-7)
        np.testing.assert_allclose(float(r["DeltaJ_geometry"])+float(r["DeltaJ_SAC_given_geometry"]), float(r["DeltaJ_total"]), rtol=0, atol=1e-7)
        g, s, t = [float(r[key]) for key in ("Gain_geometry_pct", "Gain_SAC_given_geometry_pct", "Gain_total_vs_prod_pct")]
        np.testing.assert_allclose(g+(1-g/100)*s, t, rtol=0, atol=1e-10)
        assert all(int(r[key]) == 0 for key in study.io.SAFETY)
    certs = study.io.read_csv(root / "p100_certificate_sensitivity.csv")
    assert len(certs) == 5
    first = certs[0]
    for r in certs:
        report = study.v2.load(root / "p100_certificates" / (r["candidate"]+".json"))
        assert report["certificate"] == r["certificate"]
        if r["certificate"] == "CERTIFIED_PASS":
            assert all(report["gates"].values())
            assert float(r["minimum_residual_authority"]) >= .05-1e-12
        for key in ("hF", "rhoF", "UR_F200_lower", "UR_F200_upper", "nominal_tight_F200_lower", "nominal_tight_F200_upper", "F200_applied_residual_max"):
            assert r[key] == first[key]
    cf = study.io.read_csv(root / "p100_frozen_policy_counterfactual.csv")
    expected = 10*sum(r["certificate"] == "CERTIFIED_PASS" for r in certs)
    assert len(cf) == expected and all(r["status"] == "COMPLETE" for r in cf)
    for r in cf:
        assert int(r["validation_seed"]) in study.DEV
        assert all(int(r[key]) == 0 for key in study.io.SAFETY)
        np.testing.assert_allclose(float(r["DeltaJ_geometry"])+float(r["DeltaJ_SAC_given_geometry"]), float(r["DeltaJ_total"]), rtol=0, atol=1e-7)
        # The +5% experiment must reproduce the old C2_8 intervention.
        if r["geometry"] == "P100_1050":
            old = study.v2.load(root / "geometry_pairs/C2_8" / f"seed_{r['validation_seed']}_result.json")
            np.testing.assert_allclose(float(r["J_SAC_geometry"]), old["J_SAC_geometry"], rtol=0, atol=1e-6)
    assert len(protocol["witness_parameters"]) == 512
    assert study.locked()["source_and_production_hashes"] == study.protected()
    study.save(root / "output_reconciliation_receipt.json", dict(passed=True,
        attribution_paths=50, component_rows=200, p100_certificates=5, p100_paths=len(cf),
        frozen_C2_8_reproduced=True, production_hashes_unchanged=True,
        source_sha256=study.io.digest(__file__), witness_not_claimed_complete=True,
        safety_scope="Gaussian empirical only; no continuous-domain claim"))
    print("50 attribution paths and 50 P100 paths reconciled; source/policy hashes unchanged")


if __name__ == "__main__":
    main()
