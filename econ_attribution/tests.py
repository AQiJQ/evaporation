"""Attribution contracts; optional full legacy suite, never SAC training."""
import argparse
from copy import deepcopy
import importlib
import inspect
import subprocess
import unittest
import numpy as np
from . import core, study
from ..model import EvaporatorModel
from ..config import ExperimentConfig


class EconomicAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = study.s.env()
        cls.geometry = study.s.geometry_hashes(cls.env)

    def test_01_components_match_runtime_total(self):
        rng = np.random.default_rng(42)
        model = self.env[1]
        for _ in range(30):
            x = [rng.uniform(25.1, 27.), rng.uniform(48., 52.)]
            u = [rng.uniform(160., 225.), rng.uniform(190., 240.)]
            w = [10., 5., 40., 25.]
            self.assertAlmostEqual(core.cost_components(model, x, u, w).sum(), model.economic_cost(x, u, w), places=9)

    def test_02_absolute_geometry_sac_identity(self):
        result = core.effect_split(1000., 1100., 990.)
        self.assertEqual(result["DeltaJ_geometry"]+result["DeltaJ_SAC_given_geometry"], result["DeltaJ_total"])

    def test_03_different_percentage_denominators(self):
        r = core.effect_split(1000., 1100., 990.)
        self.assertAlmostEqual(r["Gain_total_vs_prod_pct"], 1.)
        self.assertNotAlmostEqual(r["Gain_geometry_pct"]+r["Gain_SAC_given_geometry_pct"], 1.)
        self.assertAlmostEqual(r["Gain_geometry_pct"]+(1-r["Gain_geometry_pct"]/100)*r["Gain_SAC_given_geometry_pct"], 1.)

    def test_04_residual_sign_distribution(self):
        r = core.distribution([-2., -1., 0., 1., 2.])
        self.assertEqual((r["positive_fraction"], r["negative_fraction"], r["zero_fraction"]), (.4, .4, .2))
        self.assertEqual(r["mean"], 0.)

    def test_05_P100_does_not_change_F200_geometry(self):
        for c in study.p100_candidates():
            e = study.s.independent(self.env, c)
            for name in ("robust_input_lower", "robust_input_upper", "u_lower_tight", "u_upper_tight"):
                self.assertEqual(getattr(e[2], name)[1], getattr(self.env[2], name)[1])
            self.assertEqual(study.s.geometry_hashes(e), self.geometry)

    def test_06_coverage_deterministic(self):
        e = study.s.independent(self.env, study.p100_candidates()[0])
        e[0].robust_region_random_samples = 16
        a, b = study.s.coverage(e), study.s.coverage(e)
        self.assertEqual(a["max_normalized_facet_excess"], b["max_normalized_facet_excess"])
        self.assertTrue(a["passed"])

    def test_07_frozen_actor_hash_and_sources(self):
        source = study.v2.load(study.s.ROOT / "counterfactual_sources.json")
        self.assertEqual(study.io.digest(study.ACTOR), source["actor_sha256"])

    def test_08_fixed_finite_parameter_registration(self):
        a, b = core.witness_parameters(), core.witness_parameters()
        self.assertEqual(len(a), 512)
        self.assertEqual(a, b)
        self.assertEqual([r["candidate"] for r in a], list(range(512)))
        with self.assertRaises(ValueError):
            core.witness_parameters(513)

    def test_09_witness_uses_only_observable_state(self):
        policy = core.StateAffineWitness([.1, -.2], [[2., .3], [-1., .4]], self.env[0].state_scale)
        obs = np.zeros(23, dtype=np.float32); obs[:2] = [.01, -.02]
        a = policy.select_action(obs)
        obs[2:] = np.arange(21)+100
        np.testing.assert_array_equal(a, policy.select_action(obs))
        self.assertLessEqual(np.max(abs(a)), 1.)

    def test_10_complete_online_mapping_QP_and_raw_mapped_applied_logging(self):
        e = study.s.independent(self.env, study.p100_candidates()[0])
        e[0].steps_per_episode = 4
        path = np.tile(e[0].disturbance_nominal, (4, 1))
        policy = core.StateAffineWitness([-.3, .25], [[1., .2], [.3, -.4]], e[0].state_scale)
        base = study.io.rollout(e, path, 420000, study.ZeroResidualPolicy())[0]
        records = study.io.rollout(e, path, 420000, policy)[0]
        rows = core.replay_attribution(e, records, base, "fixture", 420000)
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertFalse(r["QP_modification"])
            for name in ("P100", "F200"):
                self.assertAlmostEqual(r["QP_input_"+name], r["final_input_"+name], places=8)
                self.assertAlmostEqual(r["final_input_"+name]-r["local_zero_input_"+name], r["applied_residual_"+name], places=8)
                self.assertAlmostEqual(r["scaled_actor_"+name], .1*r["raw_actor_"+name], places=8)

    def test_11_development_only_no_final_seeds(self):
        self.assertEqual(study.DEV, tuple(range(420000, 420010)))
        self.assertTrue(set(study.DEV).isdisjoint(study.io.RESERVED))
        self.assertTrue(all(c["rhoF"] == study.s.RHO and c["hF"] == 30 for c in study.p100_candidates()))

    def test_12_components_timing_and_disturbance_identity(self):
        e = study.s.independent(self.env, study.p100_candidates()[0]); e[0].steps_per_episode = 4
        path = np.tile(e[0].disturbance_nominal, (4, 1))
        records = study.io.rollout(e, path, 420001, study.ZeroResidualPolicy())[0]
        values, nxt = core.economic_arrays(records, e)
        np.testing.assert_allclose(nxt[:-1], np.array([r["state"] for r in records[1:]]), atol=1e-8)
        np.testing.assert_array_equal(path, np.array([r["disturbance"] for r in records]))
        np.testing.assert_allclose(values.sum(1), [r["economic_cost"] for r in records], rtol=0, atol=1e-7)

    def test_13_pearson_spearman_ties_and_degenerate_case(self):
        self.assertEqual(core.correlation([0., 0.], [1., 2.]), None)
        np.testing.assert_array_equal(core.midranks([1, 1, 4]), [.5, .5, 2.])
        self.assertAlmostEqual(core.correlation([1, 1, 4], [1, 1, 8], True), 1.)

    def test_14_original_safety_and_production_never_mutated(self):
        self.assertEqual(study.s.geometry_hashes(self.env), self.geometry)
        definition = core.cost_definition(self.env[1])
        self.assertIn("not explicitly specified", definition["unit"])
        self.assertTrue(definition["training_reward_is_distinct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EconomicAttributionTests)
    integrations = []
    if args.all:
        for path in sorted(study.io.REPO.glob("*tests.py")):
            module = importlib.import_module("evaporation."+path.stem)
            loaded = unittest.defaultTestLoader.loadTestsFromModule(module)
            suite.addTests(loaded)
            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("test_") and fn.__module__ == module.__name__ and not inspect.signature(fn).parameters:
                    suite.addTest(unittest.FunctionTestCase(fn))
            if loaded.countTestCases() == 0 and hasattr(module, "main"):
                integrations.append(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    fallback = None
    if args.all and result.wasSuccessful() and result.skipped:
        exe = study.io.REPO / ".venv-ecc2019/Scripts/python.exe"
        command = [str(exe), "-m", "unittest", "evaporation.ecc2019_reproduction_tests.SolverTests", "-v"]
        p = subprocess.run(command, cwd=study.io.REPO.parent, capture_output=True, text=True)
        print(p.stdout, p.stderr, flush=True)
        fallback = dict(passed=p.returncode == 0, tests=3)
    passed = result.wasSuccessful() and (not result.skipped or fallback and fallback["passed"])
    if args.all and passed:
        for module in [importlib.import_module("evaporation.test"), *integrations]:
            module.main()
    study.save(study.ROOT / "tests_receipt.json", dict(passed=bool(passed), unit_tests=result.testsRun,
        optional_CasADi=fallback, full_suite=args.all, source_hashes=study.protected(), training_performed=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
