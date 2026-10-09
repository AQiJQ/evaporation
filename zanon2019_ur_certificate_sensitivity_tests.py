"""Fixed-certificate sensitivity contracts. No training or final disturbances."""
import inspect
import uuid
from pathlib import Path
import unittest
import numpy as np
from . import zanon2019_ur_certificate_sensitivity as a

class CertificateSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=a.env();cls.hashes=a.geometry_hashes(cls.e)
        cls.baseline=a.independent(cls.e,a.candidates()[0])
        cls.common=a.fixed_gates(cls.e)

    def test_01_baseline_UR_and_nominal_reproduction(self):
        m,d=self.baseline[1:3]
        np.testing.assert_allclose(m.physical_input(d.robust_input_upper),[233.65538828079238,249.52860663022335],rtol=0,atol=1e-10)
        np.testing.assert_allclose(d.u_lower_tight,self.e[2].u_lower_tight,rtol=0,atol=1e-10)
        np.testing.assert_allclose(d.u_upper_tight,self.e[2].u_upper_tight,rtol=0,atol=1e-10)

    def test_02_nonmutation(self):
        old=a.array_hash(self.e[2].robust_input_upper)
        a.independent(self.e,a.candidates()[-1])
        self.assertEqual(old,a.array_hash(self.e[2].robust_input_upper))
        self.assertEqual(self.hashes,a.geometry_hashes(self.e))

    def test_03_shared_scaling(self):
        c=next(c for c in a.candidates() if c['candidate']=='A5')
        e=a.independent(self.e,c)
        half=e[1].physical_input(e[2].robust_input_upper)-e[0].robust_economic_reference_input
        np.testing.assert_allclose(half,a.RHO*1.2*np.array([35,30]))

    def test_04_independent_scaling(self):
        c=next(c for c in a.candidates() if c['candidate']=='C4')
        e=a.independent(self.e,c)
        np.testing.assert_allclose(e[2].robust_input_upper[0],self.e[2].robust_input_upper[0])
        self.assertGreater(e[2].robust_input_upper[1],self.e[2].robust_input_upper[1])

    def test_05_full_gate_reproducibility(self):
        # Small fixture only: official study always uses all 10000 samples.
        e=a.independent(self.e,a.candidates()[0]);e[0].robust_region_random_samples=16
        one,_=a.evaluate(e,a.candidates()[0],self.common)
        two,_=a.evaluate(e,a.candidates()[0],self.common)
        self.assertEqual(one['gates'],two['gates'])
        self.assertTrue(all(one['gates'].values()),one['gates'])
        self.assertEqual(one['coverage']['max_normalized_facet_excess'],two['coverage']['max_normalized_facet_excess'])

    def test_06_failure_reasons_preserved(self):
        c=next(c for c in a.candidates() if c['candidate']=='A0')
        e=a.independent(self.e,c);e[0].robust_region_random_samples=0
        report,errs=a.evaluate(e,c,self.common)
        self.assertEqual(report['certificate'],'CERTIFIED_FAIL')
        self.assertIn('reference_input',[r['failed_gate'] for r in errs])
        # Windows sandbox TemporaryDirectory ACLs can deny child writes.
        # Ordinary unique diagnostic directories preserve failure evidence.
        root=a.ROOT/'test_evidence'/uuid.uuid4().hex;root.mkdir(parents=True)
        a.export(root,[a.geometry_row(e,c)],errs,[],[])
        self.assertIn('reference_input',(root/'ur_certificate_failure_reasons.csv').read_text())

    def test_07_frozen_geometry_hashes(self):
        for c in a.candidates():self.assertEqual(self.hashes,a.geometry_hashes(a.independent(self.e,c)))

    def test_08_alpha_frozen(self):
        for c in a.candidates():self.assertEqual(a.independent(self.e,c)[0].stochastic_residual_scale,.1)

    def test_09_actor_weights_readonly(self):
        path=a.exp.run_dir(a.exp.ROOT,'strong',42)/'models/episode_0100_actor.pth'
        before=a.prior.sha(path);actor=a.make_agent(self.e[0],'cpu');a.load_actor(actor,path)
        self.assertEqual(before,a.prior.sha(path))

    def test_10_paired_archived_disturbances(self):
        paths,sources=a.development_paths()
        self.assertEqual(tuple(paths),a.DEV)
        for seed,(path,base,sac) in paths.items():
            self.assertEqual(a.array_hash(path),sources[seed]['disturbance_sha256'])
            np.testing.assert_array_equal(path,np.array([r['disturbance'] for r in base]))

    def test_11_actual_mapping_authority(self):
        e=self.baseline
        ctx=dict(role='reference_reset',step=0,state=e[0].robust_economic_reference_state,z=e[2].z_ref)
        rows=a.authority(e,[ctx],'baseline')
        r=rows[0]
        self.assertLess(r['authority_reconstruction_error'],1e-6)
        self.assertEqual(r['QP_modification_count'],0)
        self.assertAlmostEqual(r['F200_applied_residual_max'],.4048161228,places=8)

    def test_12_no_final_seeds_or_training(self):
        self.assertEqual(a.DEV,tuple(range(420000,420010)))
        self.assertFalse(set(a.DEV)&set(range(430000,430050)))
        self.assertNotIn('sample_disturbance_path',inspect.getsource(a.development_paths))
        self.assertNotIn('training=True',inspect.getsource(a))

    def test_13_original_output_protection(self):
        self.assertNotEqual(a.ROOT,a.exp.ROOT)
        self.assertTrue(a.ROOT.name.startswith('ur_certificate_'))
        self.assertIn('Preserve prior study',inspect.getsource(a.main))

    def test_14_preregistration_and_failure_classification(self):
        self.assertEqual(len(a.candidates()),31)
        text=inspect.getsource(a.main)
        self.assertLess(text.index("io.save(root/'protocol.json'"),text.index('evaluate(ce,c,common)'))
        self.assertIn('NUMERIC_FAIL',text)
        self.assertIn('NOT_EVALUATED',text)

if __name__=='__main__':unittest.main()
