"""Independent replay of the original balanced-B region search (no SAC).

This replay is NOT the fixed K/W/Z/S/Omega sensitivity experiment: the
original algorithm rebuilds W and optimizes K/M in its own local process.
Production archives are read-only. Every attempted rho is journalled.
"""
from pathlib import Path
import argparse
import numpy as np
from . import certified_reference_search as search
from . import zanon2019_economic_recovery as exp
from . import zanon2019_paired_experiment as io
from .theta_learning import OnlineThetaLearner

ROOT = exp.ROOT / 'ur_certificate_sensitivity'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root != ROOT.resolve():
        raise ValueError('Independent audit root required')
    path = root / 'rho_search_reconstruction.json'
    if path.exists():
        raise FileExistsError('Preserve existing reconstruction evidence')
    root.mkdir(parents=True, exist_ok=True)
    cfg = search._candidate_config(np.array([25.39, 50.125]), 42)
    report = dict(status='RUNNING', reconstruction_only=True, training=False,
        original_failed_scale_logs_archived=False, state=[25.39, 50.125], seed=42,
        initial_rho=1., growth=cfg.robust_region_growth_factor,
        max_search_scale=cfg.robust_region_max_scale,
        bisection_iterations=cfg.robust_region_bisection_iterations,
        tolerance=cfg.robust_region_membership_tolerance,
        nonlinear_sampling=dict(random_samples=cfg.robust_region_random_samples,
            corners='all X_R x U_R x D vertex combinations',
            rng='seed + 81000 + attempt_index',
            claim='finite sampling coverage, NOT continuous-domain nonlinear proof'),
        acceptance='all original _candidate_region_diagnostics and final evaluate_candidate gates',
        production_scale=1.1083984375,
        distinction='Original replay may redesign K/M/W/Z/S locally; sensitivity freezes them',
        attempts=[])
    io.save(path, report)
    original = OnlineThetaLearner._candidate_region_diagnostics
    def logged(self, scale, design, attempt_index):
        result, row = original(self, scale, design, attempt_index)
        report['attempts'].append(row)
        io.save(path, report)
        print('rho replay', attempt_index, scale, row['feasible'], row['failure_reason'], flush=True)
        return result, row
    OnlineThetaLearner._candidate_region_diagnostics = logged
    try:
        designs = root / 'reconstruction_designs'
        designs.mkdir(exist_ok=True)
        model = search.EvaporatorModel(cfg)
        paper = np.array([25., 49.743])
        cost = model.economic_cost(paper, model.steady_input(paper), cfg.disturbance_nominal)
        row = search.evaluate_candidate(np.array([25.39, 50.125]), 'reconstruction', 42, cost, designs)
        report.update(status='COMPLETE', final=row,
            max_accepted_rho=max((r['scale'] for r in report['attempts'] if r['feasible']), default=None),
            selected_rho=row.get('robust_region_scale'),
            matches_archived_selected_scale=bool(row.get('robust_region_scale') == 1.1083984375))
    except Exception as error:
        report.update(status='NUMERIC_FAIL', error=repr(error))
        raise
    finally:
        OnlineThetaLearner._candidate_region_diagnostics = original
        io.save(path, report)

if __name__ == '__main__':
    main()
