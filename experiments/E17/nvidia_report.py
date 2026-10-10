"""Saved-result verification and NVIDIA-only report/plots; no GPU execution."""
import json
import math
import os
import numpy as np
from .common import HERE, sha, save, verify_plan, describe
from .summarize import paired
from .nvidia_summary import main as audit_saved


def refresh_completed_status():
    """Keep the Apple report intact while replacing its obsolete access status."""
    report = (HERE/'REPORT.md').read_text()
    start = report.index('## NVIDIA status and defensible conclusions')
    end = report.index('## Reproduction and artifacts', start)
    report = report[:start] + '''## NVIDIA status and defensible conclusions

NVIDIA execution is complete on L40S (`dash2`): full accuracy audit, 15 paired repetitions across five balanced startup rotations, actual kernel dispatch evidence and 136 passing GPU tests. See [NVIDIA_REPORT.md](NVIDIA_REPORT.md) and [CUDA throughput plot](cuda_throughput.png). The unchanged original plan and the predeclared fifth-rotation supplement are retained.

Custom CUDA/Triton pass numerical/application gates but establish no throughput advantage over the strict baseline. Several differences favor strict PyTorch; others remain inconclusive. Fused stock and compiled PyTorch execute their optimizations but fail the predeclared top-10 membership gate at K=100, so their timings are diagnostic only. There is no accuracy-qualified custom-versus-fused winner. Apple M5 and NVIDIA L40S results cannot be interpreted as language-level speedups.

''' + report[end:]
    old = 'Once NVIDIA authentication is restored, copy the frozen E17 code/plan'
    if old in report:
        report = report[:report.index(old)] + 'NVIDIA launch, fifth-rotation supplement, saved-result verification and post-timing GPU tests are documented in [NVIDIA_REPORT.md](NVIDIA_REPORT.md). All three Slurm jobs completed with exit code 0:0; no other person’s session or job was altered.\n'
    (HERE/'REPORT.md').write_text(report)
    hardware = json.loads((HERE/'hardware_status.json').read_text())
    hardware.update(status='Apple and NVIDIA benchmarks complete and audited',
                    nvidia_access='Authenticated shared socket restored; L40S jobs 172869, 172870, 172871 completed sequentially with exit code 0:0.',
                    nvidia_verification_tests={'passed': 136, 'skipped': 0},
                    nvidia_timing={'sessions': 5, 'paired_repetitions': 15, 'startup_order_balanced': True})
    save(HERE/'hardware_status.json', hardware)


def main():
    plan = verify_plan()
    audit_saved()
    accuracy = json.loads((HERE/'cuda_accuracy.json').read_text())
    summary = json.loads((HERE/'cuda_summary.json').read_text())
    balance = json.loads((HERE/'nvidia_balance_plan.json').read_text())
    assert balance['benchmark_plan_sha256'] == sha(HERE/'plan.json')
    assert balance['runner_sha256'] == sha(HERE/'nvidia_balance.py')
    sessions = [json.loads((HERE/f'cuda_direct_s{s}.json').read_text()) for s in range(5)]
    assert all(session['status'] == 'complete' for session in sessions)
    for position in range(5):
        assert set(session['order'][position] for session in sessions) == set(accuracy['accuracy'])
    runs = [dict(r, session=s) for s, session in enumerate(sessions) for r in session['runs']]
    paths = list(accuracy['accuracy'])
    dispatch = json.loads((HERE/'cuda_dispatch_probe.json').read_text())
    for path, evidence in dispatch['paths'].items():
        assert evidence['scores_finite']
        if path in ('cuda', 'triton'):
            assert len(evidence['custom_operation_calls']) == 8
            assert all(c['fallback'] is False for c in evidence['custom_operation_calls'])
            needle = 'void fused<128>' if path == 'cuda' else 'fused_residual_layer_norm'
            assert any(needle in name for name in evidence['cuda_kernel_names'])
    assert dispatch['paths']['stock']['fused_encoder_calls'] == 4
    jobs = json.loads((HERE/'cuda_job_status.json').read_text())
    assert all(job['JobState'] == 'COMPLETED' and job['ExitCode'] == '0:0' for job in jobs['jobs'].values())
    comparisons = []
    against_stock = []
    total_pairs = 0
    for scope in ('forward', 'direct'):
        for k in plan['document_counts']:
            groups = {p: [r for r in runs if (r['scope'], r['documents'], r['path']) == (scope, k, p)] for p in paths}
            for path in paths:
                if path != 'stock':
                    against_stock.append({'scope': scope, 'documents': k, 'path': path,
                                          'diagnostic_only': True,
                                          **paired(groups['stock'], groups[path], 'pairs_per_second')})
            for path, observations in groups.items():
                assert len(observations) == 15
                for run in observations:
                    assert len(run['batch_ms']) == len(run['batch_pairs'])
                    assert all(t > 0 and math.isfinite(t) for t in run['batch_ms'])
                    assert all(0 < n <= 32 for n in run['batch_pairs'])
                    assert run['pairs'] == 16*k == sum(run['batch_pairs'])
                    assert math.isclose(run['pairs_per_second'], run['pairs']/run['seconds'], rel_tol=1e-12)
                    total_pairs += run['pairs']
                if path == 'strict':
                    continue
                result = paired(groups['strict'], observations, 'pairs_per_second')
                # Independently reconstruct the session ratios and seeded bootstrap.
                ratios = []
                for s in range(5):
                    left = {r['repeat']: r for r in groups['strict'] if r['session'] == s}
                    right = {r['repeat']: r for r in observations if r['session'] == s}
                    assert set(left) == set(right) == {0, 1, 2}
                    ratios.append(np.mean([math.log(left[r]['seconds']/right[r]['seconds']) for r in range(3)]))
                draws = np.random.default_rng(1701).choice(ratios, (10000, 5)).mean(axis=1)
                np.testing.assert_allclose(result['ratio_ci95'], np.exp(np.quantile(draws, [.025, .975])), rtol=1e-12)
                comparisons.append({'scope': scope, 'documents': k, 'path': path,
                                    'diagnostic_only': not accuracy['accuracy'][path]['accuracy_qualified'], **result})
    save(HERE/'cuda_supplement.json', {'comparisons_against_strict': comparisons,
         'diagnostic_comparisons_against_stock': against_stock,
         'timed_pair_executions': total_pairs, 'timed_runs': len(runs),
         'report_source_sha256': sha(__file__),
         'independent_checks': 'Throughput, batch/count totals, session ratios and seeded bootstrap reconstructed from raw observations.',
         'order': [s['order'] for s in sessions],
         'ordering': 'Five cyclic rotations cover every startup position once per path. Original four-session results retained separately in cuda_summary.json.',
         'supplement_plan_sha256': sha(HERE/'nvidia_balance_plan.json')})
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/e17-mpl-cache')
    os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/e17-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {'strict': '#245c95', 'cuda': '#bd5b20', 'triton': '#258066', 'stock': '#777777', 'compiled': '#a68ba9'}
    for ax, scope in zip(axes, ('forward', 'direct')):
        for path in paths:
            values = [describe([r['pairs_per_second'] for r in runs if (r['scope'], r['documents'], r['path']) == (scope, k, path)]) for k in plan['document_counts']]
            med = np.array([v['median'] for v in values])
            bounds = np.array([v['iqr'] for v in values]).T
            diagnostic = not accuracy['accuracy'][path]['accuracy_qualified']
            ax.errorbar(plan['document_counts'], med, yerr=[med-bounds[0], bounds[1]-med],
                        marker='o', capsize=4, color=colors[path], linestyle='--' if diagnostic else '-',
                        label=path + (' (diagnostic)' if diagnostic else ''))
        ax.set(title='Prepared forward' if scope == 'forward' else 'Complete direct scoring',
               xlabel='Documents per query fixture', ylabel='Pairs / second', xticks=[10, 50, 100])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('NVIDIA L40S · FP32 · microbatch 32 · median and IQR of 15 repetitions', fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE/'cuda_throughput.png', dpi=160)
    plt.close(fig)
    lines = ['# E17 NVIDIA benchmark results', '',
             'The restored SSH connection enabled the frozen benchmark on an NVIDIA L40S (`dash2`, Slurm job 172869). Apple HTTP results remain in [REPORT.md](REPORT.md); no NVIDIA HTTP or cross-hardware speedup is claimed.', '',
             'Original checkpoint SHA-256: `' + plan['checkpoint_sha256'] + '`. Plan SHA-256: `' + sha(HERE/'plan.json') + '`. FP32, TF32 disabled, eval/inference_mode, cap 128, pair microbatch 32. Accuracy: all 485 queries at each K, 77,600 pairs/path. Timing: the preselected 16 queries, 15 paired repetitions across five fresh processes, warmed shapes and synchronized GPU boundaries.', '',
             '## Accuracy', '', '| Path | Strict score tolerance | Application gate | Max absolute error | Changed full lists / top-1 / top-10 membership / order |',
             '|---|---|---|---:|---|']
    for path, result in accuracy['accuracy'].items():
        cells = result['cells']
        counts = [sum(c['changes'][field] for c in cells) for field in ('full_ranking', 'top1', 'top10_membership', 'top10_order')]
        lines.append(f"| {path} | {'pass' if result['strict_numerical_equivalence'] else 'fail'} | {'pass' if result['accuracy_qualified'] else 'FAIL — diagnostic'} | {max(c['error']['max'] for c in cells):.7g} | {' / '.join(map(str, counts))} |")
    lines += ['', 'Changes count 1,455 dataset/K lists, with the same queries repeated across K. Full error distributions (median/P95/P99, stabilized relative errors), per-query score/ranking artifacts, per-cell nDCG/MRR deltas and query bootstrap intervals are retained in `cuda_accuracy.json` and `cuda_scores.jsonl.gz`. Near-zero relative errors use a 0.001 denominator floor.', '',
              'Fused stock and compiled paths execute four fused encoder calls. Both fail the unchanged 1% top-10 membership gate at K=100: SciFact 3/162 (1.852%) and NFCorpus 5/323 (1.548%). Max score error remains below 0.01, top-1 fractions remain below 1%, and nDCG/MRR loss intervals pass their separate gates. Passing those checks does not override the membership failure. Their timing comparisons are diagnostic only; no tolerance was widened.', '',
              'Custom CUDA and Triton pass strict numerical and application gates, but numerical tolerance does not guarantee identical ordering of nearly tied scores. Ranking changes are retained above and in the per-query audit.', '',
              '## Warm timing', '',
              '| Scope | K | Strict pairs/s | Fused stock pairs/s (diagnostic) | CUDA pairs/s | Triton pairs/s | Compiled pairs/s (diagnostic) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('forward', 'direct'):
        for k in plan['document_counts']:
            values = [np.median([r['pairs_per_second'] for r in runs if (r['scope'], r['documents'], r['path']) == (scope, k, p)]) for p in ('strict', 'stock', 'cuda', 'triton', 'compiled')]
            lines.append(f"| {scope} | {k} | " + ' | '.join(f'{v:.0f}' for v in values) + ' |')
    lines += ['', '| Scope | K | Custom path / strict throughput | Session bootstrap 95% CI | Interpretation |',
              '|---|---:|---|---|---|']
    for c in comparisons:
        if c['path'] not in ('cuda', 'triton'):
            continue
        lo, hi = c['ratio_ci95']
        meaning = {'left advantage supported': 'strict faster', 'right advantage supported': 'custom faster', 'inconclusive': 'inconclusive'}[c['interpretation']]
        lines.append(f"| {c['scope']} | {c['documents']} | {c['path']}: {c['ratio_geomean']:.3f} | [{lo:.3f}, {hi:.3f}] | {meaning} |")
    lines += ['', 'Ratios above 1 favor the named custom path. Intervals bootstrap five session means, not individual batches. They are exploratory, coarse, unadjusted for multiple comparisons and conditional on this workload/system. Raw ranges/IQR and all repetition observations are retained. The original four-session contract acquired a fifth path after compile succeeded. Before inspecting timing results, `nvidia_balance_plan.json` declared an additional fresh session using the unchanged frozen measurement function, restoring complete startup-position balance. Original four-session results remain in `cuda_summary.json`; the balanced 15-repetition comparisons are in `cuda_supplement.json`. The fifth process ran in a separate dependent allocation (job 172871), so allocation/time drift remains a limitation.', '',
              'These measurements do not establish a custom-kernel win over an accuracy-qualified fused stock configuration: the fused paths failed the contract. Diagnostic timings against them remain in `cuda_summary.json`. The strict baseline already executes CUDA efficient SDPA, as verified by profiler; disabling encoder fusion does not imply unoptimized attention.', '',
              'Prepared forward excludes tokenization/transfers/CPU scores. Direct scoring includes those costs and flattens the 16 fixtures into microbatches; it excludes HTTP, sorting and startup. No batch percentile is called a request percentile. Compilation/first use is recorded separately in `cuda_compile.json` with unknown disk-cache state; it is not a true cold compilation measurement.', '',
              '## Verification and limitations', '',
              'The remote asset guard matched every frozen checkpoint/tokenizer/input/source hash and the pinned custom CUDA binary. Saved-record verification independently checks candidate identities, uniqueness/completeness, finite scores, tie-breaking, full qrels, metrics (also TREC parity), expected pair counts, timing totals and session uncertainty. The separate post-timing dispatch probe records actual GPU kernel names and eight custom operation calls with fallback=False for each custom path. GPU tests run only after the benchmark, in dependent job 172870.', '',
              'All three jobs completed with exit code 0:0; 136 GPU tests passed without skips (18.78 s). PyTorch 2.14.1+cu130 and driver 610.43.02 were recorded. Before/after snapshots show 35–39°C and idle-start SM clocks of 210–2520 MHz; every after snapshot shows 2520 MHz. They sample boundaries, not continuous GPU activity, and cannot rule out clock changes or throttling during timings. No continuous hardware isolation was verified. First-use compile probe took 46.05 s and recorded three generated kernels, with unknown disk-cache state. Slurm allocated one L40S; no other person’s session/job was altered. This small model has poor general retrieval quality, and development timing does not establish production saturation behavior or generalize to other hardware/models.', '',
              'Reproduce: copy the exact plan inputs and E17 harness, then submit `sbatch experiments/E17/nvidia.sbatch` from the remote project. For the fifth rotation use `sbatch --dependency=afterok:VALIDATION_JOB experiments/E17/nvidia_balance.sbatch`. For post-timing checks submit `sbatch --dependency=afterok:JOB_ID experiments/E17/nvidia_validation.sbatch`. After retrieving raw results, run `.venv/bin/python -m experiments.E17.nvidia_report`; this audits saved artifacts and generates this report/plot without GPU execution. `cuda_checksums.json` pins all CUDA artifacts, including the ignored large score audit.', '',
              '![NVIDIA throughput and repetition variation](cuda_throughput.png)', '']
    (HERE/'NVIDIA_REPORT.md').write_text('\n'.join(lines))
    refresh_completed_status()
    artifacts = {*HERE.glob('cuda_*'), *HERE.glob('nvidia*.log'), HERE/'NVIDIA_REPORT.md',
                 HERE/'nvidia_provenance.json', HERE/'nvidia_balance_plan.json'}
    save(HERE/'cuda_checksums.json', {p.name: sha(p) for p in sorted(artifacts) if p.name != 'cuda_checksums.json'})
    verification = json.loads((HERE/'verification.json').read_text())
    verification['nvidia'] = {'status': 'verified separately from Apple', 'report': 'NVIDIA_REPORT.md',
                             'accuracy_pairs_per_path': 77600, 'timed_runs': len(runs),
                             'timed_pair_executions': total_pairs, 'sessions': 5, 'paired_repetitions': 15,
                             'GPU_tests_passed': 136, 'cuda_checksums_sha256': sha(HERE/'cuda_checksums.json')}
    results = [*HERE.glob('*.json'), *HERE.glob('*scores.jsonl.gz'), *HERE.glob('*.png'),
               *HERE.glob('nvidia*.log'), HERE/'REPORT.md', HERE/'NVIDIA_REPORT.md']
    save(HERE/'checksums.json', {p.name: sha(p) for p in sorted(results) if p.name not in ('checksums.json', 'verification.json')})
    verification['checksums_sha256'] = sha(HERE/'checksums.json')
    save(HERE/'verification.json', verification)


if __name__ == '__main__':
    main()
