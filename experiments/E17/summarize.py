"""Session-level paired uncertainty; raw request counts never used as run count."""
import json
import math
import numpy as np
from .common import HERE, save, sha, verify_plan, describe, interval


def paired(left,right,field,higher=True):
    def index(runs):
        out={}
        for r in runs:
            k=(r['session'],r['repeat'])
            if k in out:raise ValueError('Duplicate paired repetition')
            out[k]=r[field]
        return out
    a,b=index(left),index(right)
    if set(a)!=set(b) or len(a)<10:raise ValueError('Incomplete paired repetitions')
    logs={s:[] for s,r in a}
    differences={s:[] for s,r in a}
    for k in sorted(a):
        if a[k]<=0 or b[k]<=0:raise ValueError('Invalid timing')
        logs[k[0]].append(math.log(b[k]/a[k] if higher else a[k]/b[k]))
        differences[k[0]].append(b[k]-a[k])
    cluster=[float(np.mean(v)) for v in logs.values()]
    ci=interval(cluster)
    return {'ratio_geomean':math.exp(ci['mean']),'ratio_ci95':[math.exp(v) for v in ci['ci95']],
            'difference_right_minus_left':interval([float(np.mean(v)) for v in differences.values()]),
            'paired_repetitions':len(a),'independent_sessions':len(cluster),
            'session_log_ratios':cluster,
            'interpretation':'right advantage supported' if ci['ci95'][0]>0 else 'left advantage supported' if ci['ci95'][1]<0 else 'inconclusive'}


def main():
    plan=verify_plan();http=[];direct=[]
    for file in sorted(HERE.glob('http_s[0-3]_*.json')):
        data=json.loads(file.read_text());assert data['status']=='complete'
        http.extend({**r,'session':data['session'],'coordinator':data['coordinator'],'backend':data['backend']} for r in data['runs'])
    for file in sorted(HERE.glob('mps_direct_s*.json')):
        data=json.loads(file.read_text());assert data['status']=='complete'
        direct.extend({**r,'session':data['session']} for r in data['runs'])
    summary={'plan_sha256':sha(HERE/'plan.json'),'http':[],'direct':[],
             'statistical_unit':'Four independent process sessions, three paired repetitions each; session-bootstrap exploratory CI.'}
    for backend in ('pytorch','mlx'):
        for k in plan['document_counts']:
            for c in plan['concurrency']:
                subset=[r for r in http if r['backend']==backend and r['documents']==k and r['concurrency']==c]
                grouped={co:[r for r in subset if r['coordinator']==co] for co in ('python','rust')}
                assert all(len(v)==12 for v in grouped.values())
                fields=('p50_ms','p95_ms','p99_ms','requests_per_second','pairs_per_second','queue_p50_ms','processing_p50_ms')
                row={'backend':backend,'documents':k,'concurrency':c,
                     'coordinators':{co:{f:describe([r[f] for r in v]) for f in fields} for co,v in grouped.items()},
                     'paired_rust_vs_python':{f:paired(grouped['python'],grouped['rust'],f,higher='second' in f) for f in ('p50_ms','p95_ms','p99_ms','requests_per_second')}}
                row['successful_requests']=sum(r['successful_requests'] for r in subset)
                row['failures']=sum(r['failures'] for r in subset)
                summary['http'].append(row)
    for scope in ('forward','direct'):
        for k in plan['document_counts']:
            grouped={path:[r for r in direct if r['path']==path and r['scope']==scope and r['documents']==k] for path in sorted({r['path'] for r in direct})}
            assert all(len(v)==12 for v in grouped.values())
            summary['direct'].append({'scope':scope,'documents':k,
                'paths':{path:describe([r['pairs_per_second'] for r in v]) for path,v in grouped.items()},
                'paired_stock_vs_strict':paired(grouped['strict'],grouped['stock'],'pairs_per_second'),
                'paired_compiled_vs_stock':paired(grouped['stock'],grouped['compiled'],'pairs_per_second') if 'compiled' in grouped else None})
    summary['total_http_requests']=sum(r['successful_requests'] for r in http)
    summary['total_http_pairs']=sum(r['successful_pairs'] for r in http)
    summary['failures']=sum(r['failures'] for r in http)
    compact=[{k:v for k,v in r.items() if k!='observations'} for r in http]
    save(HERE/'http_timing_raw.json',{'plan_sha256':sha(HERE/'plan.json'),'runs':compact})
    save(HERE/'summary.json',summary)
    plot(summary)
    report(summary)


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
    colors={'pytorch':'#285d95','mlx':'#bc6333'}
    for i,(metric,label) in enumerate((('p50_ms','Request P50 (ms)'),('p95_ms','Request P95 (ms)'),('requests_per_second','Requests / second'))):
        for j,c in enumerate((1,4,8)):
            ax=axes[i,j]
            for backend in ('pytorch','mlx'):
                for co in ('python','rust'):
                    points=sorted((r for r in summary['http'] if r['backend']==backend and r['concurrency']==c),key=lambda r:r['documents'])
                    values=[r['coordinators'][co][metric] for r in points]
                    med=np.asarray([v['median'] for v in values]);iqr=np.asarray([v['iqr'] for v in values])
                    # IQR bars are distribution spread across repeats, not CIs.
                    ax.errorbar([r['documents'] for r in points],med,yerr=[med-iqr[:,0],iqr[:,1]-med],
                                color=colors[backend],linestyle='-' if co=='python' else '--',marker='o' if co=='python' else 's',label=f'{co} + {backend}',capsize=3)
            ax.set_title(f'Concurrency {c}');ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xticks([10,50,100])
            if i==2:ax.set_xlabel('Documents per request (microbatch 32)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,0))
    fig.suptitle('Apple M5 · local loopback · closed-loop · median and repetition IQR\n12 repeats across 4 process sessions; 16 fixed requests per wave',fontsize=13)
    fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(HERE/'http_workloads.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7))
    points=summary['http'];y=np.arange(len(points))
    v=[p['paired_rust_vs_python']['requests_per_second'] for p in points]
    med=np.asarray([p['ratio_geomean'] for p in v]);ci=np.asarray([p['ratio_ci95'] for p in v])
    ax.errorbar(med,y,xerr=[med-ci[:,0],ci[:,1]-med],fmt='o',capsize=3,color='#285d95')
    ax.axvline(1,color='black',linestyle='--');ax.set_yticks(y,[f"{p['backend']} K={p['documents']} C={p['concurrency']}" for p in points]);ax.invert_yaxis()
    ax.set_xlabel('Rust / Python request throughput (paired geometric ratio)')
    ax.set_title('Exploratory 95% CI from resampling four session means\nWithin-session repeats averaged; no multiplicity correction')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(HERE/'coordinator_uncertainty.png',dpi=160);plt.close(fig)


def report(summary):
    accuracy=json.loads((HERE/'mps_accuracy.json').read_text())
    lines=['# E17: realistic inference and serving benchmarks','',
           'The original epoch-3 checkpoint, tokenizer, FP32 weights and 128-token limit are fixed. This evaluates implementation accuracy and inference performance; the scratch model remains a poor general reranker. No retraining or test-set tuning was performed.','',
           '## Contract and accuracy','',
           'The initial contract is retained in `plan_preflight.json`. Compile compatibility revealed an aggregation assumption before repeated performance collection; only code support for the additional compiled path was corrected, and the plan was re-frozen. Workloads, selection, tolerances and statistical methods were unchanged. Preflight incomplete scores were discarded; all reported audits and timings use the final frozen code.','',
           'The immutable `plan.json` was saved before measurements. Development inputs: 162 SciFact and 323 NFCorpus queries, each with fixed BM25 top-100 documents and complete qrels. Accuracy covers all 485 queries separately at K=10/50/100 (77,600 pairs per path); HTTP uses 16 preselected queries, eight per dataset. No candidate duplication or positive injection. The shared worker sorts input documents by ID before microbatch scoring; output ties use ascending document ID.','',
           '| Apple path | Strict numerical tolerance | Application gate | Max absolute error | Full ranking changes (across all K) |',
           '|---|---|---|---:|---:|']
    for path,result in accuracy['accuracy'].items():
        lines.append(f"| {path} | {'pass' if result['strict_numerical_equivalence'] else 'fail'} | {'pass' if result['accuracy_qualified'] else 'DIAGNOSTIC ONLY'} | {max(c['error']['max'] for c in result['cells']):.8g} | {sum(c['changes']['full_ranking'] for c in result['cells'])} |")
    lines+=['',
        'Strict tolerance was predeclared as atol=1e-5, rtol=1e-4. Application acceptance requires every dataset/K cell to have max absolute score error ≤0.01, ≤1% top-1 and top-10 membership changes, and paired-query-bootstrap lower 95% bounds for nDCG@10/MRR@10 differences ≥−0.001. Full ranking/order changes, stabilized relative errors, per-query metrics and all scores are retained even when a path passes. The relative denominator is max(|reference|, 0.001); near-zero counts are disclosed. `mps_accuracy.json` contains all six cells per path and paired intervals. Passing these checks establishes agreement with this weak model, not good retrieval quality.','',
        'Profiler evidence: on this MPS installation both strict and stock settings execute `aten::_scaled_dot_product_attention_math_for_mps`; no fused stock encoder is observed. Therefore the enabled flag is not evidence of an Apple optimization. The two names disclose the requested settings; this is not a successful comparison of distinct fused MPS paths. CUDA fused-stock dispatch remains unmeasured.','']
    probe=json.loads((HERE/'mps_compile.json').read_text())
    if accuracy['paths'].get('compiled',{}).get('diagnostic_only'):
        lines += [f"Compiled full-workload audit: **failed** after {accuracy['paths']['compiled']['completed_records']} records. The single-shape probe worked, but this heterogeneous workload did not complete. No compiled winner is claimed; partial scores and exact exception are retained separately.",'']
    lines += [f"`torch.compile` Inductor/fullgraph compatibility probe: **{probe['status']}**. See `mps_compile.json` for exact diagnostics. First-use wall time is separate and is not called cold compilation because disk cache state is unknown.",'',
              '## Local loopback HTTP results','',
              '| Backend | Documents | Concurrency | Python P50 / P95 ms | Rust P50 / P95 ms | Python / Rust req/s | Rust/Python throughput ratio [95% session CI] |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['http']:
        p,s=r['coordinators']['python'],r['coordinators']['rust'];v=r['paired_rust_vs_python']['requests_per_second']
        lines.append(f"| {r['backend']} | {r['documents']} | {r['concurrency']} | {p['p50_ms']['median']:.2f} / {p['p95_ms']['median']:.2f} | {s['p50_ms']['median']:.2f} / {s['p95_ms']['median']:.2f} | {p['requests_per_second']['median']:.2f} / {s['requests_per_second']['median']:.2f} | {v['ratio_geomean']:.3f} [{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}] |")
    supported=[f"{r['backend']} K={r['documents']} C={r['concurrency']} ({r['paired_rust_vs_python']['requests_per_second']['interpretation']})" for r in summary['http'] if r['paired_rust_vs_python']['requests_per_second']['interpretation']!='inconclusive']
    lines+=['',f"Completed {summary['total_http_requests']:,} measured requests / {summary['total_http_pairs']:,} pairs; failures: {summary['failures']}. Every measured response is checked against its same-backend direct audit for scores and exact ranking. Raw observations retain returned IDs/scores, client latency, queue wait and processing time. P99, pair throughput, ranges/IQR and latency-difference intervals are in `summary.json`.",'',
        ('Exploratory throughput intervals excluding unity: '+ '; '.join(supported)+'.') if supported else 'All coordinator throughput intervals include unity; the measurements do not support a Rust/Python throughput winner.',
        'These intervals are conditional on four sessions on one machine and are not corrected for 18 comparisons. They cannot establish a universal coordinator advantage. Cross-backend differences include different framework execution, not only coordinator language.','',
        '## Timing and uncertainty','',
        'Twelve paired repeats per cell are spread over four independently started sessions. Startup ordering is balanced by a seeded Latin rotation. Each process warms with six realistic requests after the worker warmup. One model worker, bounded queue 32, microbatch 32, and no cross-request batching apply to every coordinator. Timeouts are 30 s request / 60 s worker. The exact E12 worker and its validation, tokenization, sorting, IPC and response contents are reused. Document count, pair microbatch and concurrent requests are independent dimensions.','',
        'Load is fixed-concurrency closed-loop on local loopback. Per-request latency ends at receipt of the HTTP response. Wave throughput includes client validation/scheduling overhead. Sixteen requests per wave give limited saturation and noisy P99 estimates; closed-loop load does not characterize open-loop overload or internet latency. Error bars in `http_workloads.png` show repetition IQR, not a confidence interval. `coordinator_uncertainty.png` shows paired throughput ratios and bootstrap intervals from four session means. Thousands of requests are never treated as thousands of independent benchmark runs.','',
        'Direct scoring separately flattens the selected queries and scores pair batches of 32, so boundaries may span queries; this is an offline scoring workload, not request batching in the server. `mps_direct_s*.json` separates prepared device-resident model forwards (synchronized, excludes CPU score conversion) from tokenization/transfer/model/CPU-score timing. Neither is labeled request latency. Historical E11/E14 isolated-operation results are not reused to claim new whole-model gains.','',
        '| Direct scope | Candidate count | Strict pairs/s | Stock flag pairs/s | MLX pairs/s | Stock/strict ratio [95% session CI] |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['direct']:
        paths=r['paths'];v=r['paired_stock_vs_strict']
        lines.append(f"| {r['scope']} | {r['documents']} | {paths['strict']['median']:.1f} | {paths['stock']['median']:.1f} | {paths['mlx']['median']:.1f} | {v['ratio_geomean']:.3f} [{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}] |")
    if any(r['paired_compiled_vs_stock'] for r in summary['direct']):
        lines += ['', '| Compiled scope | Candidate count | Compiled pairs/s | Compiled/stock ratio [95% session CI] |', '|---|---:|---:|---:|']
        for r in summary['direct']:
            v=r['paired_compiled_vs_stock']
            if v:
                lines.append(f"| {r['scope']} | {r['documents']} | {r['paths']['compiled']['median']:.1f} | {v['ratio_geomean']:.3f} [{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}] |")
        lines += ['', 'Compiled measurements are direct inference only; the matched HTTP worker continues to use eager PyTorch. Accuracy qualification is reported above. A failing compiled path remains diagnostic, even if warm timing is lower. All workload shapes are materialized before timing; no compilation time is included in repeated forward/direct timings.']
    lines+=['',
        'System snapshots record model/software versions, power source, thermal status and visible processes before/after sessions. Coordinator and worker RSS are sampled separately before/after; these are not peak memory and unified GPU memory may overlap RSS. Apple GPU clocks, temperature and utilization are not available from these snapshots. No exclusive hardware isolation is claimed. Startup/readiness times live in each raw HTTP file and include model loading/worker warmup; they are excluded from steady-state timing.','',
        '## NVIDIA status and defensible conclusions','',
        'NVIDIA execution is pending: a single non-interactive SSH check failed authentication; the user requested continuing on Apple. A subsequent user-run interactive attempt was also rejected. No more authentication attempts or Slurm jobs were launched. The reproducible CUDA harness retains strict PyTorch, optimized-stock, custom CUDA and custom Triton names, FP32/TF32-disabled settings, fallback=False and four Triton warps. Its GPU execution/accuracy cannot be verified on Apple.','',
        'Whether custom CUDA/Triton beat accuracy-qualified optimized stock PyTorch remains **unmeasured**. The old L40S full-test median differences (~1%) had overlapping ranges and disabled stock attention fusion; they do not establish a reliable custom-kernel advantage. This experiment supplies no replacement NVIDIA numbers. Apple M5 and NVIDIA L40S results cannot be interpreted as language-level speedups.','',
        '## Reproduction and artifacts','',
        'Run from the repository root with `.venv/bin/python -m experiments.E17.launch --device mps`. The frozen plan prevents unnoticed input/source changes. Existing complete outputs are skipped only after integrity checks; choose a new experiment for changes to the contract. Individual commands and NVIDIA launch guidance are in `launch.py` and `nvidia.sbatch`. Large gzip score/request audits remain local outside Git; `checksums.json` identifies every result. `verification.json` independently recomputes ranking, quality, request counts, percentiles, throughput and paired summaries.','',
        '![HTTP workloads](http_workloads.png)','', '![Coordinator uncertainty](coordinator_uncertainty.png)','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
