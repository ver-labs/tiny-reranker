"""Generate tables from verified observations, with failed paths visible."""
import ast
import json
from .common import HERE


def ratio(r):
    return f"{r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if 'ratio' in r else r.get('status','PENDING')


def main():
    summary=json.loads((HERE/'summary.json').read_text());lines=['# E18 measured tables','',
      'These tables describe E18-small-v1 with its batch-control addendum. FP32, L≤512, raw logits. Ratios above one favor the candidate. Confidence intervals are unadjusted exploratory session-bootstrap intervals; no confirmatory significance family is claimed. Accuracy-failing paths remain diagnostic. Missing observations are pending, not zero.','',
      '## Accuracy','', '| Device/path | Status | Strict numerical | Application | Max / P99 absolute score error | Largest top-1 / top-10 membership changes per cell |','|---|---|---|---|---|---|']
    for name,r in sorted(summary['accuracy'].items()):
        cells=r.get('cells',{});max_top1=max((v['top1_changed']/v['queries'] for v in cells.values()),default=0);max_top10=max((v['top10_membership_changed']/v['queries'] for v in cells.values()),default=0)
        errors=r.get('absolute_error');error=f"{errors['max']:.3g} / {errors['p99']:.3g}" if errors else '—'
        lines.append(f"| {name} | {r['status']} | {r.get('strict_numerical_pass','—')} | {r.get('application_pass','—')} | {error} | {max_top1:.2%} / {max_top10:.2%} |")
    lines+=['','Full error distributions, per-query rankings and nDCG/MRR differences with paired query intervals are in each path’s accuracy JSON and checksummed score artifact. Stock is the practical SDPA baseline; its self-comparison is not an independent full CPU audit. The CPU diagnostic uses four padded pairs.','',
      '## Primary complete direct scoring','', '| Device | Path | Accuracy | Pairs/s median [IQR] | Sessions / repetitions | Stock / candidate time ratio [95% CI] |','|---|---|---|---|---|---|']
    for key,cell in sorted(summary['direct'].items()):
        device,shape=key.split('/',1)
        if ast.literal_eval(shape)!=('scoring',100,16,512,'primary'):continue
        for path,d in cell['distribution'].items():
            v=d['pairs_per_second'];r=cell['comparisons'].get('stock/'+path,{})
            lines.append(f"| {device} | {path} | {d['accuracy']} | {v['median']:.1f} [{v['p25']:.1f}, {v['p75']:.1f}] | {d['sessions']} / {d['repetitions']} | {'Baseline' if path=='stock' else ratio(r)} |")
    lines+=['','## Local loopback HTTP','', '| K | C | Coordinator/worker | P50 / P95 / P99 ms (median across repetitions) | req/s [IQR] | pairs/s | Successful requests / failures | Python / Rust time ratio [95% CI] |','|---|---|---|---|---|---|---|---|']
    for key,cell in sorted(summary['http'].items(),key=lambda e:tuple(int(x[1:]) for x in e[0].split('/'))):
        k,c=(int(x[1:]) for x in key.split('/'))
        for path,d in sorted(cell['distribution'].items()):
            v=d['requests_per_second'];backend=path.split('/')[1];r=cell['comparisons'].get('python/'+backend+'/rust/'+backend,{})
            lines.append(f"| {k} | {c} | {path} | {d['p50_ms']['median']:.1f} / {d['p95_ms']['median']:.1f} / {d['p99_ms']['median']:.1f} | {v['median']:.3f} [{v['p25']:.3f}, {v['p75']:.3f}] | {d['pairs_per_second']['median']:.1f} | {d['requests']} / {d['failures']} | {ratio(r) if path.startswith('rust') else '—'} |")
    lines+=['','The 16-request repetition gives only two waves at C8. P99 is a near-maximum of a short wave, not a production tail-latency estimate. The load is fixed-concurrency closed-loop; it does not test open-loop overload. Rust coordinates a shared Python inference worker.']
    if summary['optimized_http']:
        lines+=['','## Follow-up optimized HTTP','',
          'Each family uses newly started matched baseline workers. These are conditional follow-ups selected from direct scoring; compare within this family, rather than with the earlier coordinator study.','',
          '| Study / workload | Worker (Python coordinator) | P50 / P95 / P99 ms | Requests/s | Matched baseline / candidate time [95% CI] |',
          '|---|---|---|---|---|']
        for key,cell in sorted(summary['optimized_http'].items()):
            for path,d in sorted(cell['distribution'].items()):
                lines.append(f"| {key} | {path} | {d['p50_ms']['median']:.1f} / {d['p95_ms']['median']:.1f} / {d['p99_ms']['median']:.1f} | {d['requests_per_second']['median']:.3f} | {ratio(cell['comparisons'][path]) if path in cell['comparisons'] else 'Baseline'} |")
    lines+=['','## Resident operation','', '| Device / rows × 384 | Path | Median µs [IQR] | Native / candidate time ratio [95% CI] |','|---|---|---|---|']
    for key,cell in sorted(summary['operation'].items()):
        for path,d in cell['distribution'].items():lines.append(f"| {key} | {path} | {d['median']:.1f} [{d['p25']:.1f}, {d['p75']:.1f}] | {'Baseline' if path=='native' else ratio(cell['comparisons'].get('native/'+path,{}))} |")
    lines+=['','Apple operation measurements include dispatch and synchronized amortized wall time. CUDA operation measurements use device events over 100 launches and include GPU-visible launch gaps. These scopes differ and must not be interpreted as cross-hardware kernel speedups.','',
      '## Complete secondary direct matrix','', '| Device | Scope | K / B / L / group | Path | Pairs/s median [IQR] | Accuracy |','|---|---|---|---|---|---|']
    for key,cell in sorted(summary['direct'].items()):
        device,shape=key.split('/',1);scope,k,b,l,group=ast.literal_eval(shape)
        for path,d in sorted(cell['distribution'].items()):
            v=d['pairs_per_second'];lines.append(f"| {device} | {scope} | {k} / {b} / {l} / {group} | {path} | {v['median']:.1f} [{v['p25']:.1f}, {v['p75']:.1f}] | {d['accuracy']} |")
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('Tables generated')

if __name__=='__main__':main()
