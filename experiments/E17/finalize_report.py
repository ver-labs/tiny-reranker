"""Reproducible post-collection interpretation and independent source audit.

Run after launch: .venv/bin/python -m experiments.E17.finalize_report
This adds no primary timing observations and changes no frozen tolerances.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .common import ROOT, HERE, rows, sha, save, verify_plan, describe


def main():
    plan=verify_plan();data=rows();old=json.loads((ROOT/'experiments/E16/data_plan.json').read_text())
    dev_groups={r['query_group'] for r in data};train_ids=set();test_ids=set();train_pairs=0
    for relative in ('data/processed/clean_pairs.jsonl','data/processed/conflicting_pairs.jsonl'):
        assert sha(ROOT/relative)==old['source_sha256'][relative]
        for line in (ROOT/relative).open():
            r=json.loads(line)
            normalized=hashlib.sha256(' '.join(r['query'].casefold().split()).encode()).hexdigest()
            assert normalized==r['query_id']
            if r['split']=='train' and relative.endswith('clean_pairs.jsonl'):
                train_ids.add(normalized);train_pairs+=1
            elif r['split']=='test':test_ids.add(normalized)
    assert train_pairs==735276
    assert not dev_groups&train_ids and not dev_groups&test_ids
    for dataset in ('scifact','nfcorpus'):
        folder=ROOT/'data/beir'/dataset
        corpus={r['_id']:(r.get('title','')+' '+r['text']).strip() for r in map(json.loads,(folder/'corpus.jsonl').open())}
        queries={r['_id']:r['text'] for r in map(json.loads,(folder/'queries.jsonl').open())}
        qrels={}
        split='train' if dataset=='scifact' else 'dev'
        for r in csv.DictReader((folder/f'qrels/{split}.tsv').open(),delimiter='\t'):
            qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
        for row in (r for r in data if r['dataset']==dataset):
            assert queries[row['query_id']]==row['query'] and qrels[row['query_id']]==row['qrels']
            assert [corpus[d] for d in row['document_ids']]==row['texts']
    preparation={'status':'verified','source_sha256':sha(__file__),'original_training_pairs':train_pairs,
                 'original_training_queries':len(train_ids),'development_queries':len(data),'development_top100_pairs':48500,
                 'original_train_development_normalized_overlaps':0,'original_test_development_normalized_overlaps':0,
                 'official_query_text_candidate_text_and_full_qrels':'Every record checked against source files',
                 'candidate_policy':'Frozen E16 preparation/file hashes; BM25 was not rebuilt or retuned.'}
    save(HERE/'input_verification.json',preparation)
    summary=json.loads((HERE/'summary.json').read_text())
    from .summarize import report as render_base_report
    render_base_report(summary)
    report=(HERE/'REPORT.md').read_text()
    findings='''## Measured conclusions

- No Rust throughput advantage is supported. Seventeen of eighteen coordinator throughput intervals include unity. One exploratory point (PyTorch, 10 documents, concurrency 4) favors Python: Rust/Python ratio 0.880 [0.779, 0.993]. Four sessions and uncorrected multiple comparisons make this a narrow local observation.
- Compiled PyTorch passed accuracy, but its complete-scoring throughput was lower than eager stock for the 50-document fixture: ratio 0.900 [0.811, 0.978], and 100-document fixture: 0.859 [0.837, 0.881]. All prepared-forward ratios and the 10-document complete-scoring comparison are inconclusive. These are offline microbatch-32 measurements, not compiled HTTP measurements.
- Compiled PyTorch's three changed full rankings have zero top-1, top-10 membership/order, nDCG@10 or MRR@10 changes. Numerical tolerances do not require bitwise-identical full rankings.
- Custom CUDA/Triton versus optimized stock on NVIDIA remains pending authentication. Historical overlapping ~1% differences cannot fill this gap.

'''
    report=report.replace('## Contract and accuracy',findings+'## Contract and accuracy',1)
    diagnostic=json.loads((HERE/'flag_diagnostic.json').read_text())
    assert diagnostic['source_sha256']==sha(HERE/'flag_diagnostic.py') and diagnostic['summary']['exact_score_parity']
    pairs={p:{r['repeat']:r['seconds'] for r in diagnostic['observations'] if r['path']==p} for p in ('strict','stock')}
    assert set(pairs['strict'])==set(pairs['stock'])==set(range(24))
    computed=float(np.exp(np.mean([math.log(pairs['strict'][i]/pairs['stock'][i]) for i in range(24)])))
    assert math.isclose(computed,diagnostic['summary']['stock_over_strict_ratio'],rel_tol=1e-12)
    text='''### Eager flag diagnostic and residual confounding

The strict/stock setting comparison at the 10-document offline fixture reported a stock/strict ratio 1.369 [1.132, 1.741], despite identical observed MPS attention dispatch and exact scores. It is not evidence that the fused-encoder fast path executed. To investigate the discrepancy, a separately labeled post-hoc diagnostic reused the same hot model, tokenizer and prepared input allocation for 24 balanced-order repetitions. It instead reported ratio 0.932 [0.864, 0.991], with exact score parity. That interval describes within-process variation only. The diagnostic does not replace the original observations or add independent sessions to their CI.

The advantage reverses between these setups. Residual process/model state, unobserved execution details and system variation remain possible causes; no cause is established. The eager flag gap is therefore not attributed to a named optimization. The session intervals are conditional sampling estimates and cannot account for all systematic confounding. See `flag_diagnostic.py` and `flag_diagnostic.json`.

'''
    report=report.replace('## NVIDIA status and defensible conclusions',text+'## NVIDIA status and defensible conclusions',1)
    observations=[];startup={};rss={};session_means={}
    for file in sorted(HERE.glob('http_s[0-3]_*.json')):
        r=json.loads(file.read_text());group=f"{r['coordinator']}/{r['backend']}"
        startup.setdefault(group,[]).append(r['ready_seconds'])
        rss.setdefault(group,[]).extend(v for sample in (r['rss_before'],r['rss_after']) for v in sample.values() if v is not None)
        observations.extend({**o,'backend':r['backend'],'coordinator':r['coordinator'],'session':r['session'],
                             'documents':run['documents'],'concurrency':run['concurrency'],'repeat':run['repeat']}
                            for run in r['runs'] for o in run['observations'] if o['success'])
        session_means.setdefault(group,{})[str(r['session'])]=float(np.median([run['requests_per_second'] for run in r['runs'] if run['documents']==100 and run['concurrency']==4]))
    maximum=max(observations,key=lambda r:r['latency_ms'])
    conditions={'status':'inspected','source_sha256':sha(__file__),
                'startup_ready_seconds':{g:describe(v) for g,v in startup.items()},
                'combined_individual_RSS_samples_bytes':{g:describe(v) for g,v in rss.items()},
                'RSS_scope':'Individual coordinator/worker samples combined for range only; not summed peak memory.',
                'K100_C4_session_median_requests_per_second':session_means,
                'largest_client_latency':{k:maximum[k] for k in ('latency_ms','queue_ms','processing_ms','backend','coordinator','session','documents','concurrency','repeat','request_id')},
                'outlier_policy':'Retained all observations, no filtering',
                'power':'All sampled sessions AC power; no recorded thermal/performance warnings. No GPU clocks/temperature/utilization or exclusive isolation.'}
    save(HERE/'conditions_analysis.json',conditions)
    report=report.replace('## NVIDIA status and defensible conclusions',
        f"All sampled sessions ran on AC power without recorded thermal/performance warnings. The largest measured HTTP latency was {maximum['latency_ms']:.1f} ms ({maximum['coordinator']}/{maximum['backend']}, K={maximum['documents']}, C={maximum['concurrency']}); its queue wait was {maximum['queue_ms']:.1f} ms and processing {maximum['processing_ms']:.1f} ms. It remains in the raw data. `conditions_analysis.json` retains startup ranges, RSS samples and per-session throughput for drift inspection. A specific thermal or competing-GPU cause cannot be established.\n\n## NVIDIA status and defensible conclusions",1)
    report += '''
Additional independent source audit checked all original cleaned/conflicting pairs for normalized query hashes, found zero original-training/test overlap with development, and checked every development query, candidate text and full qrel dictionary against official source files (`input_verification.json`). Core benchmark tests: 8 passed; supplemental NVIDIA provenance guards: 3 passed; reused HTTP/metrics: 19 passed; local CUDA/Triton guards: 15 passed, 121 GPU-dependent tests skipped. Both plots were visually inspected: correct ms/req/s units, unclipped labels, IQR versus CI distinguished, and no cross-hardware speedup claims (`plot_review.json`).

For the full report including this post-collection interpretation, run `python -m experiments.E17.finalize_report` after `launch`, then `python -m experiments.E17.verify` to refresh all result checksums. For only saved-record verification, run `verify`; no GPU is required. Large per-query gzip scores and full per-request JSON audits remain ignored locally; compact per-repetition timing observations are committed in `http_timing_raw.json` and `mps_direct_s*.json`.

The existing CUDA source and shared-library bytes match their E14 hashes locally (`nvidia_provenance.json`); this is an asset guard, not GPU execution or accuracy validation. `nvidia_provenance_plan.json` supplements the unchanged Apple contract, and `nvidia.sbatch` checks its hashes and L40S access before execution.

Once NVIDIA authentication is restored, copy the frozen E17 code/plan and all files named by the plan (including original checkpoint/tokenizer/development inputs) into `/nethome/sbagathi3/tiny-reranker-e13-20261010-Myeovg`; retain the existing E14 `.so`. Run `/opt/slurm/current/bin/sbatch experiments/E17/nvidia.sbatch` from that project. The harness checks every frozen hash and refuses CPU substitution. This launch has not been executed; do not overwrite the plan to accommodate a mismatch. CUDA results are summarized and audited separately by `nvidia_summary.py`.
'''
    (HERE/'REPORT.md').write_text(report)
    hardware=json.loads((HERE/'hardware_status.json').read_text());hardware['status']='Apple benchmarks complete and audited; NVIDIA pending authentication'
    save(HERE/'hardware_status.json',hardware)
    if (HERE/'cuda_supplement.json').exists():
        from .nvidia_report import refresh_completed_status
        refresh_completed_status()
    print('Independent input audit and post-collection interpretation saved')


if __name__=='__main__':main()
