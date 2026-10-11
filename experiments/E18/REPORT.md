# E18 — Pretrained reranker inference study

**Execution status: complete and independently verified.** Collection ran on 2026-10-10–11 on Apple M5 (24 GB, macOS 26.5.2, PyTorch 2.14.1, MLX 0.32.3) and NVIDIA L40S (PyTorch 2.14.1+cu130). Python 3.12.13 and Transformers 5.18.0 were pinned; full hardware/software snapshots are retained. See [measured tables](RESULTS.md), the immutable [machine-readable contract](plan.json), its [batch-control addendum](protocol_addendum.json), and the [Apple prepared-forward addendum](apple_forward_addendum.json).

This study follows the user-authorized E18 plan, using the pretrained `cross-encoder/ms-marco-MiniLM-L6-v2` at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. It has 22,713,601 parameters, six BERT layers, hidden size 384, 12 heads, and a 512-token combined query/document cap. These measurements do not use the scratch reranker or E16 adaptation checkpoint. No training was performed. Weight SHA-256 is `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`; tokenizer assets and source corpora are individually hashed in the manifests.

## What the measurements establish

On NVIDIA L40S, all five matched FP32 implementations passed the predeclared numerical and application accuracy gates across 485 queries at each of K=10, 50, and 100: 77,600 pair occurrences per path. Stock PyTorch is the same-device practical SDPA reference, not a deliberately slowed mathematical attention baseline. Four padded CPU diagnostic pairs and layer checks are separate correctness evidence, not a full CPU audit. Stock PyTorch dispatch evidence records six memory-efficient SDPA sites per forward on NVIDIA and six native MPS SDPA math sites on Apple; compilation uses generated Inductor GPU code. Neither is described as FlashAttention merely because it uses the SDPA API. The explicit residual graph agrees exactly with stock scores. Maximum absolute score errors for compiled, CUDA, and Triton are approximately `1.00e-5`, `1.55e-5`, and `1.45e-5`. No top-1 or top-10 membership changes occurred. Compiled PyTorch changed 43 full lists and nine top-10 orders; CUDA changed 33 and five, and Triton 33 and nine. Those near-tie diagnostics, including changed IDs, remain auditable even when headline checks pass.

For K100/B16/L≤512 direct scoring, compiled stock PyTorch achieved 736.8 pairs/s versus 712.4 for stock eager. The paired stock/candidate time ratio was 1.033, with an exploratory session-bootstrap 95% interval [1.032, 1.034]. CUDA and Triton achieved 715.0 and 717.5 pairs/s. They were slower than compiled stock: compiled/custom time ratios were 0.971 [0.969, 0.974] and 0.975 [0.972, 0.978]. Custom residual + LayerNorm therefore did not beat the fastest measured stock configuration on this workload. The compile gain is small and below the predeclared practical threshold whose lower interval bound must exceed 1.05; the frozen practical-selection rule retained stock.

These intervals use five independently started session cycles, each with three matched repetitions, rather than treating queries as independent performance trials. They are unadjusted exploratory intervals, conditional on this machine and protocol. The explicit graph versus stock comparison remains inconclusive. Secondary B32 scoring observations had conspicuous early-period variation; every observation was retained, and its cause was not established. On NVIDIA, B=1 is costly and larger batches plateau, but only primary B16/L512 received the complete accuracy audit. Other B/L combinations are labeled diagnostic.

The selected-query CUDA profile attributes roughly 1.9% of summed leaf GPU-kernel time to all 13 LayerNorm kernels, including the embedding normalization. Only 12 residual normalizations are eligible for the intervention. Residual-addition time is not included in that 1.9%, so it is not an upper bound for the fused intervention. Tokenization dominated this query's synchronized stage breakdown; this is a selected-query diagnostic, not an all-query fraction or an HTTP decomposition. Isolated residual-normalization operation results varied with row count and frequently regressed. They cannot establish whole-model or request gains.

A fresh software-cache startup diagnostic measured compiled first scoring at 24.74 seconds versus 0.46 seconds for stock. GPU driver cache state was unknown, and a single ordered startup pair cannot establish a startup uncertainty interval. Full model-setup plus first-score differences imply about 11,287 K100 queries to amortize the observed overhead using the median paired warm saving; this is an illustrative diagnostic, not a production break-even estimate. Combining the CUDA adapter with full-graph compilation was unsupported because the ctypes/data-pointer interface breaks tracing. That failure remains visible, outside the qualified timing paths.

## Apple results

All six Apple paths passed both numerical and application gates over the complete development audit. Compiled PyTorch's maximum absolute score error was `7.15e-6`; MLX/compiled MLX reached `1.74e-5`, and Metal/compiled Metal reached `1.67e-5`. Top-1 and top-10 membership never changed. Compiled PyTorch changed three full rankings; MLX variants changed seven full rankings and two top-10 orders; Metal variants changed two full rankings. nDCG@10 and MRR@10 differences were zero on Apple. This is numerical tolerance compliance, not bitwise equivalence.

The primary K100/B16/L512 scoring comparison used 18 paired repetitions across six independently started process cycles. Compiled PyTorch achieved median 102.1 pairs/s versus stock's 87.4; its paired stock/candidate time ratio was 1.137 [1.081, 1.198], satisfying the predeclared practical criterion. MLX achieved 40.7 pairs/s; compiled MLX achieved 48.9, with an MLX/compiled-MLX ratio of 1.253 [1.139, 1.379]. The frozen selection chose compiled PyTorch overall and compiled MLX within MLX. Both selections and stock passed the 421-query K100 reserve audit; this confirms accuracy, not a second independent performance effect.

Metal's isolated residual-plus-LayerNorm operation was faster than native MLX at all three row counts: paired ratios 1.406 [1.196, 1.812], 1.377 [1.362, 1.393], and 1.499 [1.460, 1.525]. That gain did not establish a complete-scoring improvement: MLX/Metal was 0.909 [0.799, 1.015]; compiled MLX/compiled Metal was 0.966 [0.821, 1.187]. The custom Metal model remained substantially slower than stock PyTorch. Operation latency includes host dispatch and synchronization on Apple, whereas NVIDIA uses GPU events; these are separate same-device comparisons.

The separately matched optimized HTTP families show why direct timing is insufficient. Compiled PyTorch's K100 HTTP throughput ratios were 0.995 [0.985, 1.005] at C1 and 1.003 [0.990, 1.016] at C4, both inconclusive; C8 was 1.010 [1.0001, 1.021], a small exploratory effect below the practical threshold. Compiled MLX improved its own matched HTTP baseline: 1.112 [1.062, 1.171], 1.100 [1.023, 1.172], and 1.157 [1.103, 1.221] at C1/C4/C8. Only C1/C8 clear the 5% lower-bound criterion. Each family started new matched baseline workers; these comparisons do not borrow baseline observations from the coordinator study.

## Realistic HTTP findings

The primary matrix processed 9,216 successful requests / 491,520 pairs across 32 independently started services, with 16 paired repetitions per K/C point. All returned scores and rankings validated; there were no measured failures or timeouts. Two optimized follow-up families added 3,072 requests / 307,200 pairs across 32 more service starts, also without failures. Warm-up requests and excluded technical attempts are counted separately.

At concurrency one, the median client P50 and median requests/s were:

| K | Worker | Python P50 ms / req/s | Rust P50 ms / req/s |
|---|---|---:|---:|
| 10 | PyTorch | 73.9 / 13.61 | 71.0 / 14.24 |
| 50 | PyTorch | 374.1 / 2.64 | 367.5 / 2.73 |
| 100 | PyTorch | 762.1 / 1.320 | 739.4 / 1.343 |
| 10 | MLX | 161.5 / 4.92 | 158.2 / 4.40 |
| 50 | MLX | 785.2 / 1.14 | 835.0 / 0.85 |
| 100 | MLX | 1526.5 / 0.611 | 1524.2 / 0.607 |

Rust's PyTorch throughput ratios favored Rust in six of nine unadjusted exploratory intervals, with point gains about 2–4%. None had a lower interval bound exceeding the practical 5% threshold. Three PyTorch cells and eight of nine MLX cells included parity. MLX K10/C1 favored Python: Python/Rust total-time ratio 0.868 [0.802, 0.938]. These results do not establish a universal coordinator or language advantage. P50 and throughput need not move together when slower requests dominate total elapsed time.

Concurrency raised latency with little throughput gain. For Python/PyTorch K100, P50 rose from 0.762 seconds at C1 to 2.997 at C4 and 5.917 at C8, while throughput remained about 1.32 requests/s. C4 median queue wait was 2.240 seconds against 0.755 seconds processing time. Python/MLX K100 rose from 1.526 to 6.081 and 12.080 seconds, with throughput around 0.61 requests/s. This is consistent with a serialized one-worker queue; it does not characterize open-loop overload. Full P95/P99, queue/processing percentiles, readiness, warm-up, RSS snapshots and session intervals are in [the generated tables](RESULTS.md).

Apple's separate fresh software-cache startup pair measured stock first scoring at 0.51 seconds and compiled scoring at 7.68 seconds. Newly created Inductor/Triton directories do not clear system shader/driver caches. The observed model-setup-plus-first-score difference, divided by the paired median warm K100 saving, gives an illustrative 41-query amortization; the first-score-only estimate is 46. Neither has a startup uncertainty interval, nor predicts HTTP break-even. MLX first-use probe cache state remains unknown.

The Apple four-query batching sweep was non-monotonic; B1 sometimes exceeded larger batches. Dynamic padding and framework execution differ with B, and no causal explanation was established. The fixed primary B16 was retained; this study does not prove it is globally optimal or qualify secondary batch sizes through a complete accuracy audit.

## Workload and statistical limits

All requests use unmodified first-K fixed BM25 candidates, without positive injection or document duplication. The candidate inventory contains 162 SciFact and 323 NFCorpus development queries. All relevance judgments, including unretrieved positives and queries with no retrieved positive, remain in the metrics. Timing uses 16 predetermined queries, eight per dataset; secondary sweeps use four predetermined queries. The original 64-query design fixture remains recorded; 421 queries outside that original fixture form a final one-pass confirmation. The smaller protocol was authorized and frozen before headline timings.

The actual K100 fixture contains 1,600 unique query-document pairs. Untruncated token-length medians are 383 for SciFact and 358.5 for NFCorpus; the 512-token cap truncates 16.125% and 8.5% respectively. A 128-token cap truncates almost every pair. On the separate 16-query context diagnostic, L128 minus L512 nDCG@10 was -0.0165 [-0.0700, 0.0178] and MRR@10 was -0.0365 [-0.1927, 0.0835]; L256 nDCG was +0.0017 [-0.0053, 0.0111]. These broad intervals do not qualify shorter context as application-equivalent. The timing sweeps remain diagnostics, and L512 stays fixed for primary comparisons.

Document count K, pair microbatch B, and concurrency C are distinct. HTTP is local loopback, fixed-concurrency closed-loop, with K10/50/100, C1/4/8, B16, one worker, queue capacity 32, and no cross-request batching. Python and Rust use the same Python model worker, request validation, sorting, response contents, and timeouts. The client validates every returned score and ranking against qualified direct inference after recording full-response latency. Sorting and serialization belong to HTTP timing, but not direct scoring. Server readiness and compiler warm-up are separate measurements. The [HTTP warm-up addendum](http_warmup_addendum.json), frozen before HTTP collection, adds three fully validated K10/50/100 requests per service (160 pairs) and connection warm-up before every timed repetition; their time and counts are separate.

Each HTTP repetition contains 16 requests, so C8 has only two request waves. Its P99 is close to a short-wave maximum and cannot characterize production tails or open-loop overload. Performance intervals must use the eight independent service-start cycles, not individual requests. RSS snapshots cover coordinator and worker processes; shared memory may be counted twice and snapshots are not total-system peak memory.

Prepared-input forward loops enqueue GPU computation and synchronize once at the end. Apple MLX uses [asynchronous evaluation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html), with an exact synchronous/asynchronous first-batch score check before each timing cell. This scheduling correction was fixed before collecting any Apple direct timing. CPU-readable direct scoring and HTTP retain their original synchronization behavior.

Apple and NVIDIA differ in hardware, libraries, kernels, and power conditions. Their numbers cannot support language-level or cross-hardware speedup claims. Other system activity was visible; exclusive physical-machine isolation, fixed GPU clocks, and continuous GPU temperature monitoring were not established. NVIDIA before/after temperature, clock, utilization, and memory snapshots are retained in every run; Apple temperature/clocks were unavailable. AC power and package/environment snapshots are recorded where available. A temporary idle-sleep inhibitor kept Apple execution awake without changing global power settings.

MLX's default attention API does not imply fused attention for this model: its 32-wide heads are unsupported by the installed full-attention fused kernel. Both the [installed-version implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L697), recorded runtime `force_fused=True` rejection, and graph dispatch evidence establish the mathematical fallback. This constraint is reported rather than describing a requested API flag as proof of an optimization.

The development retrieval metrics describe this pretrained model on these fixed candidates. Training overlap is unknown; they are not a leaderboard result. Shortening context changes task inputs and is a separate 16-query quality diagnostic, not an implementation-equivalence optimization. Mixed precision, open-loop saturation, distributed serving, internet latency, cross-request batching, and NVIDIA HTTP performance are unmeasured.

## Reproduction and evidence

From the child repository, after installing the pinned environment and verifying local assets:

```sh
.venv/bin/python -m pytest tests/test_e18_contract.py tests/test_e18_service.py -q
cargo test --manifest-path experiments/E18/rust-server/Cargo.toml
cargo build --release --manifest-path experiments/E18/rust-server/Cargo.toml
.venv/bin/python -u -m experiments.E18.launch --device mps
.venv/bin/python -u -m experiments.E18.finish_device --device mps
.venv/bin/python -u -m experiments.E18.startup_probe --device mps --path stock
.venv/bin/python -u -m experiments.E18.startup_probe --device mps --path compiled
.venv/bin/python -m experiments.E18.compile_amortization --device mps
.venv/bin/python -m experiments.E18.summarize
.venv/bin/python -m experiments.E18.ranking_audit
.venv/bin/python -m experiments.E18.compact_reports
.venv/bin/python -m experiments.E18.report
.venv/bin/python -m experiments.E18.plots
.venv/bin/python -m experiments.E18.verify
```

The finish command must run after the launcher exits, never concurrently with its GPU work. Apple GPU and loopback execution may require running outside the filesystem sandbox; unavailable GPU execution raises an error rather than falling back to CPU. The downloader pins every required model asset. Preparation/freezing are one-time setup steps on a fresh output directory: do not overwrite the committed frozen manifests or plan. Existing completed observations are resumed only when their plan hashes match.

NVIDIA setup, preflight, headline, post-study, and isolated startup commands are in the `.sbatch` and build scripts in this directory. The completed study used Slurm partition `rg-nextgen-hpc`, `--gres=gpu:l40s:1`, four allocated CPU cores, and project-local CUDA 12.6.85 compiler archives whose official SHA-256 hashes were verified. No other user's job was changed. The remote copy had no Git metadata; [source hashes](cuda_source_manifest.json) and [binary hashes](binary_manifest.json) identify the measured snapshot.

Compact raw per-session observations are in `raw/`. Large auditable gzip score/ranking files and HTTP responses remain outside Git in `scores/` and `raw/`; their SHA-256 hashes are committed in summaries and the final checksum manifest. Interrupted Apple accuracy output was retained and recovered at complete-record boundaries before resuming. Independent verification checks candidate identity, uniqueness, scores, rankings, direct metric formulas/TREC parity, error quantiles, accuracy decisions, expected counts, and session-level timing intervals. The final [verification](verification.json) checks all 36 Apple and 25 NVIDIA direct path sessions, 64 accepted HTTP service artifacts / 12,288 requests, six operation sessions per device, startup score parity, context diagnostics and all four 421-query reserve artifacts. The [checksums](checksums.json) cover raw results and local large audits. The retained first HTTP cycle contains 864 successful diagnostic requests plus the request-free startup failure; these are excluded from accepted headline counts, not deleted.

The focused contract tests passed (10), and the real Python/Rust coordinator tests passed (16). Their worker is a deterministic protocol fixture; these tests do not substitute CPU timing for GPU measurements. NVIDIA synthetic, real-activation, strided-rejection, scalar-offset, and dtype checks passed. Apple full audits, repeated direct/HTTP/operation measurements, profiling, context checks, startup probes, and reserve audits are complete and independently verified. The Rust file-launch bootstrap originally shadowed the stdlib `profile` module during startup; the failing request-free attempt and entire first service cycle are retained under `raw/attempts/`, excluded, and replaced in the original four-path order after the other primary cycles. The [worker provenance note](worker_bootstrap_provenance.json) records the bootstrap edit and the limits of HEAD-only source metadata. Remaining successful observations are preserved. The bootstrap correction changes import-path initialization, not inference or protocol behavior.

## Figures and decisions

Inspect [HTTP workload curves](plots/http_workloads.png), [optimized HTTP follow-ups](plots/optimized_http.png), [Apple paired direct intervals](plots/mps_primary_ratios.png), [NVIDIA paired direct intervals](plots/cuda_primary_ratios.png), [Apple batching](plots/mps_batching.png), [NVIDIA batching](plots/cuda_batching.png), and [resident operation variation](plots/operations.png). All seven figures were inspected after generation; interval titles were adjusted to prevent clipping.

For this model and matched FP32/B16/L512 protocol, use compiled PyTorch when warm direct scoring is the target on Apple; retain stock as the practical NVIDIA baseline under the frozen 5% rule, while recognizing compiled stock's small measured lead. Custom CUDA/Triton do not beat compiled stock, and Metal's operation gain does not establish a model gain. For HTTP, compiled MLX improves its own backend; compiled PyTorch and coordinator gains remain small or inconclusive at the practical threshold. These are workload-specific engineering findings, not production tail-latency guarantees or cross-hardware language comparisons.
