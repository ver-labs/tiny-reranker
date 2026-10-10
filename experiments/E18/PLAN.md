# E18 — Pretrained reranker inference: experiment plan and results ledger

**Status:** Planned; no E18 measurements collected.  
**Model:** `cross-encoder/ms-marco-MiniLM-L6-v2`.  
**Purpose:** Determine which framework, batching, serving, compilation, and custom-kernel choices improve inference for one pretrained reranker while preserving accepted ranking behavior.  
**Scope of this document:** A runnable study design and results-recording contract. Creating this plan does not start downloads, training, remote jobs, or benchmark collection.

## 1. Research question and boundaries

> For a pretrained MiniLM reranker on Apple M5 and NVIDIA L40S, which optimizations improve complete scoring or HTTP serving, at what workload sizes, and at what accuracy cost?

Use only the pretrained model in this new study. The scratch reranker and E1–E17 measurements remain historical work; they are neither a baseline nor a model-size control in E18. No training or fine-tuning is planned. One model cannot establish that parameter count caused an observed performance difference.

A valid outcome can be a speedup, a regression, an inconclusive difference, an accuracy failure, or an unsupported execution path. Record every outcome. Do not assume Rust, compilation, MLX, or a custom kernel will win.

Primary questions:

1. How do framework and pair microbatch size affect forward and complete-scoring throughput?
2. Does compilation improve the complete scoring workload after accounting for startup and compilation?
3. Does a Rust HTTP coordinator improve serving when the inference worker is identical?
4. Does fused residual addition + LayerNorm improve GPU operation time, model time, and request time?
5. Which configuration offers the best supported performance within the fixed accuracy contract?

## 2. Model and provenance

The [official model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) describes an English MS MARCO reranker, approximately 22.7M parameters, under Apache-2.0. Load raw scalar logits using `AutoModelForSequenceClassification`, not an embedding-model interface. Rank by descending raw logit; use document ID ascending for exact ties.

The [published config](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/main/config.json) currently specifies `BertForSequenceClassification`, six layers, hidden size 384, 12 heads, intermediate size 1536, 512 positions, GELU, and LayerNorm epsilon `1e-12`. Reverify these values from the immutable revision actually downloaded.

Before execution:

- [ ] Resolve the model repository to an immutable revision; save that revision, download date, model card, config, license, and exact parameter count.
- [ ] Download only required weights and tokenizer assets; hash every asset. Reuse a local copy only if its revision and hashes match.
- [ ] Load without remote code execution; pin the environment and dependency versions.
- [ ] Verify query/document token order, special tokens, segment IDs, attention masks, raw score output, pooler, and classification head.
- [ ] Use the same pinned tokenizer implementation and settings for all primary framework comparisons. Tokenize on CPU, then hand identical tensors to the framework for prepared-input checks.
- [ ] Record tokenizer throughput separately; a tokenizer change is a separate experiment.
- [ ] Record Mac model/RAM/OS, NVIDIA GPU/driver/runtime, package versions, compiler flags, source commit, and source/binary hashes.

Planned storage: `artifacts/e18_minilm_l6/` for downloaded/converted weights and `experiments/E18/` for scripts and compact reports. Keep large assets, score audits, traces, and caches outside Git, with committed checksum manifests.

## 3. Workload and data contract

Reuse the verified fixed retrieval candidates in `data/e16/development.jsonl` as **data only**, never the adapted model. Verify against source corpora and qrels plus the E16/E17 provenance manifests. Expected inventory: 162 SciFact and 323 NFCorpus development queries, 485 total, each with 100 BM25 candidates. Fail preflight if the inventory or hashes differ; investigate rather than silently replacing inputs.

These are scientific/biomedical retrieval workloads, not the full distribution of production requests. Pretrained-model training overlap is not established; this is primarily implementation equivalence and performance evaluation, not a new model-quality leaderboard claim.

| Dimension | Primary settings | Additional sensitivity checks |
|---|---|---|
| Documents per request, K | 10, 50, 100 | None initially |
| Combined token cap, L | **512** | 128, 256 for direct scoring; report quality/truncation effects separately |
| Pair microbatch, B | 8, 16, 32, 64 in direct sweep | B=1 as latency diagnostic; bounded OOM handling |
| HTTP concurrency, C | 1, 4, 8 | Open-loop saturation is optional, separately planned |
| Precision | FP32; TF32 disabled where applicable | Supported FP16/BF16 as separately qualified follow-up |
| Padding | Dynamic to longest pair in each microbatch | No unreported length sorting or bucketing |
| Candidate policy | First K original BM25 candidates | No positive injection or duplicate documents |

K, B, and C mean different things; keep all three in every result row. A 100-document request may require several pair microbatches. Primary HTTP uses B=16 for all backends. Check feasibility before freezing the plan; if B=16 cannot fit a required path, revise to one common feasible value for matched comparisons before measurement. A tuned B per backend is a separately labeled deployment comparison.

Accuracy audit: all 485 queries at each K and primary L=512: 77,600 scored pair occurrences per path. Assess changed token caps against their own eager reference; shortening context changes the task input and must not be treated as an implementation-only optimization.

Timing fixtures:

- Deterministically select 64 development queries, 32 per dataset, by ascending SHA-256 of `1801:dataset:query_id`, before inspecting timings.
- Use these fixtures for every timing path and session. Record original and tokenized length distributions, padding ratios, and truncation rates.
- Reserve the remaining 421 queries for a final fixed-configuration validation/performance pass, after choosing settings on the timing fixtures. The full correctness audit may include all queries; this is a reserve against timing over-selection, not an untouched model-quality test set.
- Use exactly the same query order, candidate order, microbatch construction, and final partial batch per matched run. Sorting candidate IDs before scoring is allowed only if applied identically and recorded; retain BM25 order for K selection.

## 4. Accuracy contract: freeze before optimization results

There are two distinct statuses: numerical parity and application acceptance. Report both, without changing thresholds after a failure.

**Primary same-device reference:** the pinned Hugging Face model in eager Python execution, FP32, `eval()` and `inference_mode()`, using the declared supported stock attention implementation. Use SDPA when supported; otherwise retain and label stock eager attention. Verify dispatch rather than assuming a requested flag activated an optimization.

BERT does not automatically share the scratch model's `TransformerEncoderLayer` fast-path behavior. Do not copy E17's global MHA flag and call that an optimized BERT implementation without evidence. Keep a mathematical/eager-attention diagnostic if helpful; do not deliberately slow the headline stock baseline to favor custom code.

Check representative short, long, heavily padded, and repeated inputs against CPU FP32 before scaling up. Validate cross-device reference differences separately. CPU is a correctness diagnostic, not a replacement for a GPU timing run.

| Check | Predeclared rule |
|---|---|
| Numerical parity, FP32 | Every score satisfies `abs(candidate-reference) <= 2e-5 + 1e-4 * abs(reference)` |
| Application maximum absolute score error | At most 0.01 |
| Top-1 change fraction | At most 1% in **each dataset/K cell** |
| Top-10 membership change fraction | At most 1% of lists in each dataset/K cell; report number of changed items too |
| nDCG@10 and MRR@10 non-inferiority | Lower paired-query bootstrap 95% bound for candidate minus reference is at least -0.001, in every dataset/K cell |
| HTTP transport correctness | Each response matches its own backend's qualified direct scores within FP32 tolerance and has identical ranking/tie-breaking |

These are study-specific acceptance criteria, not universal production requirements. Full ranking and top-10 ordering changes are always reported, even when membership and quality pass. K=10 top-10 membership is trivially the complete candidate set, so order and metrics are essential diagnostics. Record score gaps around changed ranks to explain near-tie sensitivity without overriding the gate.

Save maximum/median/P95/P99 absolute error, stabilized relative errors with denominator floor `1e-3`, and all ranking-change counts. Use 10,000 paired query bootstrap draws with seed 1801 for quality intervals. Retain all positive qrels, including unretrieved positives, and all queries with zero retrieved positives. Recompute graded linear-gain nDCG and recall against the existing TREC verifier.

A path that fails application acceptance can have diagnostic timings, clearly marked. It cannot be an accuracy-qualified performance winner. Never select a new tolerance because it makes a fast path pass. A future tolerance revision requires a new versioned protocol and fresh confirmation data.

## 5. Experiment matrix and implementation sequence

The following are separate interventions. Establish each effect before combining optimizations.

| ID | Experiment | Matched baseline | Required outputs / completion gate |
|---|---|---|---|
| P0 | Download, pin, load and validate MiniLM | Official model/config and CPU spot checks | Asset manifest, environment, frozen plan |
| P1 | Stock PyTorch on MPS and CUDA; B and L sweeps | Same-device stock FP32 at same L | Complete accuracy audit, forward/scoring curves, memory/OOM log |
| P2 | Faithful MLX implementation on Apple GPU | Same-device PyTorch FP32 and ordinary MLX | Conversion audit, layer-level checks, full accuracy, B/L curves |
| P3 | Compile stock PyTorch on MPS/CUDA; compile ordinary MLX if supported | Corresponding eager framework, same precision/B/L | Actual dispatch/graph evidence, qualification, first-use and warm results |
| P4 | Python versus Rust HTTP coordinators | Identical shared model worker per backend | K × C results, response validation, queue/processing breakdown |
| P5 | Profile selected stock paths | P1/P2 configurations | Stage breakdown, observed GPU operators, estimated opportunity |
| P6 | Custom Metal residual + LayerNorm | Ordinary MLX with identical surrounding graph | Operation and model accuracy; operation/forward/scoring measurements |
| P7 | Custom CUDA C++ and Triton residual + LayerNorm | Same-device stock PyTorch and matched explicit graph | Correctness, operation/forward/scoring measurements, dispatch proof |
| P8 | Qualified optimized paths in HTTP and combined configurations | Same coordinator/worker policy with stock backend | Per-intervention and combined request results; no assumed additive gains |
| P9 | Final confirmation, independent verification and article report | Frozen selected configurations and baselines | Reserve-fixture results, checksums, figures, decision table |

### P2: MLX conversion

Implement or reuse a compatible BERT cross-encoder only after inspecting its code and license. Map every parameter by name and shape, including embeddings, attention projections, dense layers, normalization, pooler, and scalar classification head. Preserve segment embeddings, activation, epsilon, mask semantics, and padding behavior. Require no missing/unexpected weights. Save layer-level comparisons to localize discrepancies before end-to-end qualification. A superficially similar encoder is insufficient.

### P3: compilation and stock optimization

Probe supported compilation modes in isolated processes with a bounded startup/compile timeout (initial ceiling 15 minutes per path). A timeout or unsupported backend is a recorded result, not a reason to silently change the workload. Declare candidate compilation modes and dynamic-shape strategy before timing. Materialize all measured shape signatures during warm-up and report recompilation/graph-break counts. A `torch.compile` wrapper alone is not evidence of generated GPU code.

Report first invocation with cache state stated as known-cold, known-warm, or unknown. If an actual isolated cache is used, document how it was isolated. Calculate compile break-even only if repeated warm measurements establish a saving. Keep default stock optimizations enabled in the practical baseline; an explicit unfused graph is an additional diagnostic baseline.

### P6/P7: architecture-correct custom kernels

Inspect the pinned BERT implementation before replacing anything. Its residual/normalization placement must be preserved; do not paste the scratch model's pre-norm graph into MiniLM.

Initial operation: `LayerNorm(residual + branch)` at the attention-output and feed-forward-output sites, with dropout inactive during inference. Preserve original output dtype and model epsilon. Confirm the actual eligible site count from the implementation (expected two per layer), then prove each replacement executes.

Hidden size 384 differs from the previous kernel contract. If a 512-wide tile is used, mask lanes 384–511 and divide statistics by **384**, not 512. Use FP32 accumulation in the primary FP32 path. Test zero/constant rows, near-zero variance, mixed magnitudes, representative real activations, row tails, padding, alignment, and documented strides. Reject unsupported cases or record fallback explicitly; no silent fallback in benchmark claims.

For CUDA/Triton, compare both against the identical surrounding explicit graph to isolate kernel effects, and against optimized stock execution to assess practical usefulness. For Metal, compare against ordinary MLX including any native fusion. CUDA C++ and Triton comparisons run on the same NVIDIA GPU; Metal/MLX comparisons run on the same Apple GPU. Cross-hardware tables are descriptive, not language speedup claims.

Do not add FlashAttention, quantization, new models, or unrelated kernels to the core experiment. Further kernels require a separately recorded profiling-based hypothesis.

## 6. HTTP protocol and realistic load

Primary Apple matrix: Python/Rust coordinator × PyTorch/MLX worker × K={10,50,100} × C={1,4,8}, B=16, L=512. This is 36 cells. Use shared worker code, one worker, queue capacity 32, no cross-request batching, identical CPU thread settings, inputs and serialization. Return all K ranked IDs and scores so every measured response can be audited; a top-10 response is a separate payload-size experiment.

Reuse E17's coordinator/protocol infrastructure, replacing scratch-model assumptions and worker model loading in E18-specific adapters. Avoid editing historical benchmark code or changing old result hashes. Document that Rust coordinates requests while the model still runs in a Python/framework worker; this is not a fully native Rust inference engine.

Use at least 256 completed requests per repetition (four passes over the 64 fixtures with a seeded identical permutation per matched repetition). This improves on short 16-request waves but still gives noisy P99; show variation and counts. Maintain C concurrent requests until the workload is exhausted. Report actual requests/s and pairs/s using measured wall time, never `1/P50`.

Latency timer ends after the full response is received. Validate outputs outside that request's timer; report load-generator validation cost because it can still affect submission cadence. Use persistent client connections, identical connection limits, and a client that does not saturate a CPU core before interpreting server throughput. Record load-generation policy and client resource use.

The primary load is local loopback, fixed-concurrency closed-loop. It is not an internet benchmark or an open-loop overload/SLA claim. Keep timeouts fixed after feasibility preflight (initial request timeout 120 s, worker timeout 120 s); retain all failures, timeouts and resulting queue behavior.

After direct qualification, run selected compiled/custom paths through the same Python coordinator at C=1/4/8 to measure whether direct gains survive HTTP. Additional Rust comparisons for those paths are optional unless a coordinator interaction is a main claim. NVIDIA HTTP comparison of stock/custom/compiled qualified paths is required if claiming NVIDIA **request** speedups; otherwise report NVIDIA scope as direct scoring only and mark HTTP unmeasured.

## 7. Measurement, profiling and uncertainty

Keep four timing scopes separate:

1. **GPU operation:** resident inputs, correct GPU events/trace timing; graph replay amortization labeled explicitly.
2. **Prepared forward:** GPU inputs already prepared; synchronize before and after model execution; exclude CPU score conversion.
3. **Complete direct scoring:** CPU tokenization, transfers/framework array construction, model execution, CPU-readable scores; exclude retrieval, HTTP and ranking sort.
4. **HTTP request:** client-visible queueing, IPC, tokenization, GPU execution, CPU scores, sort and response serialization.

Profile a separate diagnostic pass; disable profilers during headline timing. Track tokenization, transfer/array construction, GPU work, CPU score conversion, sorting, serialization and queueing. Use synchronized stage wall time only with that label; GPU kernel time requires device evidence. Estimate the maximum possible request saving from the measured target operation's share before interpreting a kernel speedup.

Repetition policy:

- Direct/model comparisons: at least **15 paired repetitions across at least five fresh process sessions**; three repetitions per session.
- HTTP coordinator comparison: **16 paired repetitions across eight fresh sessions**, two repetitions per session. Balance the four startup positions twice across the four Apple paths.
- Freeze the full participating path list before scheduling. If direct comparison has N paths, use at least `max(5,N)` sessions and a balanced cyclic/Latin-square order, expanding to a multiple of N where necessary. Do not add paths after timings and leave order unbalanced.
- Balance K/B/L/C cell ordering and forward/scoring scope order with seed 1801. Pair comparisons within identical session/repetition/cell indices.
- Warm each path and all measured shape signatures; exclude warm-up and compilation from steady-state results. Report warm-up count and duration.
- Run one benchmark at a time per GPU. Record AC power, visible competing activity, CPU threads, memory, and available temperature/clocks/utilization snapshots. Do not claim exclusive hardware isolation from snapshots alone.
- Keep raw observations and outliers. A technical invalidation (wrong device, failed checksum, partial run) requires an explicit exclusion record and retained raw data; being slow is not an exclusion rule.

Report median, IQR, min/max, P50/P95/P99 request latency, throughput, errors, memory scope, and per-session drift. Bootstrap matched log-throughput ratios (or matched latency differences) using **session means** as the independent units, 10,000 draws, seed 1801. Never bootstrap individual HTTP requests as independent hardware repetitions. A small number of sessions still produces coarse conditional uncertainty.

Declare K=100, L=512, B=16 complete-scoring comparisons and K=100/C=4 HTTP comparisons as primary endpoints. Show every other cell as secondary/exploratory. Use Holm correction for any family of confirmatory significance tests; label unadjusted confidence intervals as exploratory rather than claiming multiple independent discoveries. Call a primary throughput gain practically supported only when its session-level interval is entirely above a predeclared ratio of **1.05**; a smaller statistically detectable gain may be reported as small, not meaningful by this study's rule.

Do not stop early because a preferred implementation appears to win. Stop only for declared completion, execution failure, hardware unavailability or explicit budget constraints, recording partial status.

## 8. Execution stages, budget and completion gates

| Stage | Work | Exit condition |
|---|---|---|
| A — setup | Pin model/data/env, create adapters, verify shapes and hardware, run bounded feasibility probes | `plan.json` frozen with hashes, feasible matrix, accuracy rules and runtime estimate |
| B — baseline | Stock PyTorch accuracy, tokenizer/forward/scoring profile, B/L feasibility sweep | Working reference plus observed bottleneck report |
| C — framework/compile | Validate MLX conversion; qualify stock attention and compilation paths | All supported paths have explicit pass/fail status and dispatch evidence |
| D — serving | Run repeated four-path Apple coordinator matrix | Complete raw requests, no unreported failures, checked responses |
| E — kernels | Implement one fused operation on Metal/CUDA/Triton; verify and benchmark | Operation/model accuracy and repeated direct results |
| F — integration | Qualified optimized HTTP paths and fixed-configuration reserve pass | Whole-request evidence or explicit unmeasured scope |
| G — report | Recompute results independently, inspect plots, write conclusions, update checklist | Reproducible verified report; all pending items stated |

Use a bounded preflight of at most 30 minutes per available device to estimate the complete schedule. Save estimated runtime, storage, selected microbatches and allocation requirements before long jobs. The primary Apple HTTP design has 36 cells × 16 repetitions × 256 requests = **147,456 requests**, so it is a substantial study, not an instant script. Estimate this honestly rather than promising a short run. Direct and optional matrices add work.

If runtime is excessive, reduce a secondary dimension or define an explicitly smaller versioned protocol **before headline collection**. Do not silently reduce repetitions or the sample after seeing outcomes. No paid GPU rental is assumed or authorized by this plan. Use an available approved cluster allocation, persist progress/checkpoints of benchmark collection, and keep re-runs resumable without mixing changed plans.

GPU execution failure must not become silent CPU fallback. SSH/VPN/authentication blocks pause only the affected GPU work; continue independent local preparation. Never put credentials in code, commands, logs, or manifests.

## 9. Results ledger — fill only from verified artifacts

Use `PENDING`, `UNSUPPORTED`, `FAILED_ACCURACY`, `INCOMPLETE`, or `VERIFIED` explicitly. A dash means no measurement, not zero. Generate full tables programmatically; the rows below define the report schema and are not predictions.

### 9.1 Qualification

| Device | Path | Revision/precision/L | Strict parity | Application gate | Max/P99 score error | Top-1 / top-10 changes | nDCG/MRR delta + CI | Artifact |
|---|---|---|---|---|---|---|---|---|
| Apple M5 | Stock PyTorch | PENDING | — | — | — | — | — | — |
| Apple M5 | MLX | PENDING | — | — | — | — | — | — |
| Apple M5 | Compiled PyTorch | PENDING | — | — | — | — | — | — |
| Apple M5 | Custom Metal | PENDING | — | — | — | — | — | — |
| NVIDIA | Stock PyTorch | PENDING | — | — | — | — | — | — |
| NVIDIA | Compiled PyTorch | PENDING | — | — | — | — | — | — |
| NVIDIA | Custom CUDA | PENDING | — | — | — | — | — | — |
| NVIDIA | Custom Triton | PENDING | — | — | — | — | — | — |

### 9.2 Direct scoring / batching / context

| Device | Path | Precision | L | K | B | Forward pairs/s | Complete-scoring pairs/s | Paired ratio + session CI | Memory scope | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| PENDING | — | — | — | — | — | — | — | — | — | PENDING |

### 9.3 HTTP

| Device/backend | Coordinator/path | L / B | K | C | Requests/session/repeats | P50/P95/P99 ms | req/s / pairs/s | Queue/processing ms | Failures/timeouts | Paired ratio + CI |
|---|---|---|---:|---:|---|---|---|---|---|---|
| PENDING | — | — | — | — | — | — | — | — | — | — |

### 9.4 Kernel and compilation

| Device | Operation/path | Shape/dtype | Native operation time | Custom time | Paired speedup + CI | Model/request effect | Correctness | Status |
|---|---|---|---:|---:|---|---|---|---|
| PENDING | — | — | — | — | — | — | — | PENDING |

| Device/path | Compile configuration | Cache state | Compile/first-use time | Graph breaks/recompiles | Warm saving + CI | Break-even requests | Status |
|---|---|---|---:|---|---|---|---|
| PENDING | — | — | — | — | — | — | PENDING |

### 9.5 Decision table and experiment notes

| Question | Measured answer | Evidence | Limitation | Adopt / keep baseline / investigate |
|---|---|---|---|---|
| Best framework/B for each workload? | PENDING | — | — | — |
| Does compilation improve complete scoring? | PENDING | — | — | — |
| Does Rust improve matched HTTP serving? | PENDING | — | — | — |
| Does each custom kernel improve the whole workload? | PENDING | — | — | — |
| Do combined optimizations retain their individual gains? | PENDING | — | — | — |

For each completed experiment, record: hypothesis; exact single change; fixed controls; plan/source hashes; command; dates/hardware; accuracy outcome; raw artifact; measured effect and uncertainty; interpretation; limitation; decision. Preserve failed attempts and distinguish fixes before collection from post-hoc analyses.

## 10. Deliverables and handoff

Suggested files under `experiments/E18/` (future implementation, not existing commands):

- `PLAN.md` — this human-readable plan, with an execution-status checklist.
- `plan.json`, `model_manifest.json`, `data_manifest.json`, environment lock files — frozen contract and provenance.
- `download_model.py`, `prepare.py`, `inference.py`, `accuracy.py` — assets, requests, framework adapters and audits.
- `mlx_model.py`, Metal/CUDA/Triton kernel sources and loaders — faithful inference ports with explicit contracts.
- `direct_benchmark.py`, `http_benchmark.py`, worker adapters, device launch scripts — timing harnesses.
- `raw/`, ignored `scores/` and `traces/`, `checksums.json` — original evidence, including failed runs.
- `verify.py`, `verification.json`, `summarize.py` — independent score/ranking/timing/statistics recomputation.
- `RESULTS.md` — full machine-generated tables plus experiment notes from section 9.
- `REPORT.md` — readable article-style findings, methods, uncertainty and limitations.
- `plots/` — throughput versus B/K, HTTP latency versus K/C, paired uncertainty, profiling breakdown, and accuracy/performance trade-offs.

Reuse E17's scheduling, hashing, response validation, independent verification and reporting ideas. Inspect imports carefully: its backends, checkpoints, model graphs and some metrics assumptions are scratch-specific. Do not reuse old scores or write new results into historical directories.

Test the changes that matter: complete weight mapping, mask/tokenizer semantics, custom-kernel numerics, fixed request counts and uniqueness, no hidden fallback, accuracy gate decisions, paired/session aggregation, failed-run exclusion rules, worker/coordinator parity, and cleanup of started processes. Independently recompute saved artifacts after execution. Inspect every plot for scope, units, uncertainty and honest axes.

Completion checklist:

- [ ] Assets and data pinned; baseline verified.
- [ ] Feasibility and immutable execution plan recorded.
- [ ] Framework/batching/context results recorded.
- [ ] Compilation outcomes and actual dispatch recorded.
- [ ] Realistic repeated Python/Rust serving comparison completed.
- [ ] Metal/CUDA/Triton fused operation implemented and checked, or precise blocking limitation documented.
- [ ] Operation gains connected to measured model/request effects.
- [ ] Final selected configurations confirmed on reserve fixtures.
- [ ] All score/ranking/timing artifacts independently verified.
- [ ] Results tables, figures and report completed, including negative results.
- [ ] Parent roadmap updated only for completed work; existing unrelated changes preserved.
- [ ] Logical implementation/results/documentation commits made when executing the study; no push unless requested.

Article framing after results exist: **“What Actually Speeds Up a Pretrained Reranker? Experiments Across the Inference Stack.”** Conclusions must follow measured evidence. There is no requirement to find a winning optimization.

## 11. Primary references and local evidence

- [MiniLM-L6-v2 model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) and [configuration](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/main/config.json): model identity and interface; resolve immutable revisions at execution.
- [PyTorch compilation documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html): inspect documentation matching the pinned environment before implementing compiler settings.
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): API reference, not a guarantee of speedup or model compatibility.
- [E17 Apple report](../E17/REPORT.md), [NVIDIA report](../E17/NVIDIA_REPORT.md), [E17 frozen plan](../E17/plan.json): methodological lessons and reusable infrastructure, not E18 results.

References checked when preparing this plan. Execution must record actual installed versions and downloaded revisions rather than assuming these moving documentation URLs are immutable.
