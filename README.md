# tiny-reranker

Performance experiments on a pretrained **MiniLM-L6-v2 reranker**, comparing
PyTorch, MLX, compilation, custom Metal/CUDA/Triton kernels, batching, and
Python/Rust serving on **Apple M5** and **NVIDIA L40S**.

[Read the article](https://medium.com/@nandanadileep29/13-performance-experiments-on-one-reranker-the-limits-of-faster-gpu-code-94670171711f) · [Results](experiments/E18/RESULTS.md) · [Reproduce the study](experiments/E18/REPORT.md)

## What helped?

| Warm direct scoring | Stock pairs/sec | Compiled pairs/sec | Paired gain |
|---|---:|---:|---:|
| Apple M5 / PyTorch | 87.4 | 102.1 | 13.7% |
| Apple M5 / MLX | 40.7 | 48.9 | 25.3% |
| NVIDIA L40S / PyTorch | 712.4 | 736.8 | 3.3% |

Same pretrained weights; FP32, 100 documents/query, batch 16, maximum 512 tokens.
Gains come from paired repetitions, not division of the displayed medians.

- **Compilation helped direct scoring**, but PyTorch's Apple gain did not produce a practical HTTP gain.
- **Custom Metal sped up one operation**, without an established model-scoring gain. CUDA/Triton did not beat compiled stock PyTorch.
- **Rust serving showed no practical throughput advantage.** Higher concurrency mostly added queueing in the single-worker service.
- **Accuracy:** all 11 implementation audits passed; all 12,288 accepted HTTP responses validated.

These findings apply to this workload. Confidence intervals, startup costs,
secondary sweeps, and limitations are in the [report](experiments/E18/REPORT.md).
NVIDIA HTTP was not measured.

## Explore

- [E18 benchmark code and protocol](experiments/E18/)
- [Article source](experiments/E18/ARTICLE_DRAFT.md) and [images](experiments/E18/medium/upload_images/)
- [Earlier experiments and training history](EXPERIMENT_LOG.md)
- [Experimental scratch model on Hugging Face](https://huggingface.co/ndileep/tiny-reranker-4.7m-experimental) — an educational 4.7M model with weak realistic retrieval quality; distinct from the pretrained model above.
