---
language:
- en
license: cc-by-nc-4.0
library_name: pytorch
datasets:
- sentence-transformers/msmarco-bm25
tags:
- text-ranking
- experimental
- educational
- trained-from-scratch
- inference-performance
---

# tiny-reranker-4.7m-experimental

## Summary

Experimental **4.72M-parameter reranker trained from scratch**, released for education, reproducibility, and inference-performance experiments. General retrieval quality is weak in the evaluations performed. Publishing this artifact does not establish model quality or usefulness for general search.

This is the **original full-training epoch 3**, `runs/full_train/epoch_0003.pt`, SHA-256 `2ea2608a9ec98083b89606d13bed1d7c6e7a90dc6160bbe44732166137538d52`. It is distinct from the later E16 adaptation checkpoint.

## Architecture

Custom bidirectional transformer cross-encoder, **4,717,185 parameters**, four encoder layers, hidden size 128, four attention heads (32 dimensions each), feed-forward size 512, vocabulary 30,522, two segment embeddings, and 128 learned positions. All model weights were initialized randomly (E1 seed 42); no pretrained language-model weights were used.

Token, position, and segment embeddings are summed, then LayerNorm and dropout 0.1 are applied. Encoder layers use pre-normalization, exact GELU, biased linear layers, and LayerNorm epsilon 1e-5. A final LayerNorm precedes CLS pooling and a biased 128-to-1 scoring head. Attention is bidirectional with padding keys masked; there is no causal mask. Outputs are scalar raw relevance logits, **not calibrated probabilities**. Dropout is disabled in evaluation.

The exact saved tokenizer reuses assets from [cross-encoder/ms-marco-MiniLM-L6-v2 at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a). This is tokenizer reuse, not a pretrained BERT fine-tune. It uses uncased BERT WordPiece normalization, accent stripping and Chinese-character handling as stored in tokenizer.json. Special IDs: PAD=0, UNK=100, CLS=101, SEP=102, MASK=103.

Inputs are `[CLS] query [SEP] document [SEP]`; query and first separator have segment 0, document and final separator segment 1. Right padding is dynamic. `longest_first` truncation removes tokens from the longer sequence, including the query when necessary. **128 tokens covers both texts and all three special tokens**. The reused tokenizer's 512-token metadata does not change the model's 128-token cap. `config.json` records all architecture and input behavior.

## Training

Source: [sentence-transformers/msmarco-bm25](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25/tree/ce8a493a65af5e872c3c92f72a89e2e99e175f02), `triplet` configuration, `train` split, pinned revision `ce8a493a65af5e872c3c92f72a89e2e99e175f02`, 502,931 source triplets.

Whole queries are split using Unicode casefold, collapsed whitespace, and stripping for query IDs only: SHA-256 of normalized query UTF-8; first 16 hex digits modulo 10 assign buckets 0–7 to training, 8 to validation, 9 to test. Source text is preserved for model inputs. Invalid or empty triplets are excluded; positive/negative documents become labels 1/0. Exact query/document string pairs are deduplicated and pairs with conflicting labels are removed. The saved audit reports 918,054 clean pairs overall, 43,904 conflicting pairs excluded, and **735,276 training pairs** (367,638 per label). Normalized query-ID overlap between splits is zero; semantic/paraphrase overlap is not guaranteed absent.

Three full epochs used all cleaned training pairs once per epoch (2,205,828 pair exposures). Training seed 123, NumPy permutation seed `123 + epoch_index`, dynamic padding, batch size 16, no gradient accumulation or activation checkpointing. Pointwise BCEWithLogits loss; AdamW learning rate 3e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01. No learning-rate schedule. All parameters trained using FP16 autocast with GradScaler and FP32 parameter storage on Apple M5 GPU / PyTorch MPS, MacBook Air with 24 GB unified memory. Saved training environment: Python 3.12.13, PyTorch 2.14.1, NumPy 2.5.3. There were 137,865 attempted batches, 137,802 optimizer updates and 63 scaler-skipped updates. Epoch-3 mean training BCE was 0.46799069444169866. These are training diagnostics, not quality guarantees.

## Intended uses

- Learning small-transformer initialization and training.
- Reproducing inference, GPU-kernel and serving experiments.
- Investigating numerical parity, score sensitivity and ranking changes.
- Studying limitations of constructed ranking evaluations.

## Limitations

General retrieval usefulness has not been demonstrated. This is not established as suitable for deployment: limited training setup, no pretrained language-model initialization, a 128-token cap, and weak transfer to realistic retrieval. Query-removal findings suggest substantial passage preferences on sampled constructed lists; they do not establish the exact cause of poor generalization. Labels from mined negatives and limited candidate lists constrain interpretation. Small parameter count alone does not establish suitability for any device or latency target. Scores may rank irrelevant documents highly.

## Evaluation

**Unfavorable realistic retrieval results:** fixed BM25 top-100 over full BEIR corpora, no relevance injection, complete official test qrels, 128-token cap. BM25Okapi used k1=1.5, b=0.75, epsilon=0.25; preprocessing casefold plus ASCII alphanumeric tokens, no stemming or stopword removal. Same-query/document IDs excluded. Queries lacking a retrieved relevant document remain in metric denominators.

| Test dataset | Queries / pairs | Original scratch nDCG@10 | TinyBERT-L2-v2 nDCG@10 (128 tokens) | BM25 nDCG@10 |
|---|---:|---:|---:|---:|
| SciFact | 300 / 30,000 | 0.013375328821900047 | 0.5703154525374805 | 0.6522849449944161 |
| NFCorpus | 323 / 32,300 | 0.05362354030597277 | 0.3070801920238763 | 0.30641376492779704 |

Size-matched [TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2/tree/81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc) has 4,386,049 parameters, pretrained weights and a different training history. Equal token limits do not equalize those histories. These tests cover two datasets, not the complete BEIR suite.

**Constructed-list test:** 45,665 retained queries, 91,337 query-document pairs, mostly two candidates per query, from cleaned MS MARCO mined triplets. Candidate document IDs are passage hashes; duplicates are removed per query. The original source test split has 50,112 query IDs; 4,447 have no cleaned pairs and are excluded. Nonempty queries without a relevant candidate would score zero; the retained lists contain relevant candidates. FP32 scoring at 128 tokens produced nDCG@10 **0.9584659586904473** and MRR@10 **0.9437315230482864**. Such small constructed lists are much easier than full retrieval. Do not compare this score directly with published leaderboard scores.

**Constructed validation diagnostic:** a deterministic sample of the lowest 512 validation query hashes with both labels, capped at 128 tokens, yielded normal-query nDCG@10 **0.9589121014718225** and empty-query nDCG@10 **0.9192658835937564**. High scores survive removing queries, limiting what the constructed-list result establishes. The complete original validation (45,709 queries) yielded nDCG@10 0.9576079311960256 and MRR@10 0.9425714848279332; this differs from the test split above.

E15 quality results and an independent saved-score audit agree on metrics and checkpoint identity. E15 used NVIDIA L40S, CPU tokenization, PyTorch 2.14.1+cu130, FP32 with TF32 disabled, batch 128 and disabled encoder fastpath. E16 diagnostic metadata identifies the original checkpoint; **later E16 adaptation results are not results of this release**. Aggregate evaluation evidence is included in evaluation.json without training examples.

## Usage

Tested Python 3.12.13 with torch 2.14.1, transformers 5.18.0, safetensors 0.8.0, huggingface_hub 1.33.0 and tokenizers @TOKENIZERS_VERSION@. Install `requirements.txt` in a fresh environment.

The interface is `snapshot_download` plus the local PyTorch loader. Download an immutable revision, inspect `reranker.py`, and import it from that snapshot. No `trust_remote_code` argument is needed; importing local Python still executes that inspected code. This release implements no AutoModel, Sentence Transformers CrossEncoder, transformers pipeline, hosted inference, or MLX interface.

```python
from huggingface_hub import HfApi, snapshot_download
repo_id = '@REPO_ID@'
revision = HfApi().model_info(repo_id).sha  # resolve once; record this immutable commit
snapshot = snapshot_download(repo_id=repo_id, revision=revision)
```

For an unpublished local stage, set `snapshot` to its directory. Then execute:

<!-- local-example -->
```python
import sys
sys.path.insert(0, str(snapshot))
from reranker import Reranker

reranker = Reranker(snapshot, device='cpu')
result = reranker.rank('What is a reranker?', [
    'A reranker assigns relevance scores to query-document pairs.',
    'Tomatoes grow best in sunny gardens.',
    'A ranking model sorts candidate documents by a raw score.',
], batch_size=2)
print(result['scores'])  # raw logits in original document order
print(result['ranking'])  # descending score, then original index for exact ties
```
<!-- local-example -->

CPU FP32 is the tested portable starting point. `device='cuda'` or `device='mps'` must be selected explicitly and raises when unavailable. MPS CPU fallback must be disabled. GPU parity/performance must be checked separately on the target system; the CPU smoke test makes no GPU latency claim. Batch shape and backend can affect near-tie ordering. `rank` uses `eval()` and `torch.inference_mode()` and checks finite scalar scores. `example.py` and `verify_release.py` run offline from the downloaded snapshot.

## Reproducibility and related work

Weights-only safetensors preserve every checkpoint tensor exactly, excluding optimizer, scaler, RNG and resume payloads. release_manifest.json pins source, checkpoint, weights and tokenizer hashes. Synthetic validation fixtures include short/long inputs, truncation, Unicode normalization, empty texts, different batch sizes and padding. CPU FP32 source comparison tolerance: **atol=1e-5, rtol=0**. Exact token/mask/segment equality and maximum score error are reported in the local release report. Correctness checks establish implementation agreement, not retrieval quality.

The configured source GitHub repository could not be accessed publicly at release preparation (unauthenticated API returned 404); therefore no public source/report URL is asserted. This release includes sufficient inference code, configuration, synthetic verification and aggregate evaluation evidence. Full training harnesses, kernel/serving benchmark sources, datasets and large per-query audits are not included here. Source paths/hashes identify local material but are not downloadable links.

E17 investigated Apple M5/MPS/MLX and NVIDIA L40S CUDA/Triton implementations against this same weak model. Numerical tolerances can permit ranking changes on nearly tied scores; a numerical pass need not imply an application-ranking pass. No hardware performance figures are claimed by this package. Local E17 benchmark reports distinguish FP32 precision, prepared-forward versus complete scoring/HTTP scope, fixed candidate workloads, four Apple or five NVIDIA session rotations, and limited exploratory uncertainty. Those measurements do not establish universal kernel/coordinator speedups or production saturation behavior.

## Citation / authorship

Author: **Nandana Dileep**, verified from local Git authorship@ACCOUNT_ATTRIBUTION@. No paper, affiliation or publication venue is asserted. Cite this artifact by repository ID and the immutable downloaded revision:

```bibtex
@misc{tiny_reranker_experimental_2026,
  author = {Nandana Dileep},
  title = {tiny-reranker-4.7m-experimental: Original Scratch-Trained Epoch 3},
  year = {2026},
  howpublished = {Hugging Face model artifact; cite the immutable release revision}
}
```

## License and attribution

Weights/documentation: **CC BY-NC 4.0** for rights controlled by the author; inference code: **MIT**; reused tokenizer: **Apache 2.0**. See LICENSE, the component-specific full license texts, and NOTICE. The pinned dataset card has no license field; [MS MARCO terms](https://github.com/microsoft/msmarco/blob/master/README.md#terms-and-conditions) specify non-commercial research and grant no additional rights in underlying passages. No training examples are redistributed. The weights license does not relicense those passages, grant commercial data rights, or claim legal certainty about downstream uses.
