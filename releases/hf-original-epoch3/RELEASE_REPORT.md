# Original epoch-3 Hugging Face release report

Status (2026-10-10): **published publicly and remote revision verified**. Authenticated personal account: ndileep (profile name Nandana Dileep). The intended model repository did not exist before this publication; no unrelated repository was overwritten.

- Public model: [ndileep/tiny-reranker-4.7m-experimental](https://huggingface.co/ndileep/tiny-reranker-4.7m-experimental).
- Final published revision: `123070b853693f492ac0313e406fa1bbbbc42f52`.
- Immutable artifact revision pinned in the README: `96d872750fd1bc49395a6051078926ab783fbae3`. The final commit updates README/manifest; inference weights and implementation are identical to that artifact commit.
- Publication identity, complete published file list and SHA-256 values: [publication.json](publication.json).
- Downloaded final revision in a fresh directory: `/var/folders/9d/l9xvxbb94fs4fmj3wb0lw3100000gn/T/hf-downloaded-exact-revision-2wtgp28j`. Every release file hash matches the staged artifact; only Hub-maintained `.gitattributes` is additional.
- Fresh-download CPU parity/inference: [remote_validation.json](remote_validation.json), passed. Preupload check: [preupload_validation.json](preupload_validation.json), passed. Public page HTTP and unauthenticated API checks passed; privacy=false, correct PyTorch/license metadata, expected file list. Both remote README code blocks and standalone example/verification commands passed using the cached pinned artifact snapshot.

## Artifact identity

- Source checkpoint: `runs/full_train/epoch_0003.pt`.
- SHA-256: `2ea2608a9ec98083b89606d13bed1d7c6e7a90dc6160bbe44732166137538d52` (matches requested identity and saved metadata).
- Distinct later E16 adaptation `runs/e16_retrieval/epoch_0003.pt` was not read for export.
- 57 FP32 tensors, 4,717,185 parameters. Exact tensor names, shapes, dtypes and values verified against source after safetensors export.
- Staging directory: `/private/tmp/tiny-reranker-hf-original-e3-ready` (17 release files, no resume state, examples from datasets, caches or credentials).
- Exported weight and all staged-file SHA-256 values: [staged_checksums.json](staged_checksums.json).
- Source/config/checkpoint/tokenizer hashes, tensor inventory and tested versions: [source_manifest.json](source_manifest.json). Its `source_git_head` is release-tooling commit `b50c56a`; individual inspected source hashes also capture preexisting worktree changes.

Staged files: LICENSE, LICENSE-CODE, LICENSE-TOKENIZER, LICENSE-WEIGHTS, NOTICE, README.md, config.json, evaluation.json, example.py, model.safetensors, release_manifest.json, requirements.txt, reranker.py, tokenizer/tokenizer.json, tokenizer/tokenizer_config.json, validation_fixtures.json, verify_release.py. These are the publication allowlist. Hugging Face may additionally maintain its `.gitattributes`.

## Validation

[local_validation.json](local_validation.json): passed in a fresh Python 3.12.13 virtual environment, working directory `/private/tmp`, PYTHONPATH unset, Hugging Face/Transformers offline, fresh empty cache. Only staged artifacts are imported. No original checkpoint, repository imports, unpublished training inputs, CUDA, Triton or MLX are required for CPU inference.

- CPU FP32 versus original source: maximum absolute score difference **0.0**, predeclared `atol=1e-5`, `rtol=0`.
- Variable batch sizes 1/2/3/8: maximum difference **1.1920928955078125e-7** under the same tolerance.
- Eight fixed synthetic cases / 30 pair scores include short/long documents, long queries, 128-token truncation, empty strings, Unicode normalization, variable lengths and dynamic/fixed padding.
- Token IDs, masks and segments exactly match original implementation; explicit `[CLS] hello [SEP] world [SEP]` and segment-order check passes.
- Repeated ranking, original-index tie-breaking, masked-padding invariance, finite scalar output and empty document list pass.
- Exact local README Python example and standalone example.py execute successfully.
- Official Hub `ModelCard.validate()` passes; metadata uses `library_name: pytorch`, `license: cc-by-nc-4.0`, dataset identifier and experimental tags. No `base_model`, pipeline or unsupported library interface is asserted.

The demo honestly ranks the unrelated tomato passage first (score 0.6266350746), followed by the ranking-model passage (0.3510112166) and reranker explanation (-0.5332754850). These are uncalibrated raw logits; success means the implementation runs, not that it ranks well.

Exact installed validation packages are recorded in [validation_environment.txt](validation_environment.txt). Direct release requirements pin torch 2.14.1, transformers 5.18.0, safetensors 0.8.0, huggingface_hub 1.33.0 and tokenizers 0.23.2.

Commands executed (from project root unless stated otherwise):

```bash
.venv/bin/python -m scripts.hf_release --output /private/tmp/tiny-reranker-hf-original-e3-ready
uv venv --python .venv/bin/python /private/tmp/tiny-reranker-hf-clean-env
uv pip install --python /private/tmp/tiny-reranker-hf-clean-env/bin/python -r /private/tmp/tiny-reranker-hf-original-e3-ready/requirements.txt
```

Offline validation executed with working directory `/private/tmp`:

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/private/tmp/tiny-reranker-empty-hf-cache /private/tmp/tiny-reranker-hf-clean-env/bin/python -B /private/tmp/tiny-reranker-hf-original-e3-ready/verify_release.py /private/tmp/tiny-reranker-hf-original-e3-ready /Users/nandanadileep/projects/tiny-llm-macbench/tiny-reranker/releases/hf-original-epoch3/local_validation.json
```

## Licensing and provenance decisions

The user authorized choosing a suitable license. Weights/documentation: CC BY-NC 4.0 for rights controlled by author Nandana Dileep; code: existing project MIT, copyright 2026 ver-labs; tokenizer: Apache 2.0. Full texts and separate attribution accompany the package. This decision does not assert that code licensing determines weight licensing, relicense training passages, or establish certainty for downstream use.

Tokenizer source: [cross-encoder/ms-marco-MiniLM-L6-v2 revision 233902d25c440f23af6f7d6e94d2946bac0bee0a](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a). Its pinned card declares Apache 2.0; pinned Hub file listing contains no LICENSE/NOTICE file. tokenizer.json SHA-256 `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` matches downloaded upstream exactly. The saved project tokenizer_config.json and tokenizer.json are copied byte-for-byte and validated against the original bundle manifest. Original full_train uses that bundle through P1 setup; all model weights originated in E1 random initialization.

Training source: [sentence-transformers/msmarco-bm25, triplet/train, revision ce8a493a65af5e872c3c92f72a89e2e99e175f02](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25/tree/ce8a493a65af5e872c3c92f72a89e2e99e175f02). Pinned dataset card has no license field. [MS MARCO upstream terms](https://github.com/microsoft/msmarco/blob/master/README.md#terms-and-conditions) specify non-commercial research without granting additional intellectual-property rights. Release is a non-commercial research artifact without redistributed training examples. Rights to third-party passages and commercial use are not granted by the weights license. No permission to distribute the original dataset is asserted.

Configured Git remote `https://github.com/ver-labs/tiny-reranker.git` returned an unauthenticated API 404. No public GitHub/report URL is claimed. Sufficient local inference code and synthetic validation are included; full training and kernel/serving harnesses, raw score audits and datasets are unavailable through this release. Local source author is Nandana Dileep (Git authorship); The authenticated Hugging Face profile confirms Nandana Dileep (ndileep).

## Quality limitations

Original constructed test: 45,665 queries / 91,337 pairs, mostly two candidates, nDCG@10 0.9584659586904473 and MRR@10 0.9437315230482864. This is not leaderboard comparable.

Realistic fixed BM25 top-100 test at 128 tokens: scratch nDCG@10 0.013375328821900047 (SciFact, 300 queries) and 0.05362354030597277 (NFCorpus, 323 queries); size-matched pretrained TinyBERT at the same cap yields 0.5703154525374805 and 0.3070801920238763. Training histories differ. Independent E15 audit and original-checkpoint hashes agree.

Sampled original validation diagnostic: 512 queries; normal nDCG@10 0.9589121014718225 versus empty-query 0.9192658835937564. This suggests substantial passage preferences on constructed lists; exact cause of poor generalization is not established. Later E16 adaptation results are not attributed to this checkpoint. E17 implementation-accuracy checks and hardware-specific exploratory timings do not establish retrieval quality or a universal speedup. No performance numbers are claimed by this release.

Intended use: education, inference-performance research, numerical parity and ranking-sensitivity investigations. General retrieval usefulness is not demonstrated; small size alone does not guarantee device suitability or latency.

## Publication and verification commands

Authentication used supported local OAuth device login (credentials were never included in script arguments, logs or release files):

```bash
.venv/bin/hf auth login
```

Publication and full remote verification were executed with:

```bash
.venv/bin/python -m scripts.hf_publish_release --stage /private/tmp/tiny-reranker-hf-original-e3-ready --report-dir releases/hf-original-epoch3 --python /private/tmp/tiny-reranker-hf-clean-env/bin/python
```

It verifies the authenticated user, uses the personal namespace, checks existing-repository identity, revalidates locally, publishes only the allowlist, pins the immutable artifact commit in a second documentation commit, records revision/checksums, downloads that exact final revision to a fresh directory, verifies all files, runs offline CPU smoke/parity tests and the remote README download example, and checks unauthenticated public page/API visibility. A failure after upload retains publication.json with remote_verified=false; completion must not be claimed until verification succeeds.

Local release-tooling commits: `b50c56a` and `9b31bcb` (virtualenv/snapshot path verification fixes). Initial preparation documentation commit: `302d283`. Final publication documentation is a separate logical commit; use `git log -- releases/hf-original-epoch3` for its hash. No Git push was performed. Preexisting README/.gitignore changes, E16/E18 files and parent-workspace changes were preserved.

## Publication-path findings

Preupload verification caught virtualenv interpreter symlink resolution selecting the base interpreter; the publisher now preserves the supplied environment path. Initial remote standalone-example testing caught Hugging Face cached script symlinks resolving to the blob directory, making sibling imports fail. The documented README import worked; standalone entry points were corrected to import from the supplied snapshot path. Both findings were investigated and corrected without widening numerical tolerance or changing model weights. Initial remote commits d830a61e960023e3e575009babd1dc3382ea693b and 80d6bd852f673c09caceab3c25d751967a77d922 are superseded and were not reported as fully verified. The final published revision and its pinned artifact revision both pass all checks.

No remaining publication blockers. Code and tokenization remain CPU-portable; GPU device selection is explicit, but GPU behavior/performance is not validated by these release smoke tests. The release remains experimental with weak demonstrated retrieval quality.
