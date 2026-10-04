# tiny-reranker

## Initial comparison environment

- Mac: MacBook Air (Mac17,3), Apple M5
- Unified memory: 24 GB
- macOS: 26.5.2 (build 25F84)
- Python: 3.12.13 in `.venv`
- Selected PyTorch backend: MPS (`torch.backends.mps.is_available()` is `True`)
- Package versions: see `requirements.txt`

Activate `.venv` before running the baseline or scratch model so the initial
comparison uses the same Python packages and backend.

## Published baseline

Run `python -m scripts.baselines` to score the three example documents. The script loads
the tokenizer and pretrained weights for
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The first run downloads
the model files. It uses MPS when available and otherwise uses the CPU.

`BaselineReranker.rank(query, documents)` returns original document IDs with raw
relevance scores, sorted highest first. These scores are ranking values, not
probabilities. The baseline's pretrained weights are not used to train the scratch
model.

## Scratch model configuration

`configs/scratch.json` records the starting architecture. Its vocabulary size
and padding ID come from the pinned tokenizer above. The 128-token limit is an
initial workload choice; the tokenizer itself supports up to 512 tokens. Four
attention heads divide the 128-dimensional hidden state into 32 dimensions per
head. The feed-forward layer expands that hidden state to 512 dimensions.

These values describe a model whose weights will be initialized from scratch.
Run `python -m scripts.scratch_tokenizer` to load and validate only the tokenizer assets.
The scratch path does not load the baseline model or its weights.
`encode_pairs()` adds `[CLS] query [SEP] document [SEP]`, segment IDs, and an
attention mask when preparing batches for the scratch model.

Run `python -m scripts.model` for a two-pair forward-pass smoke test. `model.py` creates
random token, position, and segment embeddings; four separate bidirectional
encoder layers; normalization layers; and a one-score head. It masks padding and
uses MPS when available. Its scores are random until the model is trained.

## Initial random checkpoint

Run `python -m scripts.save_initial` to create and verify the seed-42 state before
training. The weights are in `checkpoints/initial_seed42.pt`; the tracked
`checkpoints/initial_seed42.json` records the seed, config, PyTorch version,
parameter count, and checkpoint hash. The weight file is excluded from Git.
Running the command again checks the existing checkpoint without replacing it.

## Dataset preparation

`data/manifests/msmarco_bm25_triplet.json` pins the selected dataset and its
terms. `scripts/data.py` streams its full `triplet` split and assigns whole queries to
training, validation, or test by a stable query hash (80/10/10 buckets). The
split rule is recorded in `data/manifests/query_split.json`. It assigns every
triplet for the same normalized query to one group. The previously run training
scripts, trained checkpoints, and run results are archived in `../dump/`.
