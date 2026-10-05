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

Run `python -m scripts.clean_pairs` to turn the pinned triplets into labeled
query-document pairs. It checks exact source query and document text, writes
each consistently labeled pair once to `data/processed/clean_pairs.jsonl`, and
flags pairs seen with both labels in `data/processed/conflicting_pairs.jsonl`.
Conflicting pairs are excluded from the clean file. The audit found 918,054
clean pairs and 43,904 conflicting pairs; `results/pair_cleaning.json` records
the counts. The processed JSONL files are excluded from Git.

`CleanPairDataset("train")` in `scripts/data.py` indexes the cleaned file by
split and returns one query, document, integer label, query ID, document ID,
and pair ID per item. Use `"validation"` or `"test"` for those groups. It
indexes file offsets, so passage text is read only when an item is requested.
`iter_clean_pairs("train")` is a generator that reads that same file one line
at a time; use `"validation"` or `"test"` to stream those groups. It returns
the same example fields without building an index.
`PairCollator` in `scripts/collate.py` tokenizes a batch from that dataset,
pads only to the longest pair in the batch (up to the configured maximum),
and retains labels and IDs. Run `python -m scripts.collate` for a two-pair
padding and alignment check. `make_pair_loader(...)` uses `num_workers=0`
while debugging the data pipeline.
Run `python -m scripts.print_batch` to print every field of a small batch
made from real cleaned training pairs.

Run `python -m scripts.build_eval_candidates` to create fixed validation and
test candidate lists from the clean pairs. The script hashes each passage into
a document ID, keeps one copy per query, and includes queries with both a
positive and a negative candidate. Both models should read these same saved
lists. The build produced 45,709 validation queries and 45,665 test queries;
`results/evaluation_candidates.json` records counts and file hashes. This is a
constructed MS MARCO triplet evaluation, not a published leaderboard workload.
The external baseline was previously trained on MS MARCO, so these queries are
not established as unseen for it. The scratch model must train only on training
queries; the archived all-data checkpoint is not suitable for held-out testing.

Run `python -m scripts.evaluate_initial` for the first validation comparison.
It saves 128 fixed validation query IDs, per-query rankings, MRR@10 and
nDCG@10, plus warm-up and repeated whole-rerank timing in
`results/initial_validation/`. It records model and tokenizer loading as
startup time outside the timed rerank calls, plus a separate forward-pass
measurement on pretokenized inputs. MiniLM scored 0.8906 MRR@10 and 0.9193
nDCG@10; the untrained scratch model scored 0.7891 and 0.8443. Both used
MPS, float32, a 128-token maximum, and pair batches of 2. The random model
is a pipeline sanity check, not a competitive baseline. These sampled,
constructed-list scores are not comparable to published leaderboard results;
the test split remains reserved for the final comparison.

Run `python -m scripts.evaluate_validation` to score every saved validation
candidate list with both models. It saves one ranked list and MRR@10/nDCG@10
values per query in `results/full_validation/`, sorting tied scores by
document ID. Across all 45,709 evaluable validation queries, MiniLM scored
0.8599 MRR@10 and 0.8966 nDCG@10; the random scratch reference scored
0.7576 and 0.8210. Of 50,189 source validation query IDs, 4,480 had no
clean candidates after conflicting labels were removed and could not be
ranked; they are excluded from the metric denominator and reported explicitly.
Any nonempty candidate list with no relevant document remains in evaluation
and receives zero for both metrics; there were none in this dataset. The test
split has not been scored.

Run `python -m scripts.train` for the first training-loop check. It restores
the saved initial random weights, selects MPS when available, and calls
`model.train()` to enable training behavior such as dropout. This check does
read one dynamically padded batch of 16 cleaned training pairs with
`num_workers=0`; it does not run a forward pass, calculate gradients, or
update weights. It moves the numeric inputs and labels to the selected device
while leaving string IDs in Python.
