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
`num_workers=0`; it runs a full-batch forward pass and a one-item shape check,
then calculates `BCEWithLogitsLoss` on the raw scores and float labels. It
calls `loss.backward()`, checks every trainable parameter gradient for finite
values, prints representative gradient norms, and performs one AdamW update
at learning rate `3e-4`. Numeric inputs and labels move to
the selected device while string IDs stay in Python.
After the successful update it logs `step=1` and that batch's BCE loss.
It prints every pair's label, raw score, and sigmoid probability so the
forward output can be inspected alongside the backward gradient norms.
The script calls `optimizer.zero_grad(set_to_none=True)` before the forward pass,
so no old parameter gradients carry into this backward pass.
The logit-shape check accepts one score per label and verifies that a
one-item batch retains shape `(1,)` rather than becoming a scalar.
After the update, the script checks a fixed batch of 16 validation pairs in
evaluation mode with gradient tracking disabled, logs its BCE loss, then
restores training mode. `EVAL_EVERY_STEPS` is 1 for this one-step smoke check;
a longer training loop can increase the interval. This batch loss is a pipeline
check, not a full validation metric. The update is not saved as a trained
checkpoint, and the test split remains untouched.

Run `python -m scripts.tiny_fit` to repeat training on the same 32 cleaned
training pairs, using alternating batches of 16. Every 10 optimizer steps it
measures loss and binary accuracy on those 32 pairs with dropout off and
gradients disabled. It stops early once accuracy is 100% and loss is below
0.1, or after 100 steps. This is a memorization check of the training code,
not a measure of performance on new queries. On MPS, the fixed sample went
from 50% accuracy and 0.7091 loss to 100% accuracy and 0.0091 loss after
50 steps. No diagnostic weights are saved.
The script also checks every training score, loss, and gradient for finite
values, counts mistaken predictions at each report, and prints up to two
examples so errors can be inspected before changing hyperparameters.
On the MPS run, all checked values were finite and mistakes fell from 16 to
zero by step 50. One interim mistake was a mined negative passage about the
Dodd-Frank Act for a query asking which act requires conflict-mineral
disclosure; the passage itself appears relevant despite its negative label.
This suggests a label-quality issue in that example, not evidence that the
learning rate needs changing. Hyperparameters were left unchanged.
The tiny-fit weights existed only in that process. For the real experiment,
`load_training_model()` reloads `checkpoints/initial_seed42.pt`. After the
diagnostic, all 57 loaded state tensors were checked on MPS and exactly
matched the saved random initialization; the checkpoint hash remained
`a8f10b4e4ae63ef1a4b796ab4122575c5f72afbb2a7a7a51ef49a7548c3602ee`.
