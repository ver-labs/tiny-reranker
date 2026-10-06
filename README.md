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

Run `python -m scripts.short_epoch` for one bounded epoch over the first
1,024 cleaned training pairs. Each pair is used once in a batch of 16, giving
64 optimizer steps. The script compares binary loss and accuracy on 128 fixed
validation pairs before and after, checks finite scores, losses, and gradients,
and saves a trial checkpoint and `results/short_epoch_1024.json` with sample
ID hashes. It never reads the test split. These pair-level diagnostics are
not the query-ranking metrics used for the final comparison.
On MPS, the completed run had a mean training loss of 0.6951. Validation
BCE loss changed from 0.7057 to 0.6863 and binary accuracy from 50.0% to
56.2%. All 1,024 selected pair IDs were used exactly once. The checkpoint
hash was verified after saving, and all 57 model state tensors differed from
the initial random state.

Run `python -m scripts.evaluate_short_epoch` to evaluate that saved trial
checkpoint on every fixed validation candidate list using the same 128-token
limit, 64-pair scoring batches, document-ID tie-break, MRR@10, and nDCG@10 as
the earlier full validation comparison. It checks the checkpoint and candidate
file hashes, saves per-query rankings and a summary in `results/full_validation/`,
and does not score the test split.
On all 45,709 evaluable validation queries, the short-epoch model scored
0.8109 MRR@10 and 0.8604 nDCG@10. The untrained scratch reference scored
0.7576 and 0.8210 on the identical candidate lists. These are constructed
MS MARCO mined-triplet lists, so the result is a local learning check rather
than a leaderboard score. The saved checkpoint is
`checkpoints/short_epoch_1024.pt`; its hash is recorded in both the training
report and the full-validation summary.

Run `python -m scripts.inference_bundle` to export the short-epoch model to
`artifacts/short_epoch_1024/`. The directory holds the model weights, scratch
architecture config, local tokenizer assets, and a manifest with file hashes.
The exporter reloads only those local files and checks that token IDs and
inference scores match the source checkpoint. The binary weight file is kept
locally and excluded from Git, like the training checkpoints.

For training continuation, run `python -m scripts.short_epoch --save-resume`.
This repeats the one-epoch run and saves a separate
`checkpoints/short_epoch_1024_resume.pt` with
optimizer moments, completed epoch and update counts, model and training
configuration, and Python, NumPy, CPU PyTorch, and MPS RNG states. The
manifest records whether its model weights exactly match the earlier evaluated
checkpoint. This MPS reproduction had the same displayed pair-level metrics
but its weights differed by up to 0.0001439, so the resume checkpoint is a
distinct run. The full validation ranking scores above belong to the earlier
checkpoint and inference bundle. `load_resume_state()` checks the sidecar hash
and restores its model, optimizer, counters, and RNG states. A dry reload on
MPS recovered 57 optimizer parameter states. The inference bundle does not
contain optimizer or RNG state.
When a scheduler or gradient scaler is added to training later, pass it to
`save_resume_sidecar(...)`; its state will be saved with the other training
state. `load_resume_state(...)` requires a factory for each saved optional
component and restores its state. Neither component is enabled in the current
run.
The resume checkpoint also pins `data/manifests/query_split.json` by SHA-256
and records the sequential sampler at an epoch boundary: all 1,024 selected
pairs consumed, with the next epoch starting at index zero. Mid-epoch resume
is not implemented; a future shuffled or interruptible loop would need to
save its actual order and next index at each checkpoint.

Run `python -m scripts.continue_epochs --epochs 1` to continue from the latest
resume checkpoint on the same fixed 1,024-pair training sample. The script
checks that every selected pair was used once, evaluates the fixed 128-pair
validation sample, then saves a new numbered checkpoint only at the epoch
boundary. It records the parent checkpoint, cumulative update count, split
manifest hash, and sampler position. Repeating the command continues from
the latest saved epoch. It does not implement mid-epoch checkpoints.
The demonstration continued the separate resume-ready run through epoch 2:
128 cumulative updates, with the new checkpoint at
`checkpoints/short_epoch_1024_epoch_0002.pt`. It reloaded on MPS with all 57
optimizer parameter states. Its 128-pair validation loss rose from 0.6863 to
0.7426 and accuracy changed from 56.2% to 55.5%, so it is a resume example,
not an improved model or a replacement for the earlier full-validation result.
To verify stop and resume, a later process loaded the epoch-2 checkpoint and
confirmed AdamW already held 57 parameter states at step 128, with nonzero
scoring-head momentum. It trained one more complete epoch and saved epoch 3
at update 192. Reloading that checkpoint confirmed AdamW step 192, so the
optimizer and counters continued instead of resetting. Its 128-pair
validation loss was 0.6200 with 61.7% binary accuracy; this is a small
pair-level check, not a full ranking evaluation of the epoch-3 model.
Checkpoint writes now use a temporary file in the destination directory.
The temporary file is loaded successfully before it is renamed to the final
checkpoint path; reports and manifests are written atomically too. Existing
epoch files are never overwritten by the continuation script.

E1 model-size setup lives in `experiments/E1/`. Run
`python -m experiments.E1.setup_models` to prepare independently initialized
2-layer and 4-layer scratch encoders. Their configuration files differ only in
encoder depth. Each folder records the seed, parameter count, config and
initial-weight hashes, and a finite dummy forward pass. No E1 training is
performed by the setup command.
The 2-layer model has 4,320,641 trainable parameters and the 4-layer model
has 4,717,185; all parameters are trainable in both initial models.
Run `python -m experiments.E1.train_variant --variant two_layer` to train only
the 2-layer E1 model on the fixed 1,024 training pairs for one epoch. The
script saves its checkpoint and a report with 128-pair validation loss,
training time, and sampled MPS memory. The 4-layer E1 model remains at its
saved random initialization until its separate run.
Run `python -m experiments.E1.evaluate_variant --variant two_layer` to score
that checkpoint on every fixed validation candidate list, saving per-query
rankings, MRR@10, nDCG@10, and a whole-evaluation timing in its E1 folder.
The 2-layer run completed 64 optimizer updates on MPS in 2.69 seconds for the
timed training loop. It scored 0.7978 MRR@10 and 0.8507 nDCG@10 on all
45,709 fixed validation candidate lists; the whole validation evaluation took
33.30 seconds after warm-up. Training memory is recorded as sampled MPS
allocation, not a true peak.
After training and evaluating both variants, run
`python -m experiments.E1.compare` to check matched data and settings and
write `experiments/E1/comparison.json`. On the first E1 run, the 4-layer
model scored 0.8109 MRR@10 and 0.8604 nDCG@10 versus 0.7978 and 0.8507
for 2 layers. Its timed training loop took 4.04 seconds versus 2.69 seconds;
full validation took 44.58 versus 33.30 seconds. This is one seed and one
small epoch, so the quality gap is an early observation, not a final finding.

The separate head-freezing exercise starts from each trained E1 checkpoint,
freezes the embeddings and encoder, and updates only the 129-parameter scoring
head on 1,024 new training pairs whose queries were absent from the original
E1 training sample. For each variant, run these commands (replace `two_layer`
with `four_layer` for the other model):

```sh
.venv/bin/python -m experiments.E1.freeze_backbone --variant two_layer
.venv/bin/python -m experiments.E1.evaluate_head_only --variant two_layer
```

Each run verifies that every backbone tensor stayed unchanged and saves a
separate checkpoint and validation ranking; the original E1 checkpoints remain
untouched.

On the same 45,709 constructed validation queries, the 2-layer model changed
from 0.7978 to 0.7993 MRR@10 and from 0.8507 to 0.8518 nDCG@10. The 4-layer
model changed from 0.8109 to 0.8115 MRR@10 and from 0.8604 to 0.8609
nDCG@10. Both used the same additional 1,024 training pairs. These small
single-run gains show what happened in this exercise, not a general advantage
of head-only training.

### E2: data loader workers

The first E2 run uses the 4-layer model with `num_workers=0`: the main Python
process tokenizes each batch. It starts from the saved E1 random initialization
and uses the same 1,024 training pairs, batch size 16, seed 123, and MPS backend.
Run it with `.venv/bin/python -m experiments.E2.train_workers --num-workers 0`.
Its report is `experiments/E2/workers_0.json`; its separate checkpoint is
`experiments/E2/workers_0.pt`.

This run processed 1,024 pairs in 5.06 seconds (202.3 pairs/second). Fetching
batches took 0.21 seconds total; loader startup took less than 0.01 seconds.
Validation BCE was 0.7057 before training and 0.6863 after. E2 synchronizes
MPS after each update so batch-wait and compute times can be separated. The
two-worker run must use that same timing method before comparing throughput.

The matching two-worker run used the same starting checkpoint, pair IDs,
settings, and 64 updates. It took 11.77 seconds (87.0 pairs/second), including
6.49 seconds to start the workers and 0.26 seconds waiting for batches. The
zero-worker run took 5.06 seconds; for this small epoch, extra workers slowed
training. The two final checkpoints agreed within a maximum absolute weight
difference of 0.000107, and both had validation BCE 0.6863. The detailed
comparison is in `experiments/E2/comparison.json`. Main-process RSS does not
include worker memory, so it is not a total-memory comparison.

### E3: fixed versus dynamic padding

The fixed-padding run uses the 4-layer E1 random initialization, 1,024 pairs,
batch size 16, `num_workers=0`, and MPS. Every pair is padded to 128 tokens.
Run `.venv/bin/python -m experiments.E3.train_padding --padding fixed`.
It used 46,022 padding tokens out of 131,072 token slots and took 3.12 seconds
for the synchronized 64-batch training loop. Its report is
`experiments/E3/fixed_padding.json`. Run the matching dynamic case with
`.venv/bin/python -m experiments.E3.train_padding --padding dynamic`.

Both runs used identical unpadded tokens and the same 128-token truncation
limit. Fixed padding used 46,022 padding tokens and took 3.12 seconds; dynamic
padding used 44,438 padding tokens and took 3.33 seconds. Dynamic padding saved
1,584 token slots (1.2% of fixed slots) but was 0.21 seconds slower in this
single short run. See `experiments/E3/comparison.json`. Training uses dropout,
so different tensor shapes can change its random-mask use; the validation-loss
difference is not a controlled test of model quality.

Both saved checkpoints were also ranked on the same 45,709 constructed
validation query candidate lists, using the same dynamic-padding inference
routine. Fixed-padding training scored 0.8075 MRR@10 and 0.8579 nDCG@10;
dynamic-padding training scored 0.8109 and 0.8604. Their full per-query
rankings and summaries are saved in `experiments/E3/`. To reproduce them, run
`.venv/bin/python -m experiments.E3.evaluate_padding --padding fixed` and then
the same command with `dynamic`. The ranking difference
can reflect changed dropout masks during training; it does not establish that
one padding method trains a better model.
