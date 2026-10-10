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

### E4: inference pair batching

Use the trained E1 4-layer checkpoint and the first 128 fixed validation
candidate lists (256 pairs). Each list has two candidates, so the benchmark
groups eight queries and scores pairs across that group in batches of 1, 4,
or 8. Query latency starts when its eight-query group starts and stops when
that query's sorted ranking is CPU-readable. The model, token limit, input
order, and candidates stay fixed. Each size gets ten matching-size warm-up
calls, one untimed pass over the full sample, then three timed repeats:

```sh
.venv/bin/python -m experiments.E4.benchmark --pair-batch-size 1
.venv/bin/python -m experiments.E4.benchmark --pair-batch-size 4
.venv/bin/python -m experiments.E4.benchmark --pair-batch-size 8
.venv/bin/python -m experiments.E4.compare
```

| Pair batch size | Median pairs/second | Median query latency | Timed repeats (seconds) |
| --- | ---: | ---: | --- |
| 1 | 275.5 | 32.00 ms | 1.031, 0.929, 0.927 |
| 4 | 717.0 | 14.33 ms | 0.333, 0.357, 0.372 |
| 8 | 1,500.6 | 7.75 ms | 0.156, 0.171, 0.294 |

All 128 ranked document orders and CPU-readable scores matched exactly across
sizes. The sampled MPS tensor allocations were 18,973,696, 18,982,912, and
18,995,200 bytes for sizes 1, 4, and 8, respectively. These are sampled
current allocations while input and score tensors are live, not peak memory.
The third size-8 repeat was slower, so keep the variation visible. Results
and the measurement definition are in `experiments/E4/comparison.json`.

### E5: gradient accumulation

The two E5 runs start from the same E1 4-layer random weights and use the same
1,024 pairs in the same order with fixed 128-token padding. Both use an
effective batch of 16 and make 64 AdamW updates. The direct case processes
16 pairs per forward/backward pass; accumulation processes four pairs four
times, weighting each microbatch loss by its share of the effective batch,
then updates once. Dropout is set to zero in both runs to keep different
microbatch shapes from changing the random masks. Run:

```sh
.venv/bin/python -m experiments.E5.train_variant --mode direct
.venv/bin/python -m experiments.E5.train_variant --mode accumulate
.venv/bin/python -m experiments.E5.compare
```

| Mode | Microsteps | Updates | Training time | Sampled MPS tensor allocation | Validation BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, batch 16 | 64 | 64 | 8.95 s | 154,512,384 bytes | 0.6840 |
| Accumulate, 4 × 4 | 256 | 64 | 7.06 s | 99,791,616 bytes | 0.6840 |

The saved model weights differed by at most 0.000158. The sampled tensor
allocation fell about 35%; this is not a true memory peak. Each mode was run
once, so its timing does not establish that accumulation is generally faster.
The script also scales a smaller final group by its actual size; a separate
six-example check confirmed that weighted 4+2 microbatch gradients match a
direct batch of six. Details are in `experiments/E5/comparison.json`.

Both E5 checkpoints were also evaluated on the same 45,709 constructed
validation candidate lists:

| E5 mode | MRR@10 | nDCG@10 |
| --- | ---: | ---: |
| Direct batch of 16 | 0.8075 | 0.8579 |
| Accumulate four microbatches of 4 | 0.8075 | 0.8579 |

Every query had the same document order in both rankings; the largest raw
score difference was 0.000000477. Run the full evaluations with
`.venv/bin/python -m experiments.E5.evaluate_variant --mode direct` and the
same command with `accumulate`. Their per-query rankings and summaries are in
`experiments/E5/`. This constructed validation set is not comparable to a
published leaderboard.

### E6: local precision support check

Before a longer precision comparison, run
`.venv/bin/python -m experiments.E6.check_support`. On this Mac, Python
3.12.13, PyTorch 2.14.1, macOS 26.5.2, and MPS were detected; the MPS CPU
fallback setting was unset. The 4-layer scratch model completed one real
forward pass, BCE backward pass, and AdamW update with finite values in each
of FP32, FP16 autocast, and BF16 autocast. FP16 used an enabled MPS GradScaler;
BF16 did not use scaling. Model parameters and gradients stayed FP32 while
autocast produced FP16 or BF16 logits. The exact local record is in
`experiments/E6/support.json`. This is a four-example compatibility check,
not a full precision training or quality comparison; FP32 remains the reference.

The bounded FP16 training run uses `torch.autocast("mps", dtype=torch.float16)`
with the supported `torch.amp.GradScaler("mps")`. It starts from the same saved
random weights and 1,024 training pairs as the E1 four-layer FP32 run. Run
`.venv/bin/python -m experiments.E6.train_fp16`. In each batch it scales the
loss, backpropagates, unscales gradients, lets the scaler step or skip AdamW,
then updates the scale. It completed 64 of 64 optimizer updates with zero
skips; the scale remained 65,536. The checkpoint saves model, optimizer, and
scaler states. FP32 validation BCE after training was 0.68634 versus 0.68633
for E1 FP32. The 13.02-second FP16 loop samples memory and synchronizes MPS
after every batch, so it is not directly comparable with E1's 4.04-second
timing scope. Full ranking quality and a matched speed test remain separate
steps.

The matched E6 experiment then trains FP32, FP16 autocast with MPS GradScaler,
and BF16 autocast without a scaler from the same saved random weights. All
three use the same 1,024 pairs, batch size 16, 64 updates, dynamic padding,
and timing/memory sampling method. Use
`.venv/bin/python -m experiments.E6.train_matched --precision fp32`, replacing
`fp32` with `fp16` or `bf16` for the other runs. Evaluate with
`.venv/bin/python -m experiments.E6.evaluate_matched --precision fp32`
(and the other precisions), then run
`.venv/bin/python -m experiments.E6.compare_matched`.

| Training compute | Epoch time | Sampled MPS tensor allocation | MRR@10 | nDCG@10 | Changed query rankings vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 4.40 s | 214,281,216 bytes | 0.8109 | 0.8604 | 0 |
| FP16 autocast | 4.17 s | 168,804,352 bytes | 0.8109 | 0.8604 | 4 |
| BF16 autocast | 4.78 s | 168,803,840 bytes | 0.8109 | 0.8604 | 26 |

All modes completed 64 updates without skips or nonfinite values. Full
ranking evaluation used FP32 inference for every checkpoint to isolate the
training-precision change, on the same 45,709 constructed validation queries.
The lower-precision runs used about 45.5 million fewer sampled tensor bytes
(roughly 21%); these are not true memory peaks. Each mode has one short
training-time observation, so the small speed differences are not a stable
performance conclusion. The aggregates matched, though a few individual
query rankings differed. See `experiments/E6/matched_comparison.json`.

A separate E6 control casts **all** model weights to a lower precision and
does not use autocast. Run the one-step check with
`.venv/bin/python -m experiments.E6.probe_full_cast`. On this installation,
FP16 full cast stopped because PyTorch's MPS GradScaler raised
`ValueError: Attempting to unscale FP16 gradients.` No longer FP16 full-cast
run was started. BF16 full cast completed the one-step check and a 1,024-pair
epoch; all 64 updates completed. To reproduce training and full BF16 ranking:

```sh
.venv/bin/python -m experiments.E6.train_matched --precision bf16 --full-cast
.venv/bin/python -m experiments.E6.evaluate_matched --precision bf16 --full-cast
.venv/bin/python -m experiments.E6.compare_full_cast
```

| BF16 method | Model weights | Training time | Sampled MPS tensor allocation | MRR@10 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Autocast | FP32 | 4.78 s | 168,803,840 bytes | 0.8109 | 0.8604 |
| Full cast, no autocast | BF16 | 3.26 s | 135,084,800 bytes | 0.8051 | 0.8562 |

The full-cast control changed 2,453 of 45,709 query rankings relative to
BF16 autocast. Autocast's checkpoint was evaluated in FP32, while the
full-cast checkpoint was evaluated in BF16, so this comparison changes both
training parameter precision and inference precision. The timing is one run
per mode. See `experiments/E6/full_cast_comparison.json`.

### E7: pairwise ranking loss (deferred)

Deferred at the user's request on 2026-10-07. E7 would compare the current
pointwise BCE objective with a pairwise ranking loss using positive and
negative documents from the same query. No E7 training or evaluation has
been run. Revisit it after the other planned experiments if desired.

### E9: selected checkpoint in PyTorch/MPS and MLX

E9 uses `experiments/E1/four_layer/head_only.pt`, the scratch checkpoint with
highest saved full-validation nDCG@10 (0.860881). MiniLM remains the external
pretrained reference. E7 and the remaining E8 evaluation can be revisited later.

Both backends run model inference on the same Apple M5 GPU: PyTorch uses
MPS/Metal, and MLX uses its Metal GPU backend. Tokenization and HTTP handling
run on the CPU for both. This experiment compares two GPU execution frameworks
using the same trained weights and full FP32 precision on the same Mac.

The MLX implementation preserves token/position/segment embeddings, pre-norm
encoder layers, padding masks, exact GELU, normalization epsilon, CLS pooling,
and the scoring head. Conversion explicitly checks every weight name and shape;
Linear weights keep their `[out, in]` layout. Source and converted hashes are
recorded in `experiments/E9/conversion.json`. The local exported tokenizer is
shared by both backends, and parity checks verify identical tokenized inputs.

E9 sets `MLX_ENABLE_TF32=0` before importing MLX. MLX's default matrix path can
use reduced precision despite FP32 arrays; this caused a first-layer error of
about 0.0028 in the diagnostic. Disabling it brought layer errors below
0.000002. See the [MLX numerical precision guide](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
Both the API and benchmarks use this full-FP32 setting.

```sh
# Convert the selected PyTorch checkpoint to local MLX weights.
.venv/bin/python -m experiments.E9.backend
# Check layer outputs, padding, truncation, and all validation rankings.
.venv/bin/python -m experiments.E9.validate
# Run sequentially so the two backends do not compete for the GPU.
.venv/bin/python -m experiments.E9.benchmark --backend pytorch
.venv/bin/python -m experiments.E9.benchmark --backend mlx
# Start, benchmark, and stop isolated local servers automatically.
.venv/bin/python -m experiments.E9.benchmark_serving
.venv/bin/python -m experiments.E9.compare
.venv/bin/python -m pytest experiments/E9/test_serving.py -q
```

GPU commands require access to the Mac's Metal device. The converted
`head_only_mlx.safetensors` stays local and is excluded from Git. Full validation
matched all 45,709 query rankings with a maximum score difference of
0.000000596; PyTorch and MLX both achieved 0.811525 MRR@10 and 0.860881 nDCG@10.
The test split remains reserved. Results are constructed MS MARCO triplet
validation metrics, not leaderboard results.

Direct benchmarking records pretokenized resident-input forward time separately
from complete scoring including tokenization and device-input preparation. It
uses 128 fixed validation queries (256 pairs), batch sizes 1/4/8/16/64, warm-up,
and five timed repetitions. Timing ends only after scores are CPU-readable,
including MLX execution rather than graph construction. Batch latency is not
individual-query latency. Memory counters are sampled framework allocations;
the frameworks expose different counters, so these are not comparable process
memory peaks.

The HTTP API loads one persistent model on a dedicated worker thread and uses a
bounded queue of 32 waiting requests. GPU work stays off the event loop. Both
backends expose the same request and response schema, sort score ties by document
ID, reject invalid inputs, and handle queue saturation (503) and deadlines (504).
Cancelled pending jobs are skipped; running GPU work finishes before the worker
moves on. Tests cover queue bounds, response matching, deadlines, cancellation,
and recovery after an inference exception.

To run either server (one at a time):

```sh
.venv/bin/python -m experiments.E9.serve --backend mlx --port 8099
# Or use --backend pytorch with the same arguments.
curl -sS http://127.0.0.1:8099/health
curl -sS http://127.0.0.1:8099/rerank -H 'Content-Type: application/json' \
  -d '{"request_id":"demo","query":"saving a model","documents":[{"document_id":"a","text":"Save the model state dictionary."},{"document_id":"b","text":"Growing tomatoes in a garden."}],"top_k":2}'
```

Serving benchmarking uses real loopback HTTP and persistent client connections,
with 1/4/8 requests in flight, 128 requests per repetition, and five repetitions
in each of two rounds. Backend order is reversed in round two. The workload has
two candidate documents per request; no batching across requests is implemented.
Reported latency begins after acquiring a client slot and includes HTTP,
server queueing, tokenization, GPU execution, sorting, and JSON. Warm-up and
startup are recorded separately. Every response is checked for request matching
and expected document order; invalid-input and top-k checks hit the real API.
See `experiments/E9/comparison.json` for the resulting comparison and limitations.

On this M5 Mac, the completed E9 full-FP32 run produced:

| Pair batch | PyTorch complete scoring (pairs/s) | MLX complete scoring (pairs/s) |
| --- | ---: | ---: |
| 1 | 360 | 517 |
| 4 | 825 | 993 |
| 8 | 1,199 | 1,077 |
| 16 | 1,733 | 1,319 |
| 64 | 2,210 | 1,573 |

| Concurrent HTTP requests | PyTorch requests/s | MLX requests/s | PyTorch median latency | MLX median latency |
| --- | ---: | ---: | ---: | ---: |
| 1 | 168 | 241 | 5.45 ms | 4.03 ms |
| 4 | 207 | 331 | 18.82 ms | 12.38 ms |
| 8 | 190 | 291 | 39.17 ms | 26.16 ms |

Serving values are medians across ten repeated observations per concurrency
level. All 7,680 measured HTTP requests completed with expected rankings and no
failures. MLX was faster for this two-document serving workload; PyTorch had
higher direct throughput at larger pair batches. Concurrency increases queue
latency because there is one worker and no cross-request batching. These results
do not establish a universal winner for other candidate counts or precision
settings. Full repeat timings, p95 latency, loading time, queue/processing time,
and memory samples are retained in the JSON reports.

### P1: 10K training and profiling before full-data training

P1 trains all 4,717,185 parameters of the four-layer model from the saved E1
seed-42 random initialization. It selects 10,000 cleaned training pairs without
replacement using seed 123 and uses that reproducibly shuffled order for one
epoch. The sample indices, pair IDs, cleaned-file hash, and query-split manifest
are saved in `experiments/P1/sample.json`. Every selected pair is checked to
belong to a training query; validation and test examples are excluded from the
training sample.

The fixed configuration uses PyTorch/MPS on the Apple GPU, direct batches of 16,
dynamic padding up to 128 tokens, pointwise BCE, AdamW at `3e-4`, and dropout 0.1.
FP16 autocast uses MPS GradScaler, while parameters, their stored gradients, and
BCE loss stay FP32. There is no gradient accumulation, activation checkpointing,
backbone freezing, or new loss objective. E6 supported the precision choice with
matching aggregate ranking metrics and about 21% lower sampled tensor memory;
its small timing lead was a single-run observation. E1 favored four layers for
quality. E2 favored zero loader workers for the short workload. E3 supported
saving padding slots, without establishing a dynamic-padding speed advantage.
These choices are combined for this run; their combination is not a proven
optimum. Data preparation runs in the main process.

```sh
.venv/bin/python -m experiments.P1.run
# Recover evaluation after an interrupted post-training step without retraining:
.venv/bin/python -m experiments.P1.run --evaluate-only
```

The main training pass records whole-epoch throughput and five 2,000-pair window
times. It has no operator profiler or per-stage GPU synchronization, but still
includes finite-value checks, scaler checks, scalar loss reads, CPU stage timers,
and memory snapshots. Its saved checkpoint contains model, optimizer, scaler,
RNG, configuration, and epoch-boundary sampler state. Saving reloads the temporary
checkpoint and verifies the final weights, scaler state, and optimizer counters.
The run refuses to overwrite existing training output. Mid-epoch resume and a
full-data training continuation command are not implemented by P1.

After training, a separate fresh diagnostic model warms up for five batches and
replays 50 batches (800 pairs) with MPS synchronization around each stage:
data reading, tokenization/collation, input transfer, gradient clearing,
forward/loss, forward checks, backward/unscale, gradient checks, optimizer/scaler,
and scalar logging. Its breakdown is synchronized wall time, including dispatch
and synchronization overhead, not pure GPU kernel time. It represents an early
training replay from the same sample, not every batch in the full epoch. A
separate ten-batch CPU operator trace identifies CPU-side operations; it does not
claim GPU kernel profiling. Traced and synchronized timing are excluded from
full-training duration estimates. Diagnostic weights are discarded.

The checkpoint from the main pass is reloaded and evaluated using FP32 inference
on all 45,709 fixed constructed validation candidate lists, with the same MRR@10,
nDCG@10, 128-token limit, and 64-pair evaluation batches as earlier runs. The test
split is not scored. Startup, validation, and checkpoint costs are recorded
separately. Full-data epoch estimates scale the normal throughput to 735,276
training pairs, with a range from the five timing windows. Token distributions,
background load, and sustained thermal behavior can change a longer run's speed.

P1 reports live-forward MPS tensor and driver allocations as sampled values,
and process RSS as a separate high-water mark. These counters overlap on unified
memory and should not be added together. The checkpoint and large CPU trace stay
local and are excluded from Git. The stage breakdown, CPU operator table,
training curves, validation rankings, and final estimate are in `experiments/P1/`.

Completed P1 results (Apple M5 GPU via PyTorch/MPS):

| Measurement | Result |
| --- | ---: |
| Training pairs / batches | 10,000 / 625 |
| Completed optimizer updates / scaler skips | 624 / 1 |
| Normal training loop | 41.73 s; 239.65 pairs/s |
| Setup / checkpoint save and reload | 5.06 s / 0.40 s |
| Full validation queries | 45,709 |
| Validation nDCG@10 / MRR@10 | 0.897533 / 0.861187 |
| Sampled live-forward MPS tensor allocation | 168,804,352 bytes |
| Sampled MPS driver allocation | 1,155,891,200 bytes |
| Process peak RSS | 432,881,664 bytes |
| Estimated full-training epoch, excluding validation/save | 51.1 minutes; window-based range 47.2–57.1 minutes |

The final scaler value was 32,768 after one skipped update. Every selected pair
was processed once; the skipped batch did not change model parameters. The
checkpoint audit found all 57 state tensors changed from the random initial
weights and all 57 AdamW parameter states at step 624.

| Synchronized replay stage | Share of measured stage time |
| --- | ---: |
| Backward and unscale | 36.1% |
| Forward and BCE loss | 23.0% |
| Optimizer and scaler update | 21.1% |
| Gradient checks | 6.8% |
| Tokenization and collation | 6.1% |
| Input transfer | 2.8% |
| Data reading | 1.5% |
| Forward checks, gradient clearing, scalar logging | 2.5% |

Forward/backward and optimizer work dominate this synchronized replay; data
loading and tokenization are smaller contributors. The separate CPU trace also
shows substantial scalar-read/CPU-wait activity (`aten::_local_scalar_dense`,
24.3% of self CPU time). CPU trace percentages and synchronized wall-time stage
percentages have different denominators and must not be combined.

For context, the prior four-layer head-tuned scratch checkpoint had 0.860881
nDCG@10, and MiniLM had 0.896600 on the same constructed validation set. P1's
0.897533 is a local single-run observation on two-candidate lists; it is not a
claim of superiority on a realistic retrieval benchmark. Compared with E1,
P1 changes data exposure, sample/order, and training precision together.

The full-data configuration supported by this run is the same four-layer,
FP16-autocast, direct-batch-16 setup with zero loader workers. No new method sweep
is needed before scaling the training sample. Full training has not been
started by P1. The duration extrapolation is preliminary: the longer workload
may have a different token-length distribution or thermal behavior. See
`experiments/P1/report.json`, `training.json`, `profiling.json`, and
`checkpoint_audit.json` for the recorded measurements and checks.

### Full-data training with bounded profiling

`scripts/full_train.py` trains all 735,276 cleaned training pairs once per epoch,
starting from the saved E1 random initialization. It uses P1's four-layer model,
all 4,717,185 trainable parameters, FP16 autocast with MPS GradScaler, FP32
parameters and BCE loss, AdamW at 3e-4, dropout 0.1, dynamic padding to at most
128 tokens, and direct batches of 16. There are 45,955 batches per epoch; the
last contains 12 pairs. Query splits are checked while reading training pairs.

```bash
.venv/bin/python -m scripts.full_train --preflight --profile
.venv/bin/python -m scripts.full_train --epochs 1 --profile --launch
```

The detached launcher records its PID in `runs/full_train/launch.json`, writes
`training.log`, and uses `caffeinate` when available to prevent idle sleep during
training. `progress.json` contains the latest progress and throughput estimate;
`progress.jsonl` keeps the history. The pilot suggests roughly 51 minutes per
epoch before validation and checkpoint costs; actual sustained throughput can
differ.

Profiling runs on the actual training trajectory. Early, middle, and late
windows each measure 50 batches with GPU synchronization around stages, followed
by 10 batches with CPU operator tracing. Only 180 of 45,955 batches receive this
extra instrumentation. `epoch_0001_profiling.json` records the stage table and
trace paths. CPU traces describe CPU operations and waits, not GPU kernel
durations. Epoch wall time includes profiling, logging, and periodic checkpoints.

Every 1,000 attempted batches, the trainer saves model weights, AdamW state,
GradScaler, CPU/MPS RNG, sampler order, and consumed offset. Two alternating
checkpoint slots preserve the previous checkpoint until a new save and read-back
verification succeed; `latest.json` points to the current slot. An interrupted
run can resume with the same configuration and profiling flag:

```bash
.venv/bin/python -m scripts.full_train --epochs 1 --profile --resume --launch
```

SIGINT or SIGTERM requests a save after the current batch. Resume restores the
remaining sampler order rather than restarting the epoch. Code, data,
initialization, and tokenizer hashes are checked; changing them requires a new
run directory. `--epochs` is the total desired epoch count, including completed
epochs when resuming.

After each epoch, the saved model is evaluated in FP32 on all 45,709 fixed
validation candidate lists. `epoch_0001_training.json` and
`epoch_0001_summary.json` record training and ranking metrics; `best.json` points
to the epoch with the highest validation nDCG@10. The test split remains reserved.
Full-run outputs and checkpoints stay local under `runs/full_train/`, outside Git.
The compact `results/full_training.json` records all three completed epochs,
their validation/profiling results, provenance, and checkpoint hashes in Git.

### S1: epoch-3 PyTorch/MPS versus MLX serving

The selected full-data model is `runs/full_train/epoch_0003.pt`, checkpoint SHA-256
`2ea2608a9ec98083b89606d13bed1d7c6e7a90dc6160bbe44732166137538d52`.
`experiments/S1/` preserves E9's benchmark protocol in a separate package, leaving
the earlier head-only measurements intact. Both backends run FP32 inference on
the Apple M5 GPU; MLX uses `MLX_ENABLE_TF32=0`. Tokenization and HTTP handling use
the CPU. No quantization or cross-request batching is applied.

Conversion checks parameter names/shapes, checkpoint and configuration hashes,
and saved training provenance. Numerical probes check identical tokenization,
padding invariance, singleton batches, long-input truncation, hidden layers, and
scores. The trained model needs hidden-state tolerance `atol=2e-4, rtol=1e-4`
(rather than E9's `atol=1e-4`); score tolerance remains `atol=1e-5, rtol=1e-4`.
All 45,709 fixed validation query rankings matched the PyTorch reference, with
maximum score difference 0.000005484. Both achieved nDCG@10 0.957608 and MRR@10
0.942571. The test split was not evaluated.

| Concurrent HTTP requests | PyTorch requests/s | MLX requests/s | PyTorch median latency | MLX median latency |
| --- | ---: | ---: | ---: | ---: |
| 1 | 116.6 | 187.9 | 7.83 ms | 5.25 ms |
| 4 | 159.6 | 268.1 | 23.79 ms | 14.43 ms |
| 8 | 148.8 | 246.1 | 49.17 ms | 29.31 ms |

These are medians across ten repeated workloads per backend/concurrency: five
repeats in each of two rounds with reversed backend order. Each repeat contains
128 fixed validation queries, with two candidate documents per request. The
loopback HTTP measurement includes queueing, tokenization, scoring, sorting, and
JSON responses. Each server has one model worker and a bounded queue; benchmark
servers run sequentially and stop afterward. All 7,680 timed requests succeeded
and returned the expected rankings. Throughput varied between repeats, so these
numbers describe this local workload rather than a universal backend ranking.

| Direct pair batch | PyTorch complete scoring pairs/s | MLX complete scoring pairs/s |
| --- | ---: | ---: |
| 1 | 202 | 345 |
| 4 | 710 | 646 |
| 8 | 855 | 1,278 |
| 16 | 1,081 | 1,289 |
| 64 | 2,073 | 1,161 |

Direct scoring includes tokenization and input transfer, measured over 256 pairs
with five repetitions per batch size. For this run, MLX is the better choice for
the measured small-request HTTP workload, while PyTorch is faster for large
direct batches. Requests with many candidate documents need their own serving
measurement before applying the HTTP conclusion to that workload.

```bash
.venv/bin/python -m experiments.S1.validate
.venv/bin/python -m experiments.S1.benchmark --backend pytorch
.venv/bin/python -m experiments.S1.benchmark --backend mlx
.venv/bin/python -m experiments.S1.benchmark_serving
.venv/bin/python -m experiments.S1.compare
# Start the selected model as a local HTTP service when needed:
.venv/bin/python -m experiments.S1.serve --backend mlx --port 8099
```

Reports: `experiments/S1/comparison.json`, `quality.json`, `parity.json`,
`conversion.json`, and the direct/serving benchmark JSON files. Converted MLX
weights and server logs stay local outside Git. S1's queue, timeout, cancellation,
validation, and worker-recovery tests pass (three tests).

#### Batch-size and concurrency sweeps

Run `.venv/bin/python -m experiments.S1.sweep` to reproduce the epoch-3 sweeps.
They use the first 128 fixed **validation** queries (256 pairs), not the test
dataset. Direct scoring measures batches 1, 2, 4, 8, 16, 32, 64, and 128. HTTP
serving measures concurrency 1, 2, 4, 8, 16, and 32 with two documents per request,
one model worker, and no cross-request batching. Each point uses six repeats,
split into two rounds with reversed backend order. Direct batch order reverses
in the second round; HTTP concurrency order alternates between repeats. Backends
run in isolated processes, sequentially. The client connection limit accommodates
the largest concurrency, avoiding an unintended 16-connection ceiling.

`experiments/S1/sweep_results.json` stores medians, raw values, and quartiles.
Raw direct round reports and `sweep_http_raw.json` retain timing details.
`sweep_batch.png/svg` plot complete-scoring throughput and median batch latency;
`sweep_concurrency.png/svg` plot HTTP throughput, median request latency, and p95.
Shaded areas are interquartile ranges across repeats, not confidence intervals.

| Pair batch | PyTorch scoring pairs/s | MLX scoring pairs/s |
| --- | ---: | ---: |
| 1 | 256 | 352 |
| 2 | 392 | 533 |
| 4 | 651 | 783 |
| 8 | 1,087 | 1,021 |
| 16 | 1,275 | 945 |
| 32 | 1,633 | 1,382 |
| 64 | 1,924 | 1,212 |
| 128 | 1,506 | 1,188 |

| HTTP concurrency | PyTorch requests/s | MLX requests/s | PyTorch median latency | MLX median latency |
| --- | ---: | ---: | ---: | ---: |
| 1 | 119 | 149 | 8.37 ms | 6.64 ms |
| 2 | 149 | 202 | 12.32 ms | 8.74 ms |
| 4 | 151 | 253 | 26.31 ms | 15.36 ms |
| 8 | 146 | 238 | 49.24 ms | 30.14 ms |
| 16 | 152 | 213 | 98.37 ms | 65.51 ms |
| 32 | 116 | 180 | 218.14 ms | 119.10 ms |

All 9,216 timed HTTP requests succeeded with the expected rankings; all four
benchmark servers shut down cleanly. MLX led at direct batches 1–4; PyTorch led
at batches 8–128, with its highest measured median throughput at 64. MLX led HTTP
throughput at every tested concurrency. Higher concurrency increased queueing
and latency rather than increasing model batch size. Timing varied substantially
between repeats, particularly at high concurrency and batch 128; these are local
measurements, not a precise universal crossover. Large-document HTTP reranking
remains a separate workload from this two-document concurrency test.

#### Interactive Plotly charts

Run `.venv/bin/python -m experiments.S1.plotly_sweep` to render the saved sweep
results without rerunning GPU benchmarks. Open `experiments/S1/sweep_interactive.html`
for interactive batch-size and concurrency charts, with throughput, latency,
hover values, zoom, and repeat variability. The HTML embeds Plotly for offline use.

### E12: Rust serving for inference

`rust-server/` implements the HTTP coordinator using [Axum](https://docs.rs/axum/0.8.9/axum/)
and [Tokio bounded channels](https://docs.rs/tokio/1.53.2/tokio/sync/mpsc/index.html).
This serves inference from the selected epoch-3 model; training remains in PyTorch.
Rust owns the child process, request validation, bounded queue, response matching,
timeouts, readiness, and shutdown. `experiments/E12/worker.py` loads the existing
S1 backend once and performs CPU tokenization and GPU scoring in a persistent
Python process. Both MLX/Metal and PyTorch/MPS are selectable.

To isolate the coordinator change, `experiments/E12/python_server.py` is a matched
Python reference using exactly the same child worker and JSON-lines protocol.
Comparing Rust against S1's earlier worker-thread server would also change the
transport/process architecture, so E12 reports against this new reference.

```bash
cargo build --release --locked --manifest-path rust-server/Cargo.toml
# Run one coordinator at a time for measurement:
rust-server/target/release/tiny-reranker-server --backend mlx --port 8100
.venv/bin/python -m experiments.E12.python_server --backend mlx --port 8101
# Automatically start, measure, and stop isolated servers:
.venv/bin/python -m experiments.E12.benchmark
```

Both coordinators accept S1's request schema on `POST /rerank` and expose
`GET /health`. Defaults: one model worker, 32 queued requests, pair batch size 8,
30-second HTTP deadline, and 60-second worker execution deadline. There is no
cross-request batching. CLI flags `--pair-batch-size`, `--queue-size`, `--timeout`,
and `--worker-timeout` control these settings. The HTTP request body is limited
to 8 MiB minus 1 KiB to leave protocol overhead; each JSON-lines frame is limited
to 8 MiB. Requests allow at most 256 documents, 2,000 query characters, 20,000
characters per document, and 256 characters per ID. The model tokenizer truncates
pairs to 128 tokens. Duplicate document IDs and invalid top-k are rejected.

The readiness handshake declares protocol version, backend, batch size, token
limit, and checkpoint hash. Every worker response must match both an internal
sequence number and request ID. Results are checked for finite scores, valid
unique document IDs, output length, and score/document-ID sorting. Worker logs
go to stderr; stdout carries protocol frames only. One manager owns both pipes,
so HTTP handlers cannot race over the response stream.

Queue saturation returns 503; HTTP deadline expiry returns 504. Expired queued
requests are skipped. An in-flight request continues until its response is read
or the worker execution deadline is reached, even if its HTTP client timed out.
This preserves stream alignment. An ordinary inference error returns 500 and
allows subsequent requests. Child crashes, malformed/oversized output, or
protocol mismatches return 502 for the affected request and make readiness false;
subsequent requests receive 503. Automatic worker restart is not implemented.
SIGINT/SIGTERM closes admission, drains accepted work, closes the worker's stdin,
and reaps the process, with a kill fallback if it does not exit within five seconds.

Verification commands:

```bash
cargo build --locked --manifest-path rust-server/Cargo.toml
cargo test --locked --manifest-path rust-server/Cargo.toml
cargo clippy --locked --manifest-path rust-server/Cargo.toml -- -D warnings
.venv/bin/python -m pytest tests/test_e12_service.py -q
```

Sixteen real-HTTP/process tests cover both coordinators with a GPU-free fixture:
request validation and body limits, queue-full responses, expired queued requests,
safe handling of a timed-out in-flight response, inference-error recovery, child
crashes, mismatched sequence numbers, malformed and oversized frames, invalid
ranking IDs, idle child death, and child cleanup. The Rust unit test additionally
checks nonfinite scores. Tests that bind loopback sockets require permission to
use local network ports. Build outputs and server logs remain outside Git;
`Cargo.lock` records the resolved Rust dependencies.

Completed E12 results (median of ten repeated workloads per point, two isolated
rounds in reversed coordinator/backend order; 128 two-document validation
requests per repeat):

| GPU backend | Concurrent requests | Python requests/s | Rust requests/s | Python median latency | Rust median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX | 1 | 230.8 | 230.9 | 4.06 ms | 3.76 ms |
| MLX | 4 | 306.6 | 428.4 | 12.68 ms | 8.45 ms |
| MLX | 8 | 295.8 | 379.0 | 26.79 ms | 17.76 ms |
| PyTorch/MPS | 1 | 145.3 | 129.8 | 6.57 ms | 7.29 ms |
| PyTorch/MPS | 4 | 198.9 | 180.5 | 19.29 ms | 21.38 ms |
| PyTorch/MPS | 8 | 187.5 | 190.5 | 40.71 ms | 40.04 ms |

All 15,360 timed requests returned expected rankings with no failures. Startup
also checked invalid-input/top-k semantics and scores against the saved PyTorch
validation reference on 16 warm-up queries per server. All benchmark servers
stopped and reaped their workers. That validation benchmark left the test dataset untouched. The
comparison and full repeat timings are in `experiments/E12/comparison.json` and
`benchmark_raw.json`.

Rust + MLX showed higher median throughput at concurrency 4 and 8 in this run,
while the PyTorch results did not show a consistent Rust advantage. Throughput
ranges overlap substantially; this short local workload does not establish a
universal coordinator speedup. Model weights and computation are unchanged, and
Rust does not replace the GPU inference implementation.

A qualitative demo also exposed a ranking failure: for “How do I save a PyTorch
model?”, the epoch-3 model ranked a gardening passage above a model-saving
passage. Direct PyTorch reproduced the MLX HTTP scores within 0.000002, so this
is model behavior rather than a Rust/MLX conversion discrepancy. The exact
request and scores are recorded in `experiments/E12/demo_observation.json`.
High metrics on the constructed validation lists do not establish general
ranking quality; broader realistic evaluation remains pending.

#### Fixed test evaluation and batch-size comparison

`experiments/E12/test_compare.py` evaluates the epoch-3 checkpoint selected from
validation on the untouched constructed test split. The fixed test manifest has
45,665 evaluable queries and 91,337 candidate pairs. It excludes 4,447 source
queries with no clean candidate pairs; the denominator is declared explicitly.
There are 45,658 two-document lists and seven three-document lists.

The full test is served once through each Python/Rust × PyTorch/MLX combination
at concurrency 4, worker batch limit 8, FP32, and a 128-token limit. Per-query
rankings are preserved and compared against the Python/PyTorch reference. These
full-pass timings include HTTP, metric calculation, and JSONL writes; they are
one observation per configuration, not a repeated coordinator-speed experiment.

The separate direct-scoring sweep uses the first 128 fixed test queries, batches
1/2/4/8/16/32/64/128, and six timed repeats per point in two reversed-order rounds.
These batches combine pairs from different queries. An HTTP batch limit of 64
would still execute only two or three pairs for this candidate fixture because
the servers do not batch across requests. Direct-batch and HTTP timing scopes
must therefore remain separate.

```bash
.venv/bin/python -m experiments.E12.test_compare
.venv/bin/python -m experiments.E12.plot_test
```

The frozen `test_plan.json` records model/data/code hashes and the grid before
test evaluation. No new weights, hyperparameters, or batch grid are selected
using test metrics. Results are in `experiments/E12/test_comparison.json`, the
four `test_*_summary.json` and `test_*_rankings.jsonl` files, and S1's
`test_direct_round_*.json` files. `test_progress.json` records current progress.
`test_interactive.html` renders the saved batch results and full-test quality
table without rerunning inference. These constructed lists still do not replace
a realistic retrieval evaluation.

Completed fixed test evaluation (2026-10-09), same epoch-3 weights throughout:

| HTTP coordinator | GPU backend | Test nDCG@10 | Test MRR@10 | Median request latency | Full-pass seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Python | PyTorch / MPS | 0.958466 | 0.943732 | 20.51 ms | 246.97 |
| Rust | PyTorch / MPS | 0.958466 | 0.943732 | 18.81 ms | 226.40 |
| Python | MLX / Metal | 0.958466 | 0.943732 | 12.49 ms | 141.01 |
| Rust | MLX / Metal | 0.958466 | 0.943732 | 9.56 ms | 120.97 |

All 182,660 scored HTTP requests succeeded. Every configuration produced the
same document order on all 45,665 queries. Maximum MLX score difference from
PyTorch was 0.000005723; Rust/PyTorch's difference was below 0.000000000000001.
The test split has now been viewed and must not become a tuning set. Request
latencies and pass times above are single-pass observations at concurrency 4;
use the repeated validation serving experiment for coordinator speed comparisons.
Per-query ranking files are kept locally outside Git; their hashes are retained
in the compact summary reports.

Fixed test-sample direct scoring (tokenization, transfer/conversion, GPU forward,
and CPU-readable scores), median of six timed repeats per point:

| Pairs per batch | PyTorch pairs/s | MLX pairs/s | PyTorch batch ms | MLX batch ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 332.7 | 707.3 | 2.87 | 1.35 |
| 2 | 563.4 | 885.2 | 3.27 | 1.94 |
| 4 | 867.5 | 1230.4 | 4.43 | 3.19 |
| 8 | 1569.4 | 1797.1 | 4.99 | 4.47 |
| 16 | 1901.3 | 1636.3 | 8.43 | 9.47 |
| 32 | 1652.9 | 1774.6 | 19.18 | 17.98 |
| 64 | 2206.2 | 1548.7 | 28.81 | 41.10 |
| 128 | 2692.4 | 1638.2 | 48.09 | 78.09 |

MLX was faster at batches 1–8 and 32; PyTorch was faster at 16, 64, and
128. The crossover is not monotonic. These results depend on the fixed sample
and its padded sequence shapes, and have repeat-to-repeat variation shown as
IQR bands in `experiments/E12/test_interactive.html`. Both coordinators use the
same worker; direct inference measures the backend rather than HTTP overhead.

### E11 — epoch-3 MLX inference profiling

Completed operation-level inference profiling on 2026-10-09, using the existing
epoch-3 checkpoint, FP32 with TF32 disabled, and fixed validation inputs. No
weights changed and no additional test-set selection occurred. The first 128
validation queries supply pair prefixes for batches 1/2/8/16/64/128; batch 1 has
37 padded tokens and the remaining batches have 128. Each mode has 30 repeats,
with instrumented scoring, normal scoring, and resident-input forward interleaved
in rotating order. Model loading and HTTP serving are excluded from steady-state
timings. The serving process was paused to isolate GPU use.

| Pairs per batch | Padded tokens | Normal scoring ms | Resident-input forward ms | Tokenization ms | Input preparation ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37 | 1.416 | 1.289 | 0.189 | 0.061 |
| 2 | 128 | 3.536 | 2.857 | 0.674 | 0.065 |
| 8 | 128 | 9.400 | 6.373 | 1.908 | 0.223 |
| 16 | 128 | 14.690 | 11.328 | 2.853 | 0.220 |
| 64 | 128 | 56.044 | 30.681 | 29.798 | 0.436 |
| 128 | 128 | 156.686 | 69.105 | 102.470 | 0.555 |

Tokenization and preparation come from the instrumented mode; medians in this
table are not additive. Preparation includes materializing input arrays and
synchronizing. Resident forward includes CPU-readable scores. This fixture uses
one padded batch of a sample prefix, unlike S1's throughput sweep over all sample
pairs split into batches; do not substitute these latencies for the previous
throughput results. Sequence shapes and raw text lengths influence the results.

The profiler reconstructs all 44 model stages, checks final scores against the
unchanged MLX model, then replays each operation on resident inputs with explicit
materialization and synchronization. All six fixtures had zero score difference
and identical score order. The category breakdown below sums isolated operation
medians, including dispatch and per-operation synchronization; it is **not** a
percentage of uninterrupted model GPU time. Ten normalization calls, eight
residual additions, and the other repeated operations are grouped by category.

| Category | Batch 2 isolated sum ms | Batch 128 isolated sum ms |
| --- | ---: | ---: |
| layer norm | 2.591 | 18.046 |
| residual add | 1.942 | 10.085 |
| attention projections | 2.258 | 14.546 |
| attention core | 1.301 | 18.416 |
| feed forward linear | 2.241 | 19.305 |
| gelu | 1.196 | 21.829 |
| embedding | 0.302 | 1.601 |
| scoring head | 0.279 | 0.635 |

For small requests, the provisional E11 learning candidate is fused residual
addition plus LayerNorm. In this pre-norm model the residual sum is also needed
by the following branch, so a fused implementation may need to return both the
sum and normalized values; silently discarding the sum would change the model.
Normalization and residual additions have appreciable isolated overhead, while
GELU and dense feed-forward layers become more prominent at large batches. A
kernel's end-to-end benefit was unmeasured at this profiling stage; the
subsequent implementation and benchmark below records the result.

Reproduction and offline plot:

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python -u -m experiments.E11.profile_mlx --capture
.venv/bin/python -m experiments.E11.plot_profile
```

Results and all repeat samples are in `experiments/E11/profiling.json`; the
initial, non-interleaved pass is preserved in `profiling_initial.json`. CPU call
attribution is in `cpu_dispatch.txt` and the local `cpu_dispatch.prof`; cumulative
CPU timings include GPU waits and must not be presented as GPU-kernel durations.
`profiling_interactive.html` shows phase timings, repeat IQR, and isolated category
times offline. This profile does not directly compare PyTorch and MLX.

Two local Metal captures preserve three unchanged forward passes each at batches
2 and 64 (`epoch3_batch_2.gputrace` and `epoch3_batch_64.gputrace`). Capture was
separate from timing runs. Xcode/xctrace are not installed on this Mac, so GPU
timelines, per-kernel timestamps, occupancy and bandwidth have not been inspected.
The trace bundles and binary CPU profile are excluded from Git. MLX's
[official capture documentation](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
describes the capture environment and opening traces in Xcode. Existing captures
are preserved on rerun.

### E11 — fused residual addition + LayerNorm Metal kernel

Implemented in `experiments/E11/residual_layer_norm.metal` using MLX's
[custom Metal interface](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
The wrapper `experiments/E11/kernel.py` exposes both ordinary MLX and custom
paths. Each 32-thread SIMD group handles one token row, keeps the residual sum
in registers, reduces the mean and population variance in FP32 using two passes,
and applies epsilon, learned scale and bias. SIMD reductions need no threadgroup
shared-memory barrier in this one-SIMD-group-per-row design. The kernel writes
both the residual sum and its normalized value, preserving the later branch.

The custom contract is nonempty rank-2/3 FP32 inputs with matching residual/branch
shapes, FP32 affine vectors, and hidden sizes 32/64/128/256. The trained model
uses hidden size 128. MLX packs strided arrays before dispatch, with copy cost
included in caller timings. Unsupported sizes/dtypes fall back to ordinary MLX
or raise when `fallback=False`; malformed shapes/epsilon raise. Inputs should
have finite values with representable residual sums and reduction intermediates.
The numerical tests include constant 1e20 rows and mixed-sign values scaled by
1e15, not a claim of overflow-free behavior for every possible FP32 input.

`experiments/E11/mlx_model.py` adds `use_fused=True` to the existing architecture.
It fuses eight residual-add/normalization pairs across the four pre-norm layers,
including the final normalization. The embedding norm and first attention norm
remain ordinary MLX. The residual sum continues down the original branch;
normalization parameters belong to the next consumer exactly as in the reference.
Weights and architecture are unchanged. The default is ordinary MLX; this
experiment does not change the HTTP workers. The custom model path rejects
training mode, and no custom backward implementation is provided. The user
confirmed this task is epoch-3 inference benchmarking, not another training epoch.

All 35 actual-GPU tests passed: supported shapes/hidden sizes, odd row counts,
multiple lengths, constant/cancellation/large finite values, strided packing,
fallbacks and rejection, all encoder-state comparisons, and the training guard.
Operation tolerance is atol=rtol=2e-5 against ordinary MLX and a float64 NumPy
oracle on random fixtures; complete-model state tolerance is atol=rtol=3e-5.
The implementation follows the standard two-pass population-variance algorithm;
MLX's [LayerNorm source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/layer_norm.metal)
was checked to match that convention and the epsilon/affine placement.

```bash
.venv/bin/python -m pytest tests/test_e11_kernel.py -q
.venv/bin/python -u -m experiments.E11.benchmark_kernel
.venv/bin/python -m experiments.E11.plot_kernel
```

`benchmark_kernel.py` checks trained-model states and padding behavior, measures
first kernel use separately, warms both paths, runs 30 paired operation timings,
compares both execution paths on all 45,709 validation queries, and then runs
the fixed 128-query batch sweep. Whole-model timings have 20 repeats per point
in two reversed batch-order rounds; path order and forward/scoring order also
alternate. Both paths share the same model weights and encoded inputs. These
are synchronized direct-inference wall timings, not pure GPU timestamps or HTTP
serving speed measurements. No test data is accessed for this kernel experiment.

Full validation produced identical rankings and quality across ordinary/fused
MLX: nDCG@10 0.957607931196 and MRR@10 0.942571484828, with zero changed query
rankings. Maximum score difference was 0.000001669 across 91,424 candidate pairs.
`kernel_quality.json` retains compact quality results and the hash of the locally
saved `fused_validation_rankings.jsonl`; the large ranking file is outside Git.

Completed E11 timing results (20 paired whole-model repeats per point):

| Pairs per batch | Isolated operation speedup | Prepared-forward throughput ratio | Complete-scoring throughput ratio | Ordinary scoring pairs/s | Fused scoring pairs/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.671x | 1.140x | 1.024x | 414.6 | 424.7 |
| 2 | 1.022x | 1.082x | 0.921x | 555.1 | 510.9 |
| 8 | 1.277x | 1.072x | 0.939x | 856.3 | 804.1 |
| 16 | 1.542x | 1.086x | 1.067x | 1072.5 | 1144.9 |
| 64 | 1.599x | 1.052x | 1.032x | 1304.0 | 1346.1 |
| 128 | 1.920x | 1.091x | 1.091x | 1397.0 | 1524.3 |

First kernel use took 2.812 ms, including setup/JIT work
when needed; it is excluded from steady-state timing. This is not pure compiler time.

The fused operation improved isolated median wall time at all tested points,
and prepared-input whole-model throughput was 5–14% higher in this run. Complete
scoring was mixed: approximately 8% lower throughput at batch 2, 6% lower at
batch 8, and modest gains at the other points. Repeat ranges overlap substantially;
these medians do not establish a reliable production speedup. Tokenization and
other CPU work remain part of complete scoring, and this run does not attribute
the differences to a single cause. Ordinary MLX remains the default, particularly
for the existing two-document HTTP workload. No HTTP speedup is claimed and no
training was performed.

The completed experiment retains the custom kernel as a learning result with a
measured limitation. Raw samples, hashes, probe errors, complete validation
quality, and startup timing are saved in `experiments/E11/kernel_benchmark.json`.
`kernel_interactive.html` plots both throughput scopes, repeat IQR, isolated
operation ratios, and ranking parity. The GPU correctness tests and full-validation
comparison verify the implemented path; expected floating-point differences
are small, but outputs are not claimed bitwise identical.

#### Residual addition + LayerNorm: isolated latency savings

This measures residual addition followed by LayerNorm only; attention itself
is not fused. Results are medians of 30 paired repeats on resident inputs,
including Python dispatch, materialization and synchronization, not pure GPU
kernel timestamps.

| Pairs per batch | Ordinary MLX ms | Custom kernel ms | Saved ms | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.473 | 0.283 | 0.190 | 40.1% |
| 2 | 0.270 | 0.264 | 0.006 | 2.2% |
| 8 | 0.666 | 0.521 | 0.144 | 21.7% |
| 16 | 0.636 | 0.412 | 0.223 | 35.1% |
| 64 | 1.013 | 0.633 | 0.379 | 37.5% |
| 128 | 2.698 | 1.405 | 1.293 | 47.9% |

At batch 2, the median saving was only 0.006 ms (2.2%). Ordinary MLX's IQR was
0.239–0.293 ms and the custom kernel's was 0.238–0.296 ms: the ranges almost
completely overlap, so this does not establish a meaningful small-request gain.
Larger batches showed more substantial isolated savings. These savings cannot
be multiplied by the number of model occurrences or treated as the same
percentage reduction in complete-model latency; launch scheduling, input shapes,
other operations and measurement overhead differ in the full graph. Complete
scoring results remain mixed, and ordinary MLX remains the default.

The source measurements and all repeat samples remain in
`experiments/E11/kernel_benchmark.json` under `operation_microbench`.

### E13 — Triton kernel measured on NVIDIA L40S

The inference-only FP32 residual addition + LayerNorm source is written in
`experiments/E13/triton_impl.py`. It uses Triton's
[official LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
as a reference for one-program-per-row reductions, adapting the forward algorithm
to add the residual in registers and return both the sum and normalized values.
The code uses masked loads/stores for non-power-of-two hidden dimensions, FP32
two-pass population variance, model epsilon and learned scale/bias. No tutorial
backward or training code was incorporated. GPU compilation and numerical
correctness have now been verified on the L40S; the measured experiment is below.

`experiments/E13/kernel.py` lazily imports Triton only for a supported NVIDIA
launch. Its custom contract is contiguous, nonempty rank-2/3 FP32 arrays, matching
residual/branch shapes and affine vectors on the same CUDA device, hidden size
1..1024, and 1/2/4 warps. Other layouts/dtypes/devices fall back to ordinary
PyTorch or raise under `fallback=False`. Invalid shape/epsilon/launch metadata
raises. Custom calls reject gradient-enabled inputs requiring gradients; use
`torch.inference_mode()`. Finite inputs must have representable sums and reduction
intermediates. The Mac environment and its MLX dependencies are unchanged.

`experiments/E13/model.py` preserves the original checkpoint's state-dict keys.
Its `kernel_mode` switch selects `original`, matched explicit PyTorch `reference`,
or `triton`. The two explicit paths share attention, feed-forward operations and contiguous
packing of branch outputs, and differ in the eight residual-add/LayerNorm pairs.
Packing is necessary because batch-first MHA can return transposed views; its
cost is included in both matched timing paths. The stock original remains a
separate practical baseline without this imposed packing. Returning the residual
sum preserves the pre-norm architecture. Explicit paths reject training mode;
Triton requires inference mode. Ordinary PyTorch remains the default.

Local verification: 14 CPU tests passed and 72 NVIDIA tests were explicitly
skipped. CPU tests check fallback/rejection, model-wiring parity at lengths
1/3/37/128, inference guards and the missing-GPU benchmark status. The skipped
suite covers hidden sizes 1/32/96/128/129/256/1024, odd row counts, three launch
configurations, numerical edge cases, strides/dtype fallbacks, gradient rejection,
and complete-model output checks. Python syntax checks passed; they do not compile
Triton GPU code. Subsequent NVIDIA execution passed all 86 tests; the completed
hardware report is now in `experiments/E13/hardware_status.json`.

On an agreed compatible Linux/NVIDIA environment, run from the repository root:

```bash
python -m pytest tests/test_e13_triton.py -q -rs
python -m experiments.E13.benchmark --require-gpu
```

Prepare a separate compatible PyTorch CUDA + Triton environment; the benchmark
also uses NumPy, pytest and Transformers through the shared
project evaluation helpers. Do not install the Mac `requirements.txt` wholesale
on Linux: it includes MLX. The [current Triton compatibility requirements](https://github.com/triton-lang/triton#compatibility)
list Linux and NVIDIA compute capability 8.0+. Confirm the chosen runtime versions
and GPU before execution; no remote or paid GPU was provisioned for this task.

The runner requires the epoch-3 checkpoint and its full-training summary/config,
`experiments/E1/four_layer/config.json`, the pinned tokenizer bundle under
`artifacts/short_epoch_1024`, and validation candidates plus their manifest. These
large local assets are outside Git and must be copied to the chosen machine.
It checks their hashes/provenance before loading weights or comparing outputs.
Without compatible hardware it writes a pending status and produces no fabricated
benchmark results; `--require-gpu` additionally exits with code 2.

The prepared GPU benchmark measures first kernel use separately, checks 1/2/4-warp
operation variants, checks score/ranking parity on the fixed first 128 validation
queries, then sweeps batches 1/2/8/16/64/128 with 20 rotated-order repeats by default.
It measures prepared-input forward and complete scoring with CPU-readable scores.
It reports four PyTorch model paths: original, reference and Triton disable PyTorch MHA fast paths,
while a fourth `optimized` path enables the stock fused encoder. Compare the matched paths for the
kernel's isolated effect, and the stock model for practical deployment performance.
The benchmark records GPU/driver/runtime versions, hashes and raw samples in
`experiments/E13/benchmark.json` only after an actual GPU run. Its 128-query parity
fixture is not a full validation/test evaluation. The subsequent full test
ranking evaluation is recorded under E14 below. NVIDIA kernel correctness and measured latency are recorded below;
ordinary PyTorch remains the default.

On 2026-10-10, the borrowed Georgia Tech Rogues Gallery account authenticated
through `rg-login` without VPN. Slurm job 172826 installed an isolated environment
on CPU node `crush`. Job 172829 completed the GPU tests and benchmark on `dash2`:
NVIDIA L40S, compute capability 8.9, driver 610.43.02, PyTorch 2.14.1+cu130,
CUDA runtime 13.0, Triton 3.8.0. No credentials are saved in the project. Setup
and benchmark batch scripts, exact installed packages, successful test logs and
the CUDA dispatch diagnostic are retained in `experiments/E13/`.

All 86 tests passed. On the fixed 128-query/256-pair validation fixture, Triton
versus matched reference had maximum score error 9.5367431640625e-7 and zero
changed rankings: MRR@10 0.953125, nDCG@10 0.9653996643973239. This is sample
parity, not the full validation or test evaluation.

The initial benchmark stopped because the trained checkpoint's stock fused
PyTorch fast path differed from the explicit reference by up to 0.0066599846.
Disabling that fast path made original/reference scores exactly equal on the
fixture. The diagnostic isolates this difference to PyTorch dispatch, not Triton;
the precise internal source of the fused-path difference was not established.
The final runner keeps original/reference/Triton on matching dispatch, retains
their strict parity checks, and reports the optimized stock path separately.
The optimized path changed zero rankings on this small fixture, but that does not
establish full ranking equivalence.

Median complete scoring latency per pair batch (milliseconds; tokenization,
GPU transfer, forward computation and CPU-readable scores; 20 rotated repeats):

| Pair batch | Matched PyTorch CUDA reference | Custom Triton | Saved custom Metal on M5 |
|---:|---:|---:|---:|
| 1 | 1.311 | 1.683 | 2.307 |
| 2 | 1.713 | 2.098 | 3.660 |
| 8 | 3.040 | 3.415 | 9.025 |
| 16 | 4.734 | 5.083 | 13.090 |
| 64 | 14.921 | 14.987 | 45.023 |
| 128 | 29.410 | 29.363 | 83.914 |

The custom Triton path is slower at small batches and approximately tied at
64/128. At batch 2, prepared-input forward latency was 0.947 ms reference versus
1.305 ms Triton. With four warps, isolated synchronized operation wall latency
was 0.031 ms reference versus 0.056 ms Triton at batch 2; at batch 128 it was
0.042 versus 0.060 ms. These measurements include Python dispatch and GPU
synchronization, so they do not establish the custom kernel's device execution
time or identify the cause of the slowdown. One/two/four-warp samples and raw
measurements are in `benchmark.json`.

First kernel use in the benchmark process took 692.675 ms. The preceding test
process used the same persistent Triton cache, so this is not a clean cold
compilation measurement. Warm measurements exclude this first-use cost.

`metal_triton_comparison.json` verifies matching checkpoint, candidate and sample
hashes before combining the reports. Raw M5-versus-L40S times include hardware
and framework differences; compare custom/reference speedups separately on each
device when assessing the kernel implementation. No Triton speedup or deployment
switch is justified by this run.

### Experimental Triton on the Apple GPU

An isolated `.venv-triton-apple` environment was installed on 2026-10-10 with
Python 3.14.6, triton-msl 0.3.0, the backend author's documented Triton
3.7.0+git4da2e268 macOS wheel, PyTorch 2.14.1 and MLX 0.32.3.
The main `.venv` remains unchanged. Exact package pins are in
`experiments/E13/apple_requirements.txt`.

Setup is **blocked by the missing Metal compiler**, not by missing Python
packages. Imports succeeded, but an actual compilation attempt failed while
the backend established its Metal compiler/SDK identity. Even the MLX runtime
route requires that toolchain; stopping compilation at emitted MSL did not avoid
this prerequisite. The unmodified backend checks remain in place.
`apple_setup_status.json` records the device, installed versions and blocker;
GPU correctness and performance are explicitly false. No Apple Triton latency
or epoch-3 model parity has been measured.

The App Store refused the current Xcode because it requires macOS 26.6+;
this Mac runs 26.5.2. Apple's [compatibility table](https://developer.apple.com/xcode/system-requirements)
lists Xcode 26.6 as compatible with macOS 26.2–26.x. Download that version from
[Apple Developer downloads](https://developer.apple.com/download/all/?q=Xcode%2026.6),
install it, and complete Apple's first-launch setup. Then install the Metal
Toolchain using Xcode Settings > Components, as documented by
[the backend](https://github.com/bledden/triton-msl/blob/main/docs/USAGE.md#installation-details).
Verify `xcrun --find metal` and `xcrun metal --version` with that Xcode selected.

To reproduce the Python environment and resume the correctness check:

```bash
uv venv --python 3.14 .venv-triton-apple
uv pip install --python .venv-triton-apple/bin/python -r experiments/E13/apple_requirements.txt
.venv-triton-apple/bin/python -u -m experiments.E13.apple_smoke
```

The smoke runner uses the same `triton_impl.py` residual + LayerNorm source
through triton-msl's documented `triton_call` MLX interface. It checks FP32
hidden-size-128 tensors against a float64 oracle and requires exact residual
sums before marking success. Its requested 1/2/4-warp values are API parameters;
triton-msl's MLX route derives the actual threadgroup size from compiler metadata,
so they must not be reported as three distinct GPU configurations without inspection.
This experimental Apple route remains blocked and is separate from the completed NVIDIA E13 benchmark.

## E14 — handwritten CUDA port of the Metal kernel versus Triton

Completed 2026-10-10 on the same NVIDIA L40S (dash2), FP32 with TF32 disabled.
Metal cannot execute on this NVIDIA GPU: `experiments/E14/residual_layer_norm.cu`
ports E11's algorithm into CUDA C++, preserving one warp per row, lane-strided
values, centered two-pass population variance, affine normalization and both
outputs. It supports hidden sizes 32/64/128/256. Precise sqrt plus reciprocal
implements reciprocal square root; backend reduction order and math instructions
need not be bitwise identical to Metal. No fast math or fused multiply-add.

The cluster's NVHPC 24.9 module supplies nvcc 12.6. The shared library statically
links the CUDA runtime and launches on PyTorch's current stream through ctypes.
The existing PyTorch 2.14.1+cu130/Triton 3.8.0 environment and driver 610.43.02
were retained. The measured binary SHA256 is recorded alongside all model/kernel
source hashes. Recompile on Linux; the ignored `.so` cannot run on this Mac.

All 50 CUDA tests passed on L40S, including a float64 oracle across four hidden
sizes, random/constant/cancellation inputs, strict CPU/stride/gradient guards.
Epoch-3 weights loaded strictly with verified checkpoint, tokenizer and candidate
hashes. On the same 128-query/256-pair validation fixture, CUDA and Triton each
had maximum reference score error 9.5367431640625e-7, zero changed rankings,
MRR@10 0.953125 and nDCG@10 0.9653996643973239. This is sample parity, not a
full held-out test evaluation; no training or deployment switch was performed.

Complete scoring medians include tokenization, GPU transfer, forward and
CPU-readable scores; 20 rotated repeats. M5 values are saved E11 measurements,
verified to use the same checkpoint, candidates and sample, but different hardware.

| Pair batch | Handwritten CUDA port on L40S (ms) | Triton, 4 warps on L40S (ms) | Original custom Metal on M5 (ms) |
|---:|---:|---:|---:|
| 1 | 1.488 | 1.668 | 2.307 |
| 2 | 1.876 | 2.061 | 3.660 |
| 8 | 3.204 | 3.390 | 9.025 |
| 16 | 4.884 | 5.071 | 13.090 |
| 64 | 14.924 | 14.986 | 45.023 |
| 128 | 29.329 | 29.303 | 83.914 |

To separate host dispatch from GPU work, a second job captured 100 operations
per CUDA graph and measured replay using CUDA events, divided by 100. Each graph
was checked against the reference after replay. These are resident-input,
amortized operation times, not isolated kernel timestamps or serving latency.
Thirty rotated repeats were retained; graph capture and compilation are excluded.
Repeated inputs can benefit from cache reuse, so these are microbenchmarks.

| Pair batch | PyTorch reference (µs) | CUDA port (µs) | Triton 4 warps (µs) | Triton 1 warp (µs) |
|---:|---:|---:|---:|---:|
| 1 | 3.164 | 2.058 | 1.311 | 1.198 |
| 2 | 3.400 | 2.283 | 1.556 | 1.556 |
| 8 | 4.096 | 2.570 | 2.109 | 1.964 |
| 16 | 5.600 | 3.174 | 2.621 | 2.497 |
| 64 | 13.220 | 6.742 | 6.419 | 5.745 |
| 128 | 23.347 | 11.069 | 11.346 | 10.117 |

Both custom operations save device work versus the two-operation reference.
Triton's one-warp device medians are lower than the CUDA port at every tested
size; four warps are lower through batch 64 and close at 128. Ordinary synchronized
wall calls reverse that comparison: the ctypes CUDA port takes approximately
40–47 µs, versus 59–64 µs for Triton at four warps. The CUDA port also improves
small-batch model forward latency versus Triton, but both lose to matched PyTorch
at those batches. At batches 64/128 whole-model differences are small. This is
consistent with host dispatch costs hiding device savings; the graph experiment
does not attribute those costs to a specific Python function. One-warp Triton
whole-model latency was not measured in E14, so its device advantage must not be
claimed as a whole-model result. Keep the ordinary framework default.

Raw repeats, numerical/ranking parity, compiled/source hashes and scopes are in
`benchmark.json`, `device_benchmark.json` and `comparison.json`; successful Slurm
logs are `e14_benchmark_172833.log` and `e14_device_172834.log`. The same-GPU chart
is `comparison.png`; M5 is excluded from that chart to avoid mixing hardware.

Reproduce from the existing isolated E13 cluster environment:

```bash
sbatch experiments/E14/cluster_benchmark.sbatch
# After the above job succeeds:
sbatch experiments/E14/device_benchmark.sbatch
# After downloading reports and the measured .so into experiments/E14:
.venv/bin/python -m experiments.E14.compare
```

The CUDA library is inference only; it does not implement backward propagation.

The CUDA port is a handwritten kernel, not normal PyTorch. For the direct
same-GPU model comparison, prepared-input forward medians are:

| Pair batch | Normal PyTorch reference (ms) | Custom CUDA port (ms) | Custom Triton, 4 warps (ms) |
|---:|---:|---:|---:|
| 1 | 0.830 | 0.994 | 1.161 |
| 2 | 0.953 | 1.131 | 1.306 |
| 8 | 0.909 | 1.094 | 1.296 |
| 16 | 0.901 | 1.073 | 1.268 |
| 64 | 1.449 | 1.420 | 1.438 |
| 128 | 2.727 | 2.709 | 2.696 |

Decision: keep normal PyTorch for the measured small-batch NVIDIA workload. The
custom CUDA port beats Triton there but does not beat the normal reference. Large
batches are close; lower isolated GPU time alone does not establish lower model
latency. The full held-out CUDA ranking evaluation below now confirms quality
parity for the fixed implementations.


### E14 — full held-out test ranking quality

Completed 2026-10-10 on L40S node `dash2`, Slurm job 172835. Epoch 3 was
fixed from validation before test evaluation; no training, checkpoint selection
or launch tuning used these test results. Normal PyTorch, handwritten CUDA and
Triton all evaluated every one of the 45,665 evaluable test queries and 91,337
candidate pairs. Every query's normalized hash was verified against its stored ID
and test bucket 9, using the split manifest pinned by training. Candidate text
hashes, query uniqueness, labels, counts and the saved candidate-file hash were
checked. The 4,447 source test queries with no clean pairs remain excluded under
the existing manifest; this run did not create new candidate lists.

Pair batch size was 128, maximum sequence length 128, dynamic padding, FP32,
TF32 disabled, Triton four warps. PyTorch MHA fast paths were disabled across all
three paths to isolate the residual + LayerNorm changes. Each batch was tokenized
once and the same resident inputs were supplied to each path in rotating order.
The optimized stock PyTorch fast path was not part of this quality comparison.

| Implementation | MRR@10 | nDCG@10 | Changed query rankings vs PyTorch | Maximum score error |
|---|---:|---:|---:|---:|
| Normal PyTorch | 0.943731523048 | 0.958465958690 | 0 | 0 |
| Custom CUDA port | 0.943731523048 | 0.958465958690 | 0 | 1.90734863e-06 |
| Custom Triton | 0.943731523048 | 0.958465958690 | 0 | 1.66893005e-06 |

All 91,337 scores in each custom path passed atol=1e-5, rtol=1e-4 versus
normal PyTorch. Rankings were exactly unchanged for every test query. Both
quality metrics match the earlier E12 held-out evaluation. These constructed
MS MARCO mined-triplet lists mostly contain two documents; this does not establish
quality on realistic retrieval pools or equivalence with published leaderboards.

The evaluation loop took 33.992 seconds including warm-up, tokenization, all
three inference paths, audit writing and checks. Single-pass durations in the
report are diagnostic, not a new latency benchmark. The previous repeated warm
latency conclusions and ordinary-framework default remain unchanged.

Results are in `experiments/E14/test_quality.json`, fixed settings and source
hashes in `test_plan.json`, and successful logs in `e14_test_172835.log`.
The ignored local `full_test_scores.jsonl.gz` contains every candidate score,
label, ranking and per-query metric for all paths, with its checksum in the
summary. After download, `verify_test.py` checked source/artifact hashes, matched
every record to the original saved test candidates, and recomputed all ranking
metrics, ranking changes and maximum score differences. Verification succeeded
for the full dataset; its record is `test_audit_verification.json`.

```bash
# Existing isolated E13/E14 Linux environment and compiled CUDA library:
sbatch experiments/E14/evaluate_test.sbatch
# After downloading results and the gzip audit:
.venv/bin/python -m experiments.E14.verify_test
```

Conclusion: both custom kernels preserve epoch-3 ranking quality on the complete
saved test set. They remain learning/benchmark paths; quality parity alone does
not justify changing the default execution path.

### E14 — measured latency across the full test set

Completed on L40S `dash2`, Slurm job 172836. This is a dedicated latency run
across all 45,665 evaluable test queries / 91,337 pairs, rather than an estimate
from the small fixture or the quality evaluator's shared-tokenization pass.
Each implementation completed three full-dataset passes at pair batch size 128
(714 batches per pass, final batch 73 pairs). Path order rotated across rounds.
Epoch 3, FP32/TF32 disabled, Triton four warps, dynamic padding/max length 128,
and disabled PyTorch MHA fast paths match the full quality evaluation.

Every path tokenizes the full data independently. Each batch timer includes
its tokenization, host-to-GPU transfer, model execution and CPU float scores.
Scoring time is the sum of those batch timers. File loading, model loading,
compilation/warm-up, ranking sorting and quality auditing are excluded; no HTTP
server is involved. Gross pass wall time, which additionally includes loop checks
and progress output, is also retained. Percentiles span batches throughout the
full dataset; the table reports the median of each statistic across three passes.

| Implementation | Full-test scoring (s) | Pass scoring range (s) | Batch P50 (ms) | Batch P95 (ms) | Pairs/s |
|---|---:|---:|---:|---:|---:|
| Normal PyTorch | 21.371 | 21.134–23.735 | 29.634 | 30.801 | 4273.9 |
| Custom CUDA port | 21.272 | 21.120–21.516 | 29.523 | 30.717 | 4293.8 |
| Custom Triton | 21.104 | 21.045–21.252 | 29.476 | 30.571 | 4327.9 |

Triton's median total scoring time is approximately 1.2% lower than normal
PyTorch; the CUDA port is approximately 0.5% lower. The pass ranges overlap,
so three passes do not establish a reliable performance winner at batch 128.
All nine passes produced exactly the previous full-test rankings and metrics;
quality auditing happens after each timed pass. This result does not change the
small-batch framework default or imply HTTP request latency.

Raw timings for every batch, padded lengths, pair counts, all nine pass totals,
P50/P95/P99, provenance hashes and quality checks are in
`experiments/E14/full_test_latency.json`. The successful job log is
`e14_full_latency_172836.log`. Download verification checked the source hashes,
all 714 batch records in every pass, pair/metric counts, total durations and
recomputed percentiles. The entire allocated GPU job completed and released its
resource.

```bash
sbatch experiments/E14/benchmark_full_test.sbatch
```

### E15 — pretrained rerankers and full retrieval tests

Completed on NVIDIA L40S, Slurm job 172839. Epoch 3 (4,717,185
parameters) was compared with frozen pretrained
[cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)
(4,386,049 parameters) and
[cross-encoder/ms-marco-MiniLM-L2-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2)
(15,615,745 parameters). TinyBERT is the size-matched baseline; MiniLM is
a larger reference. Revisions and asset hashes are pinned in the manifests.
No weights were changed and no test-based tuning was performed.

All three models use FP32, TF32 disabled, batch 128, dynamic padding and a
128-token cap for the primary comparison. Each model keeps its own tokenizer.
Normal PyTorch is used for the scratch model; this experiment compares model
quality, not custom kernel performance.

| Model / method | Constructed test nDCG@10 | SciFact nDCG@10 | NFCorpus nDCG@10 |
|---|---:|---:|---:|
| BM25 retrieval | — | 0.652285 | 0.306414 |
| Ours, epoch 3, 128 tokens | 0.958466 | 0.013375 | 0.053624 |
| TinyBERT-L2-v2, 128 tokens | 0.839037 | 0.570315 | 0.307080 |
| MiniLM-L2-v2, 128 tokens | 0.871109 | 0.574879 | 0.309453 |
| TinyBERT-L2-v2, 512 tokens | 0.838738 | 0.660672 | 0.330102 |
| MiniLM-L2-v2, 512 tokens | 0.871084 | 0.601533 | 0.320234 |

The 512-token rows are additional context comparisons, not matched against
our model's 128-token capacity. These results do not identify a globally best
reranker. At 128 tokens MiniLM's small nDCG advantage over TinyBERT has paired
95% bootstrap intervals crossing zero on both retrieval datasets. TinyBERT at
512 tokens improves over BM25 on NFCorpus (difference 0.023688, interval
[0.012128, 0.036438]); its SciFact difference is inconclusive (0.008387,
interval [-0.020530, 0.038282]). Bootstrap uses 2,000 query resamples, seed 42.

**Our strong constructed-test score does not generalize to these two retrieval
tests.** TinyBERT is slightly smaller and substantially stronger here. The
constructed test has 45,665 queries / 91,337 pairs, mostly two documents per
query. Its queries are held out from our training, but the pretrained models
were already trained on MS MARCO; unseen queries for those models are not
established. It is not evidence of general superiority over them.

The additional tests use the complete official
[BEIR](https://github.com/beir-cellar/beir) SciFact and NFCorpus test splits:
300 and 323 queries respectively. Every reranker receives exactly the same
BM25 top-100 candidates: 30,000 and 32,300 pairs. Retrieval uses rank-bm25
0.2.2 BM25Okapi (k1=1.5, b=0.75, epsilon=0.25), lowercase ASCII alphanumeric
tokens, title plus text, and no stemming or stopword removal. This is a defined
local retrieval setup, not a reproduction of an Elasticsearch leaderboard run.
No relevant documents were injected into candidate lists. Queries with no
retrieved positive (35 SciFact, 74 NFCorpus) remain in the evaluation.

Graded nDCG uses linear relevance gains and all official positive qrels for
the ideal ranking. Unretrieved positives count against recall and nDCG.
Recall@100 is identical across rerankers because candidates are fixed:
0.873056 on SciFact and 0.233078 on NFCorpus. NFCorpus's retrieval ceiling
therefore also limits every reranker. Per-query nDCG and recall agree with
pytrec-eval-terrier 0.5.10 to 1e-12; MRR@10 and metric edge cases are tested.

All 15 complete score audits were downloaded and independently verified
against original queries, candidates and qrels, including hashes, finite scores,
rankings, per-query metrics and macro averages. A separate CPU check reproduces
the scratch model's GPU scores and exact rankings on two SciFact queries
(maximum score difference 4.4e-6); this is a sample sanity check, not a full CPU
evaluation. The quality gap's cause is not established by this experiment.

Results: `experiments/E15/quality.json`, `comparison_summary.json`,
`audit_verification.json`, `scratch_scifact_cpu_probe.json` and
`quality_comparison.png`. Successful logs and source/data/model provenance
are retained alongside them. Large per-query gzip audits, model weights and
corpora are retained locally but excluded from Git. Scoring durations in the
quality report are single-pass diagnostics, not a repeated latency benchmark.

```bash
# Download pinned baselines and official BEIR archives before preparation.
.venv/bin/python -m experiments.E15.download_baseline --model tinybert
.venv/bin/python -m experiments.E15.download_baseline --model minilm
uv run --with rank-bm25==0.2.2 --with pytrec-eval-terrier==0.5.10 --with numpy==2.5.3 --python .venv/bin/python -m experiments.E15.prepare
# Existing isolated Linux CUDA environment:
sbatch experiments/E15/setup.sbatch
sbatch experiments/E15/benchmark.sbatch
# After downloading the complete results and all gzip audits:
uv run --with rank-bm25==0.2.2 --with pytrec-eval-terrier==0.5.10 --with numpy==2.5.3 --python .venv/bin/python -m experiments.E15.verify
.venv/bin/python -m experiments.E15.summarize
```

Next quality work should use separate development queries and realistic
retrieval candidates before another final test. Preserve these test results;
do not tune on them. E7 pairwise loss and E10 early exits remain deferred.


### E17 — realistic inference and HTTP benchmark audit

Completed Apple M5 benchmarks using the original epoch-3 checkpoint and fixed
128-token FP32 inference: complete development accuracy at 10/50/100 documents,
and 12 paired repetitions across four process sessions for direct inference
and Python/Rust × PyTorch/MLX loopback HTTP at concurrency 1/4/8.
All four Apple paths (strict, eager stock, compiled stock, MLX) passed the
predeclared numerical/application checks. Compiled stock changed three lower
rankings; top-1/top-10 and quality metrics were unchanged. All 6,912 measured
HTTP requests / 368,640 pairs passed score and ranking audits without failures.

No Rust throughput advantage is supported; 17/18 coordinator throughput
intervals include parity. One exploratory point favors Python. The MPS eager
fast-path flag does not activate the fused encoder; a same-model diagnostic
reversed its apparent timing advantage, so no gain is attributed to that flag.
Compiled PyTorch was slower in complete scoring for the 50/100-document
fixtures (paired ratios 0.900/0.859); prepared-forward gains are inconclusive.
NVIDIA CUDA/Triton versus optimized-stock measurements remain pending because
SSH authentication failed. No CPU or cross-hardware results substitute for them.

See [E17 report and reproduction](experiments/E17/REPORT.md),
[latency/throughput plots](experiments/E17/http_workloads.png), and
[paired coordinator uncertainty](experiments/E17/coordinator_uncertainty.png).
The fixed plan, compact raw repetitions, accuracy distributions, independent
verification and checksums are in that directory. Full score/response audits
remain local outside Git. These are short closed-loop waves on one unisolated
machine; the model's poor retrieval quality is unchanged.
