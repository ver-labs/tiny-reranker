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
