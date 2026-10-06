"""Train the same scratch weights with direct or accumulated gradients."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_torch_save, atomic_write_text
from scripts.collate import make_pair_loader
from scripts.data import iter_clean_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import load_scratch_tokenizer
from scripts.short_epoch import TRAIN_COUNT, VALIDATION_COUNT, ids_hash, measure
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E1 = ROOT / "experiments" / "E1" / "four_layer"
SEED = 123
EFFECTIVE_BATCH = 16
MICROBATCH = {"direct": 16, "accumulate": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_sample() -> dict[str, int]:
    return {
        "tensor_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MICROBATCH), required=True)
    args = parser.parse_args()
    checkpoint = HERE / f"{args.mode}.pt"
    report_path = HERE / f"{args.mode}.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"E5 {args.mode} output already exists")
    if not torch.backends.mps.is_available():
        raise RuntimeError("E5 MPS comparison requires MPS")
    device = torch.device("mps")
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"] or \
            sha256(initial_path) != manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 starting config or weights differ from manifest")
    original_config = ScratchConfig.from_file(config_path)
    # Disable stochastic dropout in both runs so microbatch shape does not
    # change random-mask use. The initialized weights remain identical.
    config = replace(original_config, dropout=0.0)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(original_config) or initial["seed"] != manifest["seed"]:
        raise RuntimeError("Initial weights differ from E1 configuration")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(SEED)
    rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(rows) != TRAIN_COUNT or ids_hash(rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("Training sample differs from the fixed 1,024 pairs")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("Validation sample differs from the fixed 128 pairs")
    microbatch_size = MICROBATCH[args.mode]
    loader = make_pair_loader(rows, tokenizer, batch_size=microbatch_size,
                              max_length=config.max_length, padding="max_length",
                              num_workers=0, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer,
                                         batch_size=EFFECTIVE_BATCH,
                                         max_length=config.max_length)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)

    microsteps = updates = examples_seen = 0
    loss_sum = 0.0
    seen_ids = set()
    memory_samples = [memory_sample()]
    pending = 0
    group_size = min(EFFECTIVE_BATCH, TRAIN_COUNT)
    optimizer.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    started = time.perf_counter()
    for batch in loader:
        batch_count = len(batch["labels"])
        if pending == 0:
            group_size = min(EFFECTIVE_BATCH, TRAIN_COUNT - examples_seen)
        if batch_count > group_size - pending:
            raise RuntimeError("Microbatch crosses an effective-batch boundary")
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        logits = one_logit_per_label(model(**inputs), labels)
        raw_loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(raw_loss)):
            raise RuntimeError(f"Nonfinite value at microstep {microsteps + 1}")
        memory_samples.append(memory_sample())  # Forward activations are live.
        # BCEWithLogitsLoss averages the microbatch. Weight its contribution by
        # actual examples, including a smaller final effective-batch group.
        (raw_loss * (batch_count / group_size)).backward()
        memory_samples.append(memory_sample())
        pending += batch_count
        examples_seen += batch_count
        microsteps += 1
        loss_sum += float(raw_loss.detach().item()) * batch_count
        if pending == group_size:
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient before update {updates + 1}")
            optimizer.step()
            torch.mps.synchronize()
            memory_samples.append(memory_sample())
            updates += 1
            pending = 0
            optimizer.zero_grad(set_to_none=True)
            if updates % 16 == 0:
                print(f"{args.mode}: update={updates} microsteps={microsteps} "
                      f"seen={examples_seen} mean_loss={loss_sum / examples_seen:.4f}",
                      flush=True)
    torch.mps.synchronize()
    training_seconds = time.perf_counter() - started
    expected_updates = (TRAIN_COUNT + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH
    expected_microsteps = (TRAIN_COUNT + microbatch_size - 1) // microbatch_size
    if examples_seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or pending or \
            updates != expected_updates or microsteps != expected_microsteps:
        raise RuntimeError("E5 batch accounting differs from the intended exposure")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    atomic_torch_save({
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "update_count": updates,
        "microstep_count": microsteps,
        "mode": args.mode,
    }, checkpoint)
    report = {
        "experiment": "E5 gradient accumulation",
        "mode": args.mode,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "initial_checkpoint_sha256": sha256(initial_path),
        "source_config_sha256": sha256(config_path),
        "effective_config": asdict(config),
        "dropout_note": "Dropout set to zero in both training runs to isolate gradient accumulation; no weights changed before training",
        "padding": "fixed to 128 tokens for both runs",
        "seed": SEED,
        "training_pair_ids_sha256": ids_hash(rows),
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "train_pairs": TRAIN_COUNT,
        "microbatch_size": microbatch_size,
        "effective_batch_size": EFFECTIVE_BATCH,
        "microsteps": microsteps,
        "optimizer_updates": updates,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "mean_training_bce_loss": loss_sum / examples_seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "training_time": {
            "seconds": training_seconds,
            "scope": "one epoch: main-process tokenization, device transfer, forward, backward, optimizer; excludes loading, validation, checkpoint save",
            "seconds_per_optimizer_update": training_seconds / updates,
            "mps_synchronized_after_each_optimizer_update": True,
        },
        "memory": {
            "sampled_max_current_tensor_bytes": max(row["tensor_bytes"] for row in memory_samples),
            "sampled_max_current_driver_bytes": max(row["driver_bytes"] for row in memory_samples),
            "measurement": "maximum of MPS current allocation samples with activations live, after backward, and after each update; not a true peak",
        },
        "backend": "mps",
        "precision": "float32",
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"{args.mode}: {updates} updates, {microsteps} microsteps in "
          f"{training_seconds:.2f}s; sampled MPS tensor allocation "
          f"{report['memory']['sampled_max_current_tensor_bytes']:,} bytes; "
          f"validation BCE {before_loss:.4f} -> {after_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
