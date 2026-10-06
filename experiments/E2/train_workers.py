"""Train the same 4-layer scratch model with a chosen loader worker count."""

import argparse
import hashlib
import json
import resource
import statistics
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_torch_save, atomic_write_text
from scripts.collate import make_pair_loader
from scripts.data import iter_clean_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import load_scratch_tokenizer
from scripts.short_epoch import BATCH_SIZE, TRAIN_COUNT, VALIDATION_COUNT, ids_hash, measure
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "experiments" / "E1" / "four_layer"
HERE = Path(__file__).resolve().parent
TRAIN_SEED = 123


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def mps_memory(device: torch.device) -> dict | None:
    if device.type != "mps":
        return None
    return {
        "current_tensor_bytes": torch.mps.current_allocated_memory(),
        "current_driver_bytes": torch.mps.driver_allocated_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    checkpoint = HERE / f"workers_{args.num_workers}.pt"
    report_path = HERE / f"workers_{args.num_workers}.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"E2 output for {args.num_workers} workers already exists")

    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    initial_manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != initial_manifest["config_sha256"] or \
            sha256(initial_path) != initial_manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 initialization files differ from their manifest")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config) or initial["seed"] != initial_manifest["seed"]:
        raise RuntimeError("Initial weights do not match the 4-layer configuration")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("E2 comparisons on this Mac require MPS; CPU timing is a separate run")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(TRAIN_SEED)

    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(train_rows) != TRAIN_COUNT or ids_hash(train_rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("E2 training sample differs from E1")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("E2 validation sample differs from E1")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, shuffle=False,
                                    num_workers=args.num_workers)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)

    seen_ids = set()
    seen = 0
    loss_sum = 0.0
    waits = []
    compute_times = []
    memory_samples = [mps_memory(device)]
    synchronize(device)
    started = time.perf_counter()
    iterator = iter(train_loader)
    startup_seconds = time.perf_counter() - started
    for step in range(1, TRAIN_COUNT // BATCH_SIZE + 1):
        wait_started = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - wait_started)
        compute_started = time.perf_counter()
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training value at step {step}")
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        synchronize(device)
        compute_times.append(time.perf_counter() - compute_started)
        seen += len(labels)
        loss_sum += float(loss.detach().item()) * len(labels)
        memory_samples.append(mps_memory(device))
        if step % 16 == 0:
            print(f"workers={args.num_workers}: step={step} seen={seen} "
                  f"mean_train_loss={loss_sum / seen:.4f}", flush=True)
    total_seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT:
        raise RuntimeError("E2 epoch did not use every selected pair exactly once")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("E2 loader yielded more pairs than expected")
    del iterator
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "completed_epochs": 1,
        "update_count": step,
        "training_seed": TRAIN_SEED,
        "num_workers": args.num_workers,
    }
    atomic_torch_save(payload, checkpoint)
    report = {
        "experiment": "E2 loader workers",
        "num_workers": args.num_workers,
        "model": "four_layer scratch encoder",
        "initial_checkpoint": str(initial_path.relative_to(ROOT)),
        "initial_checkpoint_sha256": sha256(initial_path),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "training_seed": TRAIN_SEED,
        "training_pairs": TRAIN_COUNT,
        "training_pair_ids_sha256": ids_hash(train_rows),
        "validation_pairs": VALIDATION_COUNT,
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "batch_size": BATCH_SIZE,
        "max_length": config.max_length,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "optimizer_updates": step,
        "mean_training_bce_loss": loss_sum / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "timing": {
            "scope": "64 training batches; excludes model/tokenizer loading, data selection, validation, checkpoint save",
            "loader_startup_seconds": startup_seconds,
            "batch_wait_total_seconds": sum(waits),
            "batch_wait_mean_seconds": statistics.mean(waits),
            "batch_wait_median_seconds": statistics.median(waits),
            "batch_wait_max_seconds": max(waits),
            "training_compute_total_seconds": sum(compute_times),
            "total_seconds": total_seconds,
            "pairs_per_second": TRAIN_COUNT / total_seconds,
            "mps_synchronized_after_each_update": True,
        },
        "memory": {
            "main_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "main_process_peak_rss_note": "macOS ru_maxrss high-water mark; excludes worker processes",
            "mps_sampled_max_tensor_bytes": max(row["current_tensor_bytes"] for row in memory_samples),
            "mps_sampled_max_driver_bytes": max(row["current_driver_bytes"] for row in memory_samples),
            "mps_measurement": "maximum current-allocation snapshots at epoch start and after each update; not true peak",
        },
        "backend": device.type,
        "precision": "float32",
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"workers={args.num_workers}: {TRAIN_COUNT} pairs in {total_seconds:.2f}s "
          f"({TRAIN_COUNT / total_seconds:.1f} pairs/s); batch wait {sum(waits):.2f}s; "
          f"validation BCE {before_loss:.4f} -> {after_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
