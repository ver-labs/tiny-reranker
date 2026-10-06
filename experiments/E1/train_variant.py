"""Train one E1 variant for one bounded epoch with matched data exposure."""

import argparse
import hashlib
import json
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
from scripts.short_epoch import (
    BATCH_SIZE, TRAIN_COUNT, VALIDATION_COUNT, capture_rng_state,
    ids_hash, measure, split_manifest_record,
)
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


EXPERIMENT = Path(__file__).resolve().parent
VARIANTS = ("two_layer", "four_layer")
TRAIN_SEED = 123


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict | None:
    if device.type != "mps":
        return None
    return {
        "current_tensor_bytes": torch.mps.current_allocated_memory(),
        "current_driver_bytes": torch.mps.driver_allocated_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    folder = EXPERIMENT / args.variant
    checkpoint = folder / "epoch1.pt"
    report_path = folder / "epoch1.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"{args.variant} epoch-1 output already exists")
    initial = folder / "initial.pt"
    initial_manifest = json.loads((folder / "initial.json").read_text(encoding="utf-8"))
    if sha256(initial) != initial_manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("Initial checkpoint differs from its manifest")
    config_path = folder / "config.json"
    if sha256(config_path) != initial_manifest["config_sha256"]:
        raise RuntimeError("Model config differs from its setup manifest")
    config = ScratchConfig.from_file(config_path)
    saved_initial = torch.load(initial, map_location="cpu", weights_only=True)
    if saved_initial["config"] != asdict(config) or saved_initial["seed"] != initial_manifest["seed"]:
        raise RuntimeError("Initial checkpoint does not match the selected E1 config")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ScratchReranker(config)
    model.load_state_dict(saved_initial["model_state_dict"], strict=True)
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(TRAIN_SEED)
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((EXPERIMENT.parents[1] / "results" / "short_epoch_1024.json").read_text())
    if len(train_rows) != TRAIN_COUNT or ids_hash(train_rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("E1 training sample differs from the prepared 1,024 pairs")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("E1 validation sample differs from the prepared 128 pairs")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    memory_samples = [memory_snapshot(device)] if device.type == "mps" else []
    seen = 0
    seen_ids = set()
    total_loss = 0.0
    synchronize(device)
    started = time.perf_counter()
    for step, batch in enumerate(train_loader, start=1):
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite E1 training value at step {step}")
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise RuntimeError(f"Nonfinite E1 gradient at step {step}")
        optimizer.step()
        seen += len(labels)
        total_loss += float(loss.detach().item()) * len(labels)
        if device.type == "mps":
            memory_samples.append(memory_snapshot(device))
        if step % 16 == 0:
            print(f"{args.variant}: step={step} seen={seen} "
                  f"mean_train_loss={total_loss / seen:.4f}", flush=True)
    synchronize(device)
    train_seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or step != TRAIN_COUNT // BATCH_SIZE:
        raise RuntimeError("E1 epoch did not use each selected pair exactly once")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "completed_epochs": 1,
        "update_count": step,
        "training_seed": TRAIN_SEED,
        "rng_state": capture_rng_state(device),
    }
    atomic_torch_save(payload, checkpoint)
    report = {
        "variant": args.variant,
        "trained": True,
        "initial_checkpoint_sha256": initial_manifest["initial_checkpoint_sha256"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "parameter_count": initial_manifest["trainable_parameter_count"],
        "training_seed": TRAIN_SEED,
        "training_pairs": TRAIN_COUNT,
        "training_pair_ids_sha256": ids_hash(train_rows),
        "validation_pairs": VALIDATION_COUNT,
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "split_manifest": split_manifest_record(),
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "optimizer_updates": step,
        "mean_training_bce_loss": total_loss / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "training_time": {
            "seconds": train_seconds,
            "scope": "64 batches: tokenization, device transfer, forward, backward, optimizer; "
                     "excludes model/tokenizer loading, data selection, validation, and checkpoint save",
            "mps_synchronized_at_start_and_end": device.type == "mps",
        },
        "memory": None if not memory_samples else {
            "measurement": "maximum of MPS current-allocation snapshots before and after each optimizer step; not a true peak",
            "sampled_max_tensor_bytes": max(row["current_tensor_bytes"] for row in memory_samples),
            "sampled_max_driver_bytes": max(row["current_driver_bytes"] for row in memory_samples),
        },
        "backend": device.type,
        "precision": "float32",
        "full_ranking_evaluation_completed": False,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"{args.variant}: epoch complete in {train_seconds:.2f}s; "
          f"validation BCE {before_loss:.4f} -> {after_loss:.4f}; "
          f"saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
