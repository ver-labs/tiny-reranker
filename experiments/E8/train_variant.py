"""Train the same 4-layer model with activation checkpointing off or on."""

import argparse
import hashlib
import json
import os
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
from scripts.short_epoch import (BATCH_SIZE, TRAIN_COUNT, VALIDATION_COUNT,
                                 capture_rng_state, ids_hash, measure,
                                 split_manifest_record)
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E1 = ROOT / "experiments" / "E1" / "four_layer"
SEED = 123


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_sample() -> dict[str, int]:
    return {"tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpointing", choices=("off", "on"), required=True)
    args = parser.parse_args()
    checkpoint = HERE / f"{args.checkpointing}.pt"
    report_path = HERE / f"{args.checkpointing}.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"E8 {args.checkpointing} output already exists")
    if not torch.backends.mps.is_available() or \
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS without CPU fallback is required")
    device = torch.device("mps")
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"] or \
            sha256(initial_path) != manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 initial model differs from its manifest")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config) or initial["seed"] != manifest["seed"]:
        raise RuntimeError("Initial checkpoint has unexpected config or seed")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model.activation_checkpointing = args.checkpointing == "on"
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(SEED)
    rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(rows) != TRAIN_COUNT or ids_hash(rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("E8 training sample differs from fixed E1 sample")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("E8 validation sample differs from fixed E1 sample")
    train_loader = make_pair_loader(rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer,
                                         batch_size=BATCH_SIZE,
                                         max_length=config.max_length)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    seen = updates = 0
    seen_ids = set()
    loss_sum = 0.0
    memory_samples = [memory_sample()]
    torch.mps.synchronize()
    started = time.perf_counter()
    for batch in train_loader:
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite E8 forward/loss at update {updates + 1}")
        memory_samples.append(memory_sample())  # Forward graph is live.
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise RuntimeError(f"Missing or nonfinite E8 gradient at update {updates + 1}")
        memory_samples.append(memory_sample())
        optimizer.step()
        torch.mps.synchronize()
        memory_samples.append(memory_sample())
        updates += 1
        seen += len(labels)
        loss_sum += float(loss.detach().item()) * len(labels)
        if updates % 16 == 0:
            print(f"checkpointing={args.checkpointing}: updates={updates} "
                  f"seen={seen} mean_loss={loss_sum / seen:.4f}", flush=True)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or \
            updates != TRAIN_COUNT // BATCH_SIZE:
        raise RuntimeError("E8 data exposure or update count differs from reference")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    atomic_torch_save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "completed_epochs": 1,
        "update_count": updates,
        "training_seed": SEED,
        "activation_checkpointing": model.activation_checkpointing,
        "rng_state": capture_rng_state(device),
        "data_split_manifest": split_manifest_record(),
    }, checkpoint)
    report = {
        "experiment": "E8 four-layer activation checkpointing",
        "activation_checkpointing": args.checkpointing,
        "implementation": "torch.utils.checkpoint.checkpoint on each encoder layer, use_reentrant=False, preserve_rng_state=True" if model.activation_checkpointing else "ordinary encoder forward",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "initial_checkpoint_sha256": sha256(initial_path),
        "config_sha256": sha256(config_path),
        "training_seed": SEED,
        "training_pairs": TRAIN_COUNT,
        "training_pair_ids_sha256": ids_hash(rows),
        "validation_pairs": VALIDATION_COUNT,
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "batch_size": BATCH_SIZE,
        "max_length": config.max_length,
        "padding": "dynamic per batch",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "optimizer_updates": updates,
        "mean_training_bce_loss": loss_sum / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "backend": "mps",
        "precision": "float32",
        "training_time": {
            "seconds": seconds,
            "scope": "64 batches: tokenization, device transfer, FP32 forward/backward, gradient check, optimizer; excludes loading, validation, save",
            "mps_synchronized_after_each_update": True,
        },
        "memory": {
            "sampled_max_current_tensor_bytes": max(row["tensor_bytes"] for row in memory_samples),
            "sampled_max_current_driver_bytes": max(row["driver_bytes"] for row in memory_samples),
            "measurement": "maximum sampled MPS current allocation with forward graph live, after backward, after update; not true peak",
        },
        "full_ranking_evaluation_completed": False,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"checkpointing={args.checkpointing}: {seconds:.2f}s, sampled tensor "
          f"{report['memory']['sampled_max_current_tensor_bytes']:,} bytes, "
          f"validation BCE {before_loss:.4f} -> {after_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
