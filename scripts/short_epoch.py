"""Train once over 1,024 fixed pairs, then check a held-out validation sample."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .collate import make_pair_loader
from .checkpoint_io import atomic_torch_save, atomic_write_text
from .data import iter_clean_pairs
from .model import ScratchConfig, ScratchReranker
from .scratch_tokenizer import load_scratch_tokenizer
from .train import LEARNING_RATE, load_training_model, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parent.parent
TRAIN_COUNT = 1024
VALIDATION_COUNT = 128
BATCH_SIZE = 16
CHECKPOINT = ROOT / "checkpoints" / "short_epoch_1024.pt"
REPORT = ROOT / "results" / "short_epoch_1024.json"
RESUME_CHECKPOINT = ROOT / "checkpoints" / "short_epoch_1024_resume.pt"
RESUME_MANIFEST = RESUME_CHECKPOINT.with_suffix(".json")
SPLIT_MANIFEST = ROOT / "data" / "manifests" / "query_split.json"


def split_manifest_record() -> dict[str, str]:
    """Identify the exact query-to-split rule used for this run."""
    return {
        "path": str(SPLIT_MANIFEST.relative_to(ROOT)),
        "sha256": hashlib.sha256(SPLIT_MANIFEST.read_bytes()).hexdigest(),
    }


def epoch_boundary_sampler_state(completed_epochs: int = 1) -> dict[str, int | str | bool]:
    """Record the completed sequential pass and where the next epoch starts."""
    return {
        "order": "sequential_selected_pair_ids",
        "selected_examples": TRAIN_COUNT,
        "consumed_in_completed_epoch": TRAIN_COUNT,
        "next_epoch_index_zero_based": completed_epochs,
        "next_sample_index": 0,
        "mid_epoch_resume_supported": False,
    }


def capture_rng_state(device: torch.device) -> dict:
    """Capture all RNG streams used by this training environment."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": torch.mps.get_rng_state() if device.type == "mps" else None,
    }


def load_resume_state(checkpoint_path: Path = RESUME_CHECKPOINT,
                      *, scheduler_factory=None, scaler_factory=None):
    """Restore model, optimizer, optional components, counters, and RNG state."""
    manifest_path = checkpoint_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Resume checkpoint differs from its manifest")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = saved["configuration"]
    source_config = json.loads((ROOT / "configs" / "scratch.json").read_text(encoding="utf-8"))
    if (settings["tokenizer_id"], settings["tokenizer_revision"]) != \
            (source_config["tokenizer_id"], source_config["tokenizer_revision"]):
        raise RuntimeError("Tokenizer selection differs from the resume checkpoint")
    if settings["train_pairs"] != TRAIN_COUNT or settings["batch_size"] != BATCH_SIZE:
        raise RuntimeError("Resume checkpoint training settings differ from this script")
    if settings["train_pair_ids_sha256"] != ids_hash(
            list(islice(iter_clean_pairs("train"), TRAIN_COUNT))):
        raise RuntimeError("Prepared training sample differs from the resume checkpoint")
    if settings["validation_pair_ids_sha256"] != ids_hash(
            list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))):
        raise RuntimeError("Validation sample differs from the resume checkpoint")
    if saved["completed_epochs"] != manifest["completed_epochs"] or \
            saved["update_count"] != manifest["update_count"]:
        raise RuntimeError("Resume counters differ from the manifest")
    if saved["data_split_manifest"] != split_manifest_record() or \
            manifest["data_split_manifest"] != split_manifest_record():
        raise RuntimeError("Data split manifest differs from the resume checkpoint")
    if saved["sampler_state"] != epoch_boundary_sampler_state(saved["completed_epochs"]) or \
            manifest["sampler_state"] != saved["sampler_state"]:
        raise RuntimeError("Resume sampler position is not the expected epoch boundary")
    config = ScratchConfig(**settings["model"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if saved["rng_state"]["torch_mps"] is not None and device.type != "mps":
        raise RuntimeError("MPS is required to restore this run's RNG state")
    model = ScratchReranker(config).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  betas=tuple(settings["betas"]), eps=settings["eps"],
                                  weight_decay=settings["weight_decay"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scheduler_state = saved.get("scheduler_state_dict")
    scaler_state = saved.get("gradient_scaler_state_dict")
    if scheduler_state is not None and scheduler_factory is None:
        raise RuntimeError("A scheduler factory is required to restore its saved state")
    if scaler_state is not None and scaler_factory is None:
        raise RuntimeError("A gradient-scaler factory is required to restore its saved state")
    scheduler = scheduler_factory(optimizer) if scheduler_state is not None else None
    scaler = scaler_factory() if scaler_state is not None else None
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler.load_state_dict(scaler_state)
    rng = saved["rng_state"]
    random.setstate(rng["python"])
    numpy_rng = rng["numpy"]
    np.random.set_state((numpy_rng["algorithm"],
                         np.array(numpy_rng["keys"], dtype=np.uint32),
                         numpy_rng["position"], numpy_rng["has_gauss"],
                         numpy_rng["cached_gaussian"]))
    torch.set_rng_state(rng["torch_cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(rng["torch_mps"])
    model.train()
    return (model, optimizer, saved["completed_epochs"], saved["update_count"],
            device, scheduler, scaler)


def ids_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(row["pair_id"] for row in rows).encode()).hexdigest()


def measure(model, loader, device, loss_fn) -> tuple[float, float]:
    """Measure held-out pair loss and binary accuracy without gradients."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs, labels = move_batch_to_device(batch, device)
                logits = one_logit_per_label(model(**inputs), labels)
                loss = loss_fn(logits, labels)
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite validation score or loss")
                total_loss += float(loss.item()) * len(labels)
                total_correct += int(((logits >= 0) == (labels == 1)).sum().item())
                total += len(labels)
        return total_loss / total, total_correct / total
    finally:
        model.train(was_training)


def save_resume_sidecar(model, optimizer, device, step, train_rows, validation_rows,
                        *, scheduler=None, scaler=None) -> None:
    """Save a separate run with all state needed to continue it."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise RuntimeError("Existing trial checkpoint differs from its report")
    original = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    exact_match = all(torch.equal(value, original["model_state_dict"][name])
                      for name, value in weights.items())
    max_weight_difference = max(
        float((value - original["model_state_dict"][name]).abs().max().item())
        for name, value in weights.items()
    )
    if report["train_pair_ids_sha256"] != ids_hash(train_rows) or \
            report["validation_pair_ids_sha256"] != ids_hash(validation_rows):
        raise RuntimeError("Prepared samples differ from the evaluated run")
    source_config = json.loads((ROOT / "configs" / "scratch.json").read_text(encoding="utf-8"))
    scaler_enabled = scaler is not None and (not hasattr(scaler, "is_enabled") or scaler.is_enabled())
    payload = {
        "model_state_dict": weights,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "gradient_scaler_state_dict": scaler.state_dict() if scaler_enabled else None,
        "completed_epochs": 1,
        "update_count": step,
        "data_split_manifest": split_manifest_record(),
        "sampler_state": epoch_boundary_sampler_state(),
        "configuration": {
            "model": asdict(model.config),
            "tokenizer_id": source_config["tokenizer_id"],
            "tokenizer_revision": source_config["tokenizer_revision"],
            "optimizer": "AdamW",
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "gradient_scaler": type(scaler).__name__ if scaler_enabled else None,
            "learning_rate": LEARNING_RATE,
            "betas": list(optimizer.defaults["betas"]),
            "eps": optimizer.defaults["eps"],
            "weight_decay": optimizer.defaults["weight_decay"],
            "batch_size": BATCH_SIZE,
            "train_pairs": TRAIN_COUNT,
            "train_pair_ids_sha256": ids_hash(train_rows),
            "validation_pair_ids_sha256": ids_hash(validation_rows),
            "shuffle": False,
        },
        "rng_state": capture_rng_state(device),
    }
    atomic_torch_save(payload, RESUME_CHECKPOINT)
    loaded = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=True)
    if loaded["completed_epochs"] != 1 or loaded["update_count"] != step or \
            len(loaded["optimizer_state_dict"]["state"]) != len(list(model.parameters())):
        raise RuntimeError("Resume checkpoint lost epoch, step, or optimizer state")
    if (scheduler is not None and loaded["scheduler_state_dict"] is None) or \
            (scaler_enabled and loaded["gradient_scaler_state_dict"] is None):
        raise RuntimeError("Resume checkpoint lost an enabled component's state")
    manifest = {
        "checkpoint": RESUME_CHECKPOINT.name,
        "sha256": hashlib.sha256(RESUME_CHECKPOINT.read_bytes()).hexdigest(),
        "compared_with_evaluated_model_checkpoint": CHECKPOINT.name,
        "evaluated_model_checkpoint_sha256": report["checkpoint_sha256"],
        "matches_evaluated_model_weights_exactly": exact_match,
        "max_model_weight_difference": max_weight_difference,
        "completed_epochs": 1,
        "update_count": step,
        "data_split_manifest": payload["data_split_manifest"],
        "sampler_state": payload["sampler_state"],
        "scheduler_enabled": scheduler is not None,
        "gradient_scaler_enabled": scaler_enabled,
        "rng_backends": ["python", "numpy", "torch_cpu"] +
                        (["torch_mps"] if device.type == "mps" else []),
    }
    atomic_write_text(json.dumps(manifest, indent=2) + "\n", RESUME_MANIFEST)
    print(f"Saved and reloaded resume checkpoint: {RESUME_CHECKPOINT}", flush=True)
    print(f"Exact match with earlier evaluated model: {exact_match}; "
          f"max weight difference: {max_weight_difference:.9g}", flush=True)
    print("Optimizer, configuration, epoch, update count, and RNG states are present.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-resume", action="store_true",
                        help="Repeat the short epoch and save a separate resume-ready checkpoint")
    args = parser.parse_args()
    if args.save_resume and RESUME_CHECKPOINT.exists():
        parser.error("resume checkpoint already exists; continue with scripts.continue_epochs")
    if not args.save_resume and CHECKPOINT.exists():
        parser.error("trial checkpoint already exists; continue with scripts.continue_epochs")
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    model, device = load_training_model()
    tokenizer = load_scratch_tokenizer()
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    if len(train_rows) != TRAIN_COUNT or len(validation_rows) != VALIDATION_COUNT:
        raise RuntimeError("Prepared sample has too few rows")
    if {row["query_id"] for row in train_rows} & {row["query_id"] for row in validation_rows}:
        raise RuntimeError("Training and validation query IDs overlap")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=model.config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=model.config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    print(f"Device: {device}; training pairs: {TRAIN_COUNT}; validation pairs: "
          f"{VALIDATION_COUNT}; batch size: {BATCH_SIZE}", flush=True)
    print(f"Before: validation_loss={before_loss:.4f} "
          f"validation_accuracy={before_accuracy:.1%}", flush=True)

    total_train_loss = 0.0
    seen = 0
    seen_ids = set()
    for step, batch in enumerate(train_loader, start=1):
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training score or loss at step {step}")
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        total_train_loss += float(loss.detach().item()) * len(labels)
        seen += len(labels)
        if step % 10 == 0 or seen == TRAIN_COUNT:
            print(f"step={step} seen={seen} train_loss_so_far="
                  f"{total_train_loss / seen:.4f}", flush=True)
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT:
        raise RuntimeError("Epoch did not use every selected pair exactly once")

    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    assert model.training
    print(f"After: validation_loss={after_loss:.4f} "
          f"validation_accuracy={after_accuracy:.1%}", flush=True)
    print(f"One epoch completed: {seen} unique pairs in {step} optimizer steps; "
          f"mean training loss={total_train_loss / seen:.4f}", flush=True)

    if args.save_resume:
        save_resume_sidecar(model, optimizer, device, step, train_rows, validation_rows)
        return

    CHECKPOINT.parent.mkdir(exist_ok=True)
    atomic_torch_save({"model_state_dict": model.cpu().state_dict(), "optimizer_steps": step},
                      CHECKPOINT)
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    report = {
        "purpose": "one bounded training epoch; pair-level smoke check, not ranking evaluation",
        "device": str(device),
        "train_pairs": TRAIN_COUNT,
        "train_pair_ids_sha256": ids_hash(train_rows),
        "validation_pairs": VALIDATION_COUNT,
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer_steps": step,
        "mean_training_loss": total_train_loss / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": checkpoint_hash,
    }
    REPORT.parent.mkdir(exist_ok=True)
    atomic_write_text(json.dumps(report, indent=2) + "\n", REPORT)
    print(f"Saved trial checkpoint: {CHECKPOINT}", flush=True)
    print(f"Saved report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
