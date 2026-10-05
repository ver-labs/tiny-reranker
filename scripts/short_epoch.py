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


def load_resume_state():
    """Restore the separate run's model, optimizer, counters, and RNG state."""
    manifest = json.loads(RESUME_MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(RESUME_CHECKPOINT.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Resume checkpoint differs from its manifest")
    saved = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=True)
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
    return model, optimizer, saved["completed_epochs"], saved["update_count"], device


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


def save_resume_sidecar(model, optimizer, device, step, train_rows, validation_rows) -> None:
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
    numpy_state = np.random.get_state()
    payload = {
        "model_state_dict": weights,
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epochs": 1,
        "update_count": step,
        "configuration": {
            "model": asdict(model.config),
            "tokenizer_id": source_config["tokenizer_id"],
            "tokenizer_revision": source_config["tokenizer_revision"],
            "optimizer": "AdamW",
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
        "rng_state": {
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
        },
    }
    temporary = RESUME_CHECKPOINT.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(RESUME_CHECKPOINT)
    loaded = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=True)
    if loaded["completed_epochs"] != 1 or loaded["update_count"] != step or \
            len(loaded["optimizer_state_dict"]["state"]) != len(list(model.parameters())):
        raise RuntimeError("Resume checkpoint lost epoch, step, or optimizer state")
    manifest = {
        "checkpoint": RESUME_CHECKPOINT.name,
        "sha256": hashlib.sha256(RESUME_CHECKPOINT.read_bytes()).hexdigest(),
        "compared_with_evaluated_model_checkpoint": CHECKPOINT.name,
        "evaluated_model_checkpoint_sha256": report["checkpoint_sha256"],
        "matches_evaluated_model_weights_exactly": exact_match,
        "max_model_weight_difference": max_weight_difference,
        "completed_epochs": 1,
        "update_count": step,
        "rng_backends": ["python", "numpy", "torch_cpu"] +
                        (["torch_mps"] if device.type == "mps" else []),
    }
    RESUME_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved and reloaded resume checkpoint: {RESUME_CHECKPOINT}", flush=True)
    print(f"Exact match with earlier evaluated model: {exact_match}; "
          f"max weight difference: {max_weight_difference:.9g}", flush=True)
    print("Optimizer, configuration, epoch, update count, and RNG states are present.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-resume", action="store_true",
                        help="Repeat the short epoch and save a separate resume-ready checkpoint")
    args = parser.parse_args()
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
    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save({"model_state_dict": model.cpu().state_dict(), "optimizer_steps": step}, temporary)
    temporary.replace(CHECKPOINT)
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
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved trial checkpoint: {CHECKPOINT}", flush=True)
    print(f"Saved report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
