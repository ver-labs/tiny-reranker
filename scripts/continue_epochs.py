"""Continue the bounded training run, saving after every complete epoch."""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import torch

from .collate import make_pair_loader
from .data import iter_clean_pairs
from .scratch_tokenizer import load_scratch_tokenizer
from .short_epoch import (
    BATCH_SIZE, RESUME_CHECKPOINT, ROOT, TRAIN_COUNT, VALIDATION_COUNT,
    capture_rng_state, epoch_boundary_sampler_state, ids_hash,
    load_resume_state, measure,
)
from .train import move_batch_to_device, one_logit_per_label


def latest_checkpoint() -> Path:
    later = sorted((ROOT / "checkpoints").glob("short_epoch_1024_epoch_*.pt"))
    return later[-1] if later else RESUME_CHECKPOINT


def save_epoch(checkpoint: Path, parent: Path, model, optimizer, device,
               epoch: int, updates: int, validation_loss: float,
               validation_accuracy: float) -> None:
    """Write one immutable resume checkpoint only after the full epoch."""
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(f"Epoch checkpoint already exists: {checkpoint}")
    previous = torch.load(parent, map_location="cpu", weights_only=True)
    if previous.get("scheduler_state_dict") is not None or \
            previous.get("gradient_scaler_state_dict") is not None:
        raise RuntimeError("This training loop needs scheduler/scaler setup before continuing")
    payload = dict(previous)
    payload["model_state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["completed_epochs"] = epoch
    payload["update_count"] = updates
    payload["sampler_state"] = epoch_boundary_sampler_state(epoch)
    payload["rng_state"] = capture_rng_state(device)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    parent_manifest = json.loads(parent.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = {
        "checkpoint": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parent_checkpoint": parent.name,
        "parent_checkpoint_sha256": parent_manifest["sha256"],
        "completed_epochs": epoch,
        "update_count": updates,
        "data_split_manifest": payload["data_split_manifest"],
        "sampler_state": payload["sampler_state"],
        "scheduler_enabled": False,
        "gradient_scaler_enabled": False,
        "validation_pair_bce_loss": validation_loss,
        "validation_pair_accuracy": validation_accuracy,
    }
    checkpoint.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of additional complete epochs (default: 1)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    parent = latest_checkpoint()
    model, optimizer, completed, updates, device, scheduler, scaler = load_resume_state(parent)
    if scheduler is not None or scaler is not None:
        raise RuntimeError("This training loop has no scheduler or gradient scaler")
    tokenizer = load_scratch_tokenizer()
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    if len(train_rows) != TRAIN_COUNT or len(validation_rows) != VALIDATION_COUNT:
        raise RuntimeError("Prepared sample has too few rows")
    saved = torch.load(parent, map_location="cpu", weights_only=True)
    if ids_hash(train_rows) != saved["configuration"]["train_pair_ids_sha256"] or \
            ids_hash(validation_rows) != saved["configuration"]["validation_pair_ids_sha256"]:
        raise RuntimeError("Selected examples differ from the resume checkpoint")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=model.config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=model.config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"Continuing from {parent.name} on {device}: {completed} completed epochs, "
          f"{updates} updates", flush=True)
    for epoch in range(completed + 1, completed + args.epochs + 1):
        model.train()
        seen_ids = set()
        seen = 0
        loss_total = 0.0
        for batch in train_loader:
            seen_ids.update(batch["pair_ids"])
            inputs, labels = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = one_logit_per_label(model(**inputs), labels)
            loss = loss_fn(logits, labels)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training value at update {updates + 1}")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient at update {updates + 1}")
            optimizer.step()
            updates += 1
            seen += len(labels)
            loss_total += float(loss.detach().item()) * len(labels)
        if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT:
            raise RuntimeError("Epoch did not use every selected pair exactly once")
        validation_loss, validation_accuracy = measure(model, validation_loader, device, loss_fn)
        checkpoint = ROOT / "checkpoints" / f"short_epoch_1024_epoch_{epoch:04d}.pt"
        save_epoch(checkpoint, parent, model, optimizer, device, epoch, updates,
                   validation_loss, validation_accuracy)
        print(f"epoch={epoch} updates={updates} train_loss={loss_total / seen:.4f} "
              f"validation_loss={validation_loss:.4f} "
              f"validation_accuracy={validation_accuracy:.1%} "
              f"saved={checkpoint.name}", flush=True)
        parent = checkpoint


if __name__ == "__main__":
    main()
