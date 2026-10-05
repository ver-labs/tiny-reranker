"""Train once over 1,024 fixed pairs, then check a held-out validation sample."""

import hashlib
import json
from itertools import islice
from pathlib import Path

import torch

from .collate import make_pair_loader
from .data import iter_clean_pairs
from .scratch_tokenizer import load_scratch_tokenizer
from .train import LEARNING_RATE, load_training_model, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parent.parent
TRAIN_COUNT = 1024
VALIDATION_COUNT = 128
BATCH_SIZE = 16
CHECKPOINT = ROOT / "checkpoints" / "short_epoch_1024.pt"
REPORT = ROOT / "results" / "short_epoch_1024.json"


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


def main() -> None:
    torch.manual_seed(123)
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
