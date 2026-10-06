"""Train only the scoring head on new training queries for one E1 variant."""

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
from scripts.scratch_tokenizer import load_scratch_tokenizer
from scripts.short_epoch import BATCH_SIZE, TRAIN_COUNT, VALIDATION_COUNT, ids_hash, measure
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label

from .evaluate_variant import TrainedVariant, synchronize


EXPERIMENT = Path(__file__).resolve().parent
VARIANTS = ("two_layer", "four_layer")
HEAD_SEED = 456


def head_only_rows() -> tuple[list[dict], list[dict]]:
    """Select new training pairs whose queries were absent from E1 training."""
    source = iter_clean_pairs("train")
    original = list(islice(source, TRAIN_COUNT))
    old_queries = {row["query_id"] for row in original}
    rows = []
    for row in source:
        if row["query_id"] not in old_queries:
            rows.append(row)
            if len(rows) == TRAIN_COUNT:
                break
    if len(original) != TRAIN_COUNT or len(rows) != TRAIN_COUNT or \
            old_queries & {row["query_id"] for row in rows}:
        raise RuntimeError("Could not select 1,024 new training pairs from disjoint queries")
    if {row["label"] for row in rows} != {0, 1}:
        raise RuntimeError("New subset needs both relevance labels")
    return original, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    folder = EXPERIMENT / args.variant
    checkpoint = folder / "head_only.pt"
    report_path = folder / "head_only.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"{args.variant} head-only output already exists")
    parent_report = json.loads((folder / "epoch1.json").read_text(encoding="utf-8"))
    reranker = TrainedVariant(folder, parent_report)
    model, device = reranker.model, reranker.device
    tokenizer = load_scratch_tokenizer(folder / "config.json")
    original_rows, rows = head_only_rows()
    if ids_hash(original_rows) != parent_report["training_pair_ids_sha256"]:
        raise RuntimeError("E1 original training subset has changed")
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != parent_report["validation_pair_ids_sha256"]:
        raise RuntimeError("Validation subset differs from the E1 run")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.scoring_head.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if trainable != model.config.hidden_size + 1 or trainable_names != \
            ["scoring_head.weight", "scoring_head.bias"]:
        raise RuntimeError("Expected only the linear scoring head to be trainable")
    # Keep the frozen encoder's dropout disabled while still tracking head gradients.
    model.eval()
    backbone_before = {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()
                       if not name.startswith("scoring_head.")}
    head_before = {name: value.detach().cpu().clone()
                   for name, value in model.scoring_head.state_dict().items()}
    torch.manual_seed(HEAD_SEED)
    train_loader = make_pair_loader(rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=model.config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=model.config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.scoring_head.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    seen = 0
    total_loss = 0.0
    seen_ids = set()
    synchronize(device)
    started = time.perf_counter()
    for step, batch in enumerate(train_loader, start=1):
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite head-only score or loss at step {step}")
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.scoring_head.parameters()):
            raise RuntimeError(f"Nonfinite head gradient at step {step}")
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(labels)
        seen += len(labels)
    synchronize(device)
    seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or step != TRAIN_COUNT // BATCH_SIZE:
        raise RuntimeError("Head-only epoch did not use every selected pair exactly once")
    state = model.state_dict()
    if any(not torch.equal(before, state[name].detach().cpu())
           for name, before in backbone_before.items()):
        raise RuntimeError("A frozen backbone parameter changed")
    if all(torch.equal(before, model.scoring_head.state_dict()[name].detach().cpu())
           for name, before in head_before.items()):
        raise RuntimeError("Scoring head did not update")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    payload = {
        "model_state_dict": {name: value.detach().cpu() for name, value in state.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(model.config),
        "update_count": step,
        "parent_checkpoint_sha256": parent_report["checkpoint_sha256"],
        "head_only_training_pair_ids_sha256": ids_hash(rows),
    }
    atomic_torch_save(payload, checkpoint)
    report = {
        "variant": args.variant,
        "exercise": "freeze trained backbone; update scoring head only",
        "parent_checkpoint": "epoch1.pt",
        "parent_checkpoint_sha256": parent_report["checkpoint_sha256"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": parent_report["config_sha256"],
        "frozen_backbone_unchanged": True,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable,
        "original_training_pair_ids_sha256": ids_hash(original_rows),
        "new_training_pair_ids_sha256": ids_hash(rows),
        "query_overlap_with_original_training": 0,
        "new_training_pairs": TRAIN_COUNT,
        "head_training_seed": HEAD_SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer_updates": step,
        "mean_training_bce_loss": total_loss / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "head_only_training_seconds": seconds,
        "backend": device.type,
        "full_ranking_evaluation_completed": False,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"{args.variant}: updated only {trainable} head parameters in {seconds:.2f}s; "
          f"backbone unchanged; validation BCE {before_loss:.4f} -> {after_loss:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
