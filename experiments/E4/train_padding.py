"""Train the same 4-layer model with fixed or dynamic pair-batch padding."""

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
from scripts.short_epoch import BATCH_SIZE, TRAIN_COUNT, VALIDATION_COUNT, ids_hash, measure
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "experiments" / "E1" / "four_layer"
HERE = Path(__file__).resolve().parent
SEED = 123


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize() -> None:
    torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding", choices=("fixed", "dynamic"), required=True)
    args = parser.parse_args()
    checkpoint = HERE / f"{args.padding}_padding.pt"
    report_path = HERE / f"{args.padding}_padding.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"{args.padding} padding output already exists")
    padding = "max_length" if args.padding == "fixed" else True
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    initial_manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != initial_manifest["config_sha256"] or \
            sha256(initial_path) != initial_manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 starting model differs from its manifest")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config) or initial["seed"] != initial_manifest["seed"]:
        raise RuntimeError("E1 starting weights do not match the 4-layer config")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This Mac comparison requires MPS")
    device = torch.device("mps")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(SEED)
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(train_rows) != TRAIN_COUNT or ids_hash(train_rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("Training pairs differ from the fixed E1 sample")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("Validation pairs differ from the fixed E1 sample")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, padding=padding,
                                    num_workers=0, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=config.max_length)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    content_hash = hashlib.sha256()
    padded_tokens = 0
    real_tokens = 0
    seen_ids = set()
    seen = 0
    loss_sum = 0.0
    batch_fetch_seconds = 0.0
    compute_seconds = 0.0
    mps_tensor_max = torch.mps.current_allocated_memory()
    mps_driver_max = torch.mps.driver_allocated_memory()
    synchronize()
    started = time.perf_counter()
    loader_iterator = iter(train_loader)
    startup_seconds = time.perf_counter() - started
    for step in range(1, TRAIN_COUNT // BATCH_SIZE + 1):
        fetched_at = time.perf_counter()
        batch = next(loader_iterator)
        batch_fetch_seconds += time.perf_counter() - fetched_at
        inputs = batch["inputs"]
        mask = inputs["attention_mask"].bool()
        real_tokens += int(mask.sum().item())
        padded_tokens += int((~mask).sum().item())
        for index, pair_id in enumerate(batch["pair_ids"]):
            used = mask[index]
            content_hash.update(json.dumps([
                pair_id, inputs["input_ids"][index, used].tolist(),
                inputs["token_type_ids"][index, used].tolist(),
                float(batch["labels"][index]),
            ], separators=(",", ":")).encode())
        seen_ids.update(batch["pair_ids"])
        computed_at = time.perf_counter()
        model_inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**model_inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training value at step {step}")
        loss.backward()
        optimizer.step()
        synchronize()
        compute_seconds += time.perf_counter() - computed_at
        mps_tensor_max = max(mps_tensor_max, torch.mps.current_allocated_memory())
        mps_driver_max = max(mps_driver_max, torch.mps.driver_allocated_memory())
        seen += len(labels)
        loss_sum += float(loss.detach().item()) * len(labels)
        if step % 16 == 0:
            print(f"{args.padding}: step={step} seen={seen} "
                  f"mean_train_loss={loss_sum / seen:.4f}", flush=True)
    total_seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT:
        raise RuntimeError("Padding run did not use each pair exactly once")
    try:
        next(loader_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Padding loader yielded extra pairs")
    del loader_iterator
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    atomic_torch_save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "update_count": step,
        "padding": args.padding,
        "seed": SEED,
    }, checkpoint)
    report = {
        "experiment": "fixed versus dynamic padding, stored in E4",
        "padding": args.padding,
        "initial_checkpoint_sha256": sha256(initial_path),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "training_pair_ids_sha256": ids_hash(train_rows),
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "tokenized_unpadded_content_sha256": content_hash.hexdigest(),
        "seed": SEED,
        "train_pairs": TRAIN_COUNT,
        "batch_size": BATCH_SIZE,
        "max_length": config.max_length,
        "num_workers": 0,
        "optimizer_updates": step,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "real_tokens": real_tokens,
        "padding_tokens": padded_tokens,
        "total_tokens_processed": real_tokens + padded_tokens,
        "padding_fraction": padded_tokens / (real_tokens + padded_tokens),
        "timing": {
            "scope": "64 training batches: tokenization, content verification, device transfer, forward, backward, optimizer; excludes loading, validation, save",
            "loader_startup_seconds": startup_seconds,
            "batch_fetch_seconds": batch_fetch_seconds,
            "model_compute_seconds": compute_seconds,
            "total_seconds": total_seconds,
            "pairs_per_second": TRAIN_COUNT / total_seconds,
            "mps_synchronized_after_each_update": True,
        },
        "memory": {
            "mps_sampled_max_tensor_bytes": mps_tensor_max,
            "mps_sampled_max_driver_bytes": mps_driver_max,
            "measurement": "sampled current MPS allocation; not true peak",
        },
        "mean_training_bce_loss": loss_sum / seen,
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "backend": "mps",
        "precision": "float32",
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"{args.padding}: {padded_tokens:,} padding tokens; {total_seconds:.2f}s "
          f"({TRAIN_COUNT / total_seconds:.1f} pairs/s); "
          f"validation BCE {before_loss:.4f} -> {after_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
