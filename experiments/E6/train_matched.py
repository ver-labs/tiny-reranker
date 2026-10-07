"""Train one matched E6 epoch in FP32, FP16 autocast, or BF16 autocast."""

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
DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_sample() -> dict[str, int]:
    return {"tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=tuple(DTYPES), required=True)
    parser.add_argument("--full-cast", action="store_true",
                        help="BF16 control: cast all model weights and do not use autocast")
    args = parser.parse_args()
    if args.full_cast and args.precision != "bf16":
        parser.error("Full-cast training is enabled only for BF16 after its support probe")
    run_name = f"full_cast_{args.precision}" if args.full_cast else f"matched_{args.precision}"
    checkpoint = HERE / f"{run_name}.pt"
    report_path = HERE / f"{run_name}.json"
    if checkpoint.exists() or report_path.exists():
        parser.error(f"Matched {args.precision} output already exists")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU substitution")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS CPU fallback is enabled; refusing an MPS comparison")
    device = torch.device("mps")
    support = json.loads((HERE / "support.json").read_text(encoding="utf-8"))
    check_name = "fp32" if args.precision == "fp32" else f"{args.precision}_autocast"
    mode_support = next(mode for mode in support["modes"] if mode["name"] == check_name)
    if not mode_support["supported_for_this_smoke_test"]:
        raise RuntimeError(f"Local {args.precision} model-operation check did not pass")
    if args.full_cast:
        probe = json.loads((HERE / "full_cast_probe.json").read_text(encoding="utf-8"))
        full_cast_support = next(mode for mode in probe["modes"]
                                 if mode["precision"] == args.precision)
        if not full_cast_support["one_update_completed"]:
            raise RuntimeError("BF16 full-cast model-operation check did not pass")
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"] or \
            sha256(initial_path) != manifest["initial_checkpoint_sha256"] or \
            sha256(initial_path) != support["initial_checkpoint_sha256"]:
        raise RuntimeError("Initial model differs from the E1/support reference")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config) or initial["seed"] != manifest["seed"]:
        raise RuntimeError("Saved initialization does not match config or seed")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.bfloat16 if args.full_cast else torch.float32).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(SEED)
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(train_rows) != TRAIN_COUNT or ids_hash(train_rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("Training pairs differ from the matched reference")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("Validation pairs differ from the matched reference")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=config.max_length, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("mps") if args.precision == "fp16" else None
    if scaler is not None and not scaler.is_enabled():
        raise RuntimeError("Installed MPS GradScaler is disabled")
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    microsteps = updates = skipped = seen = 0
    seen_ids = set()
    loss_sum = 0.0
    scales = [scaler.get_scale()] if scaler is not None else []
    memory_samples = [memory_sample()]
    dtype = DTYPES[args.precision]
    torch.mps.synchronize()
    started = time.perf_counter()
    for batch in train_loader:
        microsteps += 1
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        if labels.device.type != "mps" or \
                any(tensor.device.type != "mps" for tensor in inputs.values()):
            raise RuntimeError("A training batch left MPS")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="mps", dtype=dtype,
                            enabled=dtype is not None and not args.full_cast):
            logits = one_logit_per_label(model(**inputs), labels)
            loss = loss_fn(logits, labels)
        expected_logit_dtype = dtype if dtype is not None else torch.float32
        if logits.dtype != expected_logit_dtype or loss.dtype != torch.float32 or \
                logits.device.type != "mps" or loss.device.type != "mps" or \
                not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Invalid {args.precision} forward/loss at batch {microsteps}")
        memory_samples.append(memory_sample())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        memory_samples.append(memory_sample())
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            scales.append(scale_after)
            if scale_after < scale_before:
                if gradients_finite:
                    raise RuntimeError("GradScaler skipped despite finite gradients")
                skipped += 1
            else:
                if not gradients_finite:
                    raise RuntimeError("GradScaler updated despite nonfinite gradients")
                updates += 1
        else:
            if not gradients_finite:
                raise RuntimeError(f"Nonfinite {args.precision} gradient")
            optimizer.step()
            updates += 1
        torch.mps.synchronize()
        memory_samples.append(memory_sample())
        seen += len(labels)
        loss_sum += float(loss.detach().item()) * len(labels)
        if microsteps % 16 == 0:
            print(f"{args.precision}: batches={microsteps} seen={seen} "
                  f"updates={updates} skipped={skipped} "
                  f"mean_loss={loss_sum / seen:.4f}", flush=True)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or \
            microsteps != TRAIN_COUNT // BATCH_SIZE or updates + skipped != microsteps:
        raise RuntimeError("Matched precision run has wrong sample or update counts")
    expected_parameter_dtype = torch.bfloat16 if args.full_cast else torch.float32
    if any(parameter.dtype != expected_parameter_dtype or parameter.device.type != "mps"
           for parameter in model.parameters()):
        raise RuntimeError("Model parameter dtype or device differs from requested mode")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    atomic_torch_save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "gradient_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "scheduler_state_dict": None,
        "config": asdict(config),
        "completed_epochs": 1,
        "update_count": updates,
        "microstep_count": microsteps,
        "skipped_update_count": skipped,
        "precision": args.precision,
        "full_cast": args.full_cast,
        "training_seed": SEED,
        "rng_state": capture_rng_state(device),
        "data_split_manifest": split_manifest_record(),
    }, checkpoint)
    report = {
        "experiment": "E6 BF16 full-model cast without autocast" if args.full_cast else
                      "E6 matched precision training",
        "precision": args.precision,
        "autocast_used": dtype is not None and not args.full_cast,
        "full_cast": args.full_cast,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "initial_checkpoint_sha256": sha256(initial_path),
        "config_sha256": sha256(config_path),
        "support_report": "experiments/E6/support.json",
        "training_seed": SEED,
        "training_pairs": TRAIN_COUNT,
        "training_pair_ids_sha256": ids_hash(train_rows),
        "validation_pairs": VALIDATION_COUNT,
        "validation_pair_ids_sha256": ids_hash(validation_rows),
        "batch_size": BATCH_SIZE,
        "max_length": config.max_length,
        "padding": "dynamic per batch",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "microsteps": microsteps,
        "optimizer_updates": updates,
        "skipped_optimizer_updates": skipped,
        "mean_training_bce_loss": loss_sum / seen,
        "validation_inference_precision": "bf16 full-model cast" if args.full_cast else "fp32",
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "backend": "mps",
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "parameter_dtype": "bfloat16" if args.full_cast else "float32",
        "autocast_dtype": None if dtype is None or args.full_cast else str(dtype),
        "loss_dtype": "float32",
        "gradient_scaler": None if scaler is None else {
            "implementation": "torch.amp.GradScaler('mps')",
            "enabled": scaler.is_enabled(),
            "initial_scale": scales[0],
            "final_scale": scales[-1],
            "minimum_scale": min(scales),
            "maximum_scale": max(scales),
            "state_saved_in_checkpoint": True,
        },
        "training_time": {
            "seconds": seconds,
            "scope": "64 training batches: tokenization, device transfer, forward, backward, gradient finite check, optimizer/scaler update; excludes loading, validation, save",
            "mps_synchronized_after_each_batch": True,
        },
        "memory": {
            "sampled_max_current_tensor_bytes": max(row["tensor_bytes"] for row in memory_samples),
            "sampled_max_current_driver_bytes": max(row["driver_bytes"] for row in memory_samples),
            "measurement": "maximum sampled MPS current allocation with activations live, after backward, after update; not true peak",
        },
        "full_ranking_evaluation_completed": False,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path)
    print(f"{args.precision}: {microsteps} batches, {updates} updates, "
          f"{skipped} skips, {seconds:.2f}s, validation BCE "
          f"{before_loss:.4f} -> {after_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
