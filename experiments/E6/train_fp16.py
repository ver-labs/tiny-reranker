"""Train one bounded FP16-autocast epoch using PyTorch's supported MPS GradScaler."""

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
CHECKPOINT = HERE / "fp16_epoch1.pt"
REPORT = HERE / "fp16_epoch1.json"
SEED = 123


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_sample() -> dict[str, int]:
    return {"tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory()}


def main() -> None:
    if CHECKPOINT.exists() or REPORT.exists():
        raise FileExistsError("E6 FP16 epoch output already exists")
    support = json.loads((HERE / "support.json").read_text(encoding="utf-8"))
    fp16_support = next(mode for mode in support["modes"]
                        if mode["name"] == "fp16_autocast")
    if not fp16_support["supported_for_this_smoke_test"] or \
            not fp16_support["gradient_scaler"]["enabled"]:
        raise RuntimeError("Local MPS FP16 autocast/scaler smoke check did not pass")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; no silent CPU substitution")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS CPU fallback is enabled; refusing to label this as an MPS run")
    device = torch.device("mps")
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    initial_manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != initial_manifest["config_sha256"] or \
            sha256(initial_path) != initial_manifest["initial_checkpoint_sha256"] or \
            sha256(initial_path) != support["initial_checkpoint_sha256"]:
        raise RuntimeError("FP16 run differs from the checked E1 initialization")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config) or initial["seed"] != initial_manifest["seed"]:
        raise RuntimeError("E1 checkpoint config or seed mismatch")
    model = ScratchReranker(config)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model = model.to(device).train()
    tokenizer = load_scratch_tokenizer(config_path)
    torch.manual_seed(SEED)
    train_rows = list(islice(iter_clean_pairs("train"), TRAIN_COUNT))
    validation_rows = list(islice(iter_clean_pairs("validation"), VALIDATION_COUNT))
    reference = json.loads((ROOT / "results" / "short_epoch_1024.json").read_text())
    if len(train_rows) != TRAIN_COUNT or ids_hash(train_rows) != reference["train_pair_ids_sha256"]:
        raise RuntimeError("FP16 training pairs differ from E1")
    if len(validation_rows) != VALIDATION_COUNT or \
            ids_hash(validation_rows) != reference["validation_pair_ids_sha256"]:
        raise RuntimeError("FP16 validation pairs differ from E1")
    train_loader = make_pair_loader(train_rows, tokenizer, batch_size=BATCH_SIZE,
                                    max_length=config.max_length, shuffle=False)
    validation_loader = make_pair_loader(validation_rows, tokenizer, batch_size=BATCH_SIZE,
                                         max_length=config.max_length, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("mps")
    if not scaler.is_enabled():
        raise RuntimeError("Installed MPS GradScaler is disabled")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    before_loss, before_accuracy = measure(model, validation_loader, device, loss_fn)
    seen_ids = set()
    seen = microsteps = updates = skipped = 0
    total_loss = 0.0
    scales = [scaler.get_scale()]
    memory_samples = [memory_sample()]
    torch.mps.synchronize()
    started = time.perf_counter()
    for batch in train_loader:
        microsteps += 1
        seen_ids.update(batch["pair_ids"])
        inputs, labels = move_batch_to_device(batch, device)
        if any(tensor.device.type != "mps" for tensor in inputs.values()) or \
                labels.device.type != "mps":
            raise RuntimeError("FP16 batch left MPS")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            logits = one_logit_per_label(model(**inputs), labels)
            loss = loss_fn(logits, labels)
        if logits.dtype != torch.float16 or loss.dtype != torch.float32 or \
                not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Invalid FP16 autocast result at batch {microsteps}")
        memory_samples.append(memory_sample())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        memory_samples.append(memory_sample())
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        torch.mps.synchronize()
        memory_samples.append(memory_sample())
        if scale_after < scale_before:
            if gradients_finite:
                raise RuntimeError("GradScaler skipped an update without nonfinite gradients")
            skipped += 1
        else:
            if not gradients_finite:
                raise RuntimeError("GradScaler updated despite nonfinite gradients")
            updates += 1
        scales.append(scale_after)
        seen += len(labels)
        total_loss += float(loss.detach().item()) * len(labels)
        if microsteps % 16 == 0:
            print(f"fp16: batches={microsteps} seen={seen} updates={updates} "
                  f"skipped={skipped} mean_loss={total_loss / seen:.4f}", flush=True)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    if seen != TRAIN_COUNT or len(seen_ids) != TRAIN_COUNT or \
            microsteps != TRAIN_COUNT // BATCH_SIZE or updates + skipped != microsteps:
        raise RuntimeError("FP16 data exposure or update accounting mismatch")
    if any(parameter.dtype != torch.float32 or parameter.device.type != "mps"
           for parameter in model.parameters()):
        raise RuntimeError("FP16 autocast changed parameter dtype or device")
    after_loss, after_accuracy = measure(model, validation_loader, device, loss_fn)
    atomic_torch_save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "gradient_scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": None,
        "config": asdict(config),
        "completed_epochs": 1,
        "update_count": updates,
        "microstep_count": microsteps,
        "skipped_update_count": skipped,
        "training_seed": SEED,
        "rng_state": capture_rng_state(device),
        "data_split_manifest": split_manifest_record(),
    }, CHECKPOINT)
    report = {
        "experiment": "E6 MPS FP16 autocast with GradScaler",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "source_initial_checkpoint_sha256": sha256(initial_path),
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
        "mean_training_bce_loss": total_loss / seen,
        "validation_inference_precision": "fp32",
        "validation_before": {"bce_loss": before_loss, "binary_accuracy": before_accuracy},
        "validation_after": {"bce_loss": after_loss, "binary_accuracy": after_accuracy},
        "backend": "mps",
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
        "loss_dtype": "float32",
        "gradient_scaler": {
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
            "scope": "64 training batches: tokenization, device transfer, autocast forward, scaled backward, unscale, AdamW/scaler update; excludes loading, validation, save",
            "mps_synchronized_after_each_batch": True,
        },
        "memory": {
            "sampled_max_current_tensor_bytes": max(row["tensor_bytes"] for row in memory_samples),
            "sampled_max_current_driver_bytes": max(row["driver_bytes"] for row in memory_samples),
            "measurement": "sampled current MPS allocation with activations live, after backward, after update; not true peak",
        },
        "full_ranking_evaluation_completed": False,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", REPORT)
    print(f"FP16 epoch: {microsteps} batches, {updates} updates, {skipped} skipped; "
          f"{seconds:.2f}s; validation BCE {before_loss:.4f} -> {after_loss:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
