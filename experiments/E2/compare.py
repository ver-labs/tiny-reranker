"""Check E2's matched runs and save the worker-count comparison."""

import hashlib
import json
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent
MATCHED = (
    "model", "initial_checkpoint_sha256", "config_sha256", "training_seed",
    "training_pairs", "training_pair_ids_sha256", "validation_pairs",
    "validation_pair_ids_sha256", "batch_size", "max_length", "optimizer",
    "learning_rate", "optimizer_updates", "backend", "precision",
)


def load(worker_count: int) -> tuple[dict, dict]:
    report = json.loads((HERE / f"workers_{worker_count}.json").read_text(encoding="utf-8"))
    if report["num_workers"] != worker_count:
        raise RuntimeError("E2 worker count differs from its filename")
    checkpoint = HERE / report["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise RuntimeError("E2 checkpoint differs from its manifest")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["num_workers"] != worker_count or saved["update_count"] != report["optimizer_updates"]:
        raise RuntimeError("E2 checkpoint counters differ from its report")
    return report, saved


def main() -> None:
    zero, zero_state = load(0)
    two, two_state = load(2)
    if any(zero[field] != two[field] for field in MATCHED):
        raise RuntimeError("E2 runs differ in a setting other than worker count")
    a = zero_state["model_state_dict"]
    b = two_state["model_state_dict"]
    if a.keys() != b.keys():
        raise RuntimeError("E2 checkpoints have different model tensors")
    maximum_difference = max(float((a[name] - b[name]).abs().max().item()) for name in a)
    # MPS training is close here but does not reproduce every weight bit for bit.
    if maximum_difference > 2e-4:
        raise RuntimeError(f"E2 model weights diverged: {maximum_difference}")
    for field in ("mean_training_bce_loss",):
        if abs(zero[field] - two[field]) > 1e-5:
            raise RuntimeError(f"E2 training outcome differs: {field}")
    for field in ("bce_loss", "binary_accuracy"):
        if abs(zero["validation_after"][field] - two["validation_after"][field]) > 1e-5:
            raise RuntimeError(f"E2 validation outcome differs: {field}")
    result = {
        "experiment": "E2 loader workers",
        "matched_fields": list(MATCHED),
        "max_absolute_model_weight_difference": maximum_difference,
        "model_weight_tolerance": 2e-4,
        "single_run_only": True,
        "zero_workers": zero,
        "two_workers": two,
        "comparison": {
            "total_seconds_extra_with_two_workers": two["timing"]["total_seconds"] - zero["timing"]["total_seconds"],
            "loader_startup_seconds_extra_with_two_workers": two["timing"]["loader_startup_seconds"] - zero["timing"]["loader_startup_seconds"],
            "batch_wait_seconds_extra_with_two_workers": two["timing"]["batch_wait_total_seconds"] - zero["timing"]["batch_wait_total_seconds"],
            "throughput_ratio_two_over_zero": two["timing"]["pairs_per_second"] / zero["timing"]["pairs_per_second"],
            "interpretation": "For this 1,024-pair epoch, worker startup outweighed any benefit from parallel batch preparation.",
            "memory_limit": "Main-process RSS excludes worker processes; MPS snapshots are not true peaks.",
        },
    }
    path = HERE / "comparison.json"
    atomic_write_text(json.dumps(result, indent=2) + "\n", path)
    print(f"E2 matched inputs; max model-weight difference {maximum_difference:.3g}")
    print(f"0 workers: {zero['timing']['total_seconds']:.2f}s, "
          f"{zero['timing']['pairs_per_second']:.1f} pairs/s")
    print(f"2 workers: {two['timing']['total_seconds']:.2f}s, "
          f"{two['timing']['pairs_per_second']:.1f} pairs/s")
    print(f"Two-worker startup: {two['timing']['loader_startup_seconds']:.2f}s")


if __name__ == "__main__":
    main()
