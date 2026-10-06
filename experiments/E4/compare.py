"""Verify matching content and summarize fixed versus dynamic padding."""

import hashlib
import json
from pathlib import Path

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent
MATCHED = (
    "initial_checkpoint_sha256", "config_sha256", "training_pair_ids_sha256",
    "validation_pair_ids_sha256", "tokenized_unpadded_content_sha256", "seed",
    "train_pairs", "batch_size", "max_length", "num_workers", "optimizer_updates",
    "optimizer", "learning_rate", "real_tokens", "backend", "precision",
)


def load(name: str) -> dict:
    report = json.loads((HERE / f"{name}_padding.json").read_text(encoding="utf-8"))
    checkpoint = HERE / report["checkpoint"]
    if report["padding"] != name or \
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise RuntimeError(f"{name} report and checkpoint differ")
    return report


def main() -> None:
    fixed, dynamic = load("fixed"), load("dynamic")
    if any(fixed[field] != dynamic[field] for field in MATCHED):
        raise RuntimeError("Padding runs have different settings or unpadded token content")
    if fixed["total_tokens_processed"] != fixed["train_pairs"] * fixed["max_length"]:
        raise RuntimeError("Fixed padding did not fill the configured token length")
    if not dynamic["padding_tokens"] < fixed["padding_tokens"]:
        raise RuntimeError("Dynamic padding did not reduce padding tokens")
    result = {
        "experiment": "fixed versus dynamic padding; stored in E4 as requested",
        "matched_fields": list(MATCHED),
        "single_run_only": True,
        "fixed": fixed,
        "dynamic": dynamic,
        "comparison": {
            "padding_tokens_saved_by_dynamic": fixed["padding_tokens"] - dynamic["padding_tokens"],
            "total_token_slots_saved_by_dynamic": fixed["total_tokens_processed"] - dynamic["total_tokens_processed"],
            "fraction_of_fixed_token_slots_saved": (
                fixed["total_tokens_processed"] - dynamic["total_tokens_processed"]
            ) / fixed["total_tokens_processed"],
            "dynamic_minus_fixed_training_seconds": (
                dynamic["timing"]["total_seconds"] - fixed["timing"]["total_seconds"]
            ),
            "interpretation": "Dynamic padding reduced token slots, but this one short run does not establish a speed advantage.",
            "quality_note": "Training uses dropout; changing tensor shapes changes random-mask use, so validation losses are not a controlled quality comparison.",
        },
    }
    atomic_write_text(json.dumps(result, indent=2) + "\n", HERE / "comparison.json")
    print(f"Same unpadded tokens; dynamic saved {result['comparison']['padding_tokens_saved_by_dynamic']:,} token slots")
    print(f"Fixed {fixed['timing']['total_seconds']:.2f}s; "
          f"dynamic {dynamic['timing']['total_seconds']:.2f}s")


if __name__ == "__main__":
    main()
