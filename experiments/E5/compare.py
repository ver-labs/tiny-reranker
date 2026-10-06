"""Verify equal E5 data exposure and compare direct/accumulated training."""

import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent
MATCHED = (
    "initial_checkpoint_sha256", "source_config_sha256", "effective_config",
    "padding", "seed", "training_pair_ids_sha256", "validation_pair_ids_sha256",
    "train_pairs", "effective_batch_size", "optimizer_updates", "optimizer",
    "learning_rate", "backend", "precision",
)


def load(name: str) -> tuple[dict, dict]:
    report = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    checkpoint = HERE / report["checkpoint"]
    if report["mode"] != name or \
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise RuntimeError(f"E5 {name} checkpoint differs from its report")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["mode"] != name or saved["update_count"] != report["optimizer_updates"] or \
            saved["microstep_count"] != report["microsteps"]:
        raise RuntimeError(f"E5 {name} counters differ from its report")
    return report, saved


def main() -> None:
    direct, direct_state = load("direct")
    accumulated, accumulated_state = load("accumulate")
    if any(direct[key] != accumulated[key] for key in MATCHED):
        raise RuntimeError("E5 direct and accumulated runs differ in matched settings")
    if direct["microbatch_size"] != 16 or accumulated["microbatch_size"] != 4 or \
            direct["microsteps"] != 64 or accumulated["microsteps"] != 256:
        raise RuntimeError("E5 direct and accumulated microstep counts are wrong")
    a, b = direct_state["model_state_dict"], accumulated_state["model_state_dict"]
    if a.keys() != b.keys():
        raise RuntimeError("E5 checkpoints have different model tensors")
    max_difference = max(float((a[key] - b[key]).abs().max().item()) for key in a)
    if max_difference > 2e-4:
        raise RuntimeError(f"E5 weights diverged beyond MPS tolerance: {max_difference}")
    loss_difference = abs(direct["validation_after"]["bce_loss"] -
                          accumulated["validation_after"]["bce_loss"])
    if loss_difference > 1e-4:
        raise RuntimeError(f"E5 validation loss diverged: {loss_difference}")
    direct_memory = direct["memory"]["sampled_max_current_tensor_bytes"]
    accumulated_memory = accumulated["memory"]["sampled_max_current_tensor_bytes"]
    comparison = {
        "experiment": "E5 gradient accumulation",
        "matched_fields": list(MATCHED),
        "single_run_per_mode": True,
        "direct": {
            "microbatch_size": 16,
            "microsteps": direct["microsteps"],
            "optimizer_updates": direct["optimizer_updates"],
            "training_seconds": direct["training_time"]["seconds"],
            "seconds_per_update": direct["training_time"]["seconds_per_optimizer_update"],
            "sampled_max_current_mps_tensor_bytes": direct_memory,
            "validation_bce": direct["validation_after"]["bce_loss"],
        },
        "accumulate": {
            "microbatch_size": 4,
            "microsteps": accumulated["microsteps"],
            "optimizer_updates": accumulated["optimizer_updates"],
            "training_seconds": accumulated["training_time"]["seconds"],
            "seconds_per_update": accumulated["training_time"]["seconds_per_optimizer_update"],
            "sampled_max_current_mps_tensor_bytes": accumulated_memory,
            "validation_bce": accumulated["validation_after"]["bce_loss"],
        },
        "max_absolute_model_weight_difference": max_difference,
        "absolute_validation_bce_difference": loss_difference,
        "sampled_tensor_memory_saved_bytes": direct_memory - accumulated_memory,
        "sampled_tensor_memory_saved_fraction": (direct_memory - accumulated_memory) / direct_memory,
        "memory_label": "maximum sampled current MPS tensor allocation with activations live/after backward/after update; not true peak",
        "timing_note": "Single short run per mode; do not infer a general speed advantage from the observed times.",
        "dropout_note": direct["dropout_note"],
    }
    summary_paths = [HERE / f"e5_{name}_summary.json" for name in ("direct", "accumulate")]
    if all(path.exists() for path in summary_paths):
        direct_summary, accumulated_summary = [
            json.loads(path.read_text(encoding="utf-8")) for path in summary_paths
        ]
        if direct_summary["candidate_sha256"] != accumulated_summary["candidate_sha256"] or \
                direct_summary["checkpoint_sha256"] != direct["checkpoint_sha256"] or \
                accumulated_summary["checkpoint_sha256"] != accumulated["checkpoint_sha256"]:
            raise RuntimeError("E5 ranking summaries differ from the compared checkpoints")
        ranking_changes = count = 0
        max_score_difference = 0.0
        with (HERE / "e5_direct_rankings.jsonl").open() as left, \
             (HERE / "e5_accumulate_rankings.jsonl").open() as right:
            for a_line, b_line in zip_longest(left, right):
                if a_line is None or b_line is None:
                    raise RuntimeError("E5 ranking files have different query counts")
                a, b = json.loads(a_line), json.loads(b_line)
                if a["query_id"] != b["query_id"]:
                    raise RuntimeError("E5 ranking query IDs differ")
                a_ids = [item["document_id"] for item in a["ranking"]]
                b_ids = [item["document_id"] for item in b["ranking"]]
                if set(a_ids) != set(b_ids):
                    raise RuntimeError("E5 candidate IDs differ")
                ranking_changes += a_ids != b_ids
                a_scores = {item["document_id"]: item["score"] for item in a["ranking"]}
                b_scores = {item["document_id"]: item["score"] for item in b["ranking"]}
                max_score_difference = max(max_score_difference, *(
                    abs(a_scores[key] - b_scores[key]) for key in a_scores
                ))
                count += 1
        if count != direct_summary["queries"] or count != accumulated_summary["queries"]:
            raise RuntimeError("E5 ranked query count differs from the summaries")
        comparison["ranking_evaluation"] = {
            "candidate_sha256": direct_summary["candidate_sha256"],
            "queries": count,
            "direct_mrr_at_10": direct_summary["mrr_at_10"],
            "accumulate_mrr_at_10": accumulated_summary["mrr_at_10"],
            "direct_ndcg_at_10": direct_summary["ndcg_at_10"],
            "accumulate_ndcg_at_10": accumulated_summary["ndcg_at_10"],
            "changed_document_orders": ranking_changes,
            "max_absolute_raw_score_difference": max_score_difference,
            "evaluation": direct_summary["evaluation"],
        }
    atomic_write_text(json.dumps(comparison, indent=2) + "\n", HERE / "comparison.json",
                      replace_existing=True)
    print(f"Same 1,024 pairs and 64 updates; max weight difference {max_difference:.3g}")
    print(f"MPS sampled tensor allocation: {direct_memory:,} -> {accumulated_memory:,} bytes")
    print(f"Training time: {comparison['direct']['training_seconds']:.2f}s -> "
          f"{comparison['accumulate']['training_seconds']:.2f}s")
    if "ranking_evaluation" in comparison:
        ranked = comparison["ranking_evaluation"]
        print(f"MRR@10 {ranked['direct_mrr_at_10']:.4f} in both; "
              f"nDCG@10 {ranked['direct_ndcg_at_10']:.4f} in both; "
              f"{ranked['changed_document_orders']} changed rankings")


if __name__ == "__main__":
    main()
