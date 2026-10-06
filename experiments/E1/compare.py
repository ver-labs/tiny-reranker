"""Verify matched E1 settings and summarize quality and compute costs."""

import json
from pathlib import Path

from scripts.checkpoint_io import atomic_write_text


EXPERIMENT = Path(__file__).resolve().parent
VARIANTS = ("two_layer", "four_layer")


def main() -> None:
    training = {
        name: json.loads((EXPERIMENT / name / "epoch1.json").read_text(encoding="utf-8"))
        for name in VARIANTS
    }
    quality = {
        name: json.loads((EXPERIMENT / name / f"{name}_epoch1_summary.json").read_text(encoding="utf-8"))
        for name in VARIANTS
    }
    left, right = VARIANTS
    for key in ("training_seed", "training_pairs", "training_pair_ids_sha256",
                "validation_pairs", "validation_pair_ids_sha256", "split_manifest",
                "batch_size", "optimizer", "learning_rate", "optimizer_updates",
                "backend", "precision"):
        if training[left][key] != training[right][key]:
            raise RuntimeError(f"E1 training settings differ: {key}")
    for key in ("candidate_sha256", "queries", "metric_denominator", "backend",
                "precision", "truncation_max_tokens", "pair_batch_size", "tie_break"):
        if quality[left][key] != quality[right][key]:
            raise RuntimeError(f"E1 validation settings differ: {key}")
    report = {
        "experiment": "E1: encoder depth, 2 versus 4 layers",
        "status": "both variants trained for one short epoch and evaluated on fixed validation lists",
        "training_controls": {
            key: training[left][key] for key in
            ("training_seed", "training_pairs", "training_pair_ids_sha256",
             "batch_size", "optimizer", "learning_rate", "optimizer_updates", "backend")
        },
        "validation_controls": {
            key: quality[left][key] for key in
            ("candidate_sha256", "queries", "truncation_max_tokens", "pair_batch_size")
        },
        "models": {
            name: {
                "layers": 2 if name == "two_layer" else 4,
                "trainable_parameters": training[name]["parameter_count"],
                "training_seconds": training[name]["training_time"]["seconds"],
                "training_sampled_max_mps_tensor_bytes": training[name]["memory"]["sampled_max_tensor_bytes"],
                "training_sampled_max_mps_driver_bytes": training[name]["memory"]["sampled_max_driver_bytes"],
                "validation_mrr_at_10": quality[name]["mrr_at_10"],
                "validation_ndcg_at_10": quality[name]["ndcg_at_10"],
                "full_validation_seconds": quality[name]["inference_timing"]["seconds"],
                "training_report": f"{name}/epoch1.json",
                "validation_summary": f"{name}/{name}_epoch1_summary.json",
            }
            for name in VARIANTS
        },
        "four_minus_two": {
            "trainable_parameters": training[right]["parameter_count"] - training[left]["parameter_count"],
            "validation_mrr_at_10": quality[right]["mrr_at_10"] - quality[left]["mrr_at_10"],
            "validation_ndcg_at_10": quality[right]["ndcg_at_10"] - quality[left]["ndcg_at_10"],
            "training_seconds": training[right]["training_time"]["seconds"] - training[left]["training_time"]["seconds"],
            "full_validation_seconds": quality[right]["inference_timing"]["seconds"] - quality[left]["inference_timing"]["seconds"],
        },
        "limits": [
            "Single seed and one 1,024-pair epoch; quality difference may be noisy.",
            "Validation candidates are constructed from MS MARCO mined triplets, not a leaderboard workload.",
            "Training memory values are sampled after updates, not true peaks.",
            "Timings are single sequential runs and include runtime variation.",
        ],
    }
    output = EXPERIMENT / "comparison.json"
    atomic_write_text(json.dumps(report, indent=2) + "\n", output, replace_existing=True)
    print(f"2 layers: MRR@10={quality[left]['mrr_at_10']:.4f}, "
          f"nDCG@10={quality[left]['ndcg_at_10']:.4f}, "
          f"train={training[left]['training_time']['seconds']:.2f}s, "
          f"full validation={quality[left]['inference_timing']['seconds']:.2f}s")
    print(f"4 layers: MRR@10={quality[right]['mrr_at_10']:.4f}, "
          f"nDCG@10={quality[right]['ndcg_at_10']:.4f}, "
          f"train={training[right]['training_time']['seconds']:.2f}s, "
          f"full validation={quality[right]['inference_timing']['seconds']:.2f}s")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
