"""Summarize the separate BF16 full-cast control and FP16 probe failure."""

import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent


def main() -> None:
    probe = json.loads((HERE / "full_cast_probe.json").read_text(encoding="utf-8"))
    fp16_probe = next(mode for mode in probe["modes"] if mode["precision"] == "fp16")
    bf16_probe = next(mode for mode in probe["modes"] if mode["precision"] == "bf16")
    if fp16_probe["one_update_completed"] or not bf16_probe["one_update_completed"]:
        raise RuntimeError("Full-cast probe outcomes changed")
    names = ("matched_fp32", "matched_bf16", "full_cast_bf16")
    reports = {name: json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
               for name in names}
    summaries = {name: json.loads((HERE / f"{name}_summary.json").read_text(encoding="utf-8"))
                 for name in names}
    base = reports["matched_bf16"]
    control = reports["full_cast_bf16"]
    for name, report in reports.items():
        if hashlib.sha256((HERE / report["checkpoint"]).read_bytes()).hexdigest() != \
                report["checkpoint_sha256"] or \
                report["checkpoint_sha256"] != summaries[name]["checkpoint_sha256"]:
            raise RuntimeError(f"{name} checkpoint differs from report or summary")
        if report["initial_checkpoint_sha256"] != base["initial_checkpoint_sha256"] or \
                report["training_pair_ids_sha256"] != base["training_pair_ids_sha256"] or \
                report["optimizer_updates"] != base["optimizer_updates"] or \
                summaries[name]["candidate_sha256"] != summaries["matched_bf16"]["candidate_sha256"]:
            raise RuntimeError(f"{name} source weights, data exposure, or candidates differ")
    if control["autocast_used"] or not control["full_cast"] or \
            control["parameter_dtype"] != "bfloat16" or \
            summaries["full_cast_bf16"]["evaluation_precision"] != "bf16":
        raise RuntimeError("BF16 full-cast control metadata is inconsistent")
    changed = count = 0
    max_score_difference = 0.0
    with (HERE / "matched_bf16_rankings.jsonl").open() as left, \
         (HERE / "full_cast_bf16_rankings.jsonl").open() as right:
        for a_line, b_line in zip_longest(left, right):
            if a_line is None or b_line is None:
                raise RuntimeError("BF16 ranking files have unequal query counts")
            a, b = json.loads(a_line), json.loads(b_line)
            if a["query_id"] != b["query_id"]:
                raise RuntimeError("BF16 query IDs differ")
            a_ids = [item["document_id"] for item in a["ranking"]]
            b_ids = [item["document_id"] for item in b["ranking"]]
            if {(item["document_id"], item["label"]) for item in a["ranking"]} != \
                    {(item["document_id"], item["label"]) for item in b["ranking"]}:
                raise RuntimeError("BF16 candidate IDs or labels differ")
            changed += a_ids != b_ids
            a_scores = {item["document_id"]: item["score"] for item in a["ranking"]}
            b_scores = {item["document_id"]: item["score"] for item in b["ranking"]}
            max_score_difference = max(max_score_difference, *(
                abs(a_scores[key] - b_scores[key]) for key in a_scores
            ))
            count += 1
    if count != summaries["matched_bf16"]["queries"] or \
            count != summaries["full_cast_bf16"]["queries"]:
        raise RuntimeError("BF16 query count differs from summaries")
    result = {
        "experiment": "E6 full-model cast without autocast, separate control",
        "fp16_full_cast": {
            "one_update_completed": False,
            "failure": fp16_probe["failure"],
            "longer_run_started": False,
        },
        "bf16_full_cast": {
            "one_update_completed": True,
            "training_pairs": control["training_pairs"],
            "optimizer_updates": control["optimizer_updates"],
            "training_seconds": control["training_time"]["seconds"],
            "sampled_max_current_tensor_bytes": control["memory"]["sampled_max_current_tensor_bytes"],
            "validation_bce_128_pairs": control["validation_after"]["bce_loss"],
            "mrr_at_10": summaries["full_cast_bf16"]["mrr_at_10"],
            "ndcg_at_10": summaries["full_cast_bf16"]["ndcg_at_10"],
            "training_precision": "bf16 model parameters and compute; no autocast",
            "evaluation_precision": "bf16 model parameters and compute",
        },
        "bf16_autocast_reference": {
            "training_seconds": base["training_time"]["seconds"],
            "sampled_max_current_tensor_bytes": base["memory"]["sampled_max_current_tensor_bytes"],
            "validation_bce_128_pairs": base["validation_after"]["bce_loss"],
            "mrr_at_10": summaries["matched_bf16"]["mrr_at_10"],
            "ndcg_at_10": summaries["matched_bf16"]["ndcg_at_10"],
            "training_precision": "FP32 parameters, BF16 autocast compute",
            "evaluation_precision": "fp32",
        },
        "same_candidate_queries": count,
        "changed_document_orders_vs_bf16_autocast": changed,
        "max_absolute_raw_score_difference_vs_bf16_autocast": max_score_difference,
        "mrr_at_10_change_full_cast_minus_autocast": (
            summaries["full_cast_bf16"]["mrr_at_10"] - summaries["matched_bf16"]["mrr_at_10"]
        ),
        "ndcg_at_10_change_full_cast_minus_autocast": (
            summaries["full_cast_bf16"]["ndcg_at_10"] - summaries["matched_bf16"]["ndcg_at_10"]
        ),
        "interpretation_limit": "Full cast changes both training parameter precision and evaluation precision; one short run per mode cannot isolate their separate effects or establish a stable speed result.",
        "evaluation": summaries["full_cast_bf16"]["evaluation"],
    }
    atomic_write_text(json.dumps(result, indent=2) + "\n", HERE / "full_cast_comparison.json")
    print(f"FP16 full cast: failed one-step MPS GradScaler unscale ({fp16_probe['failure']['message']})")
    print(f"BF16 full cast: {count:,} matched query lists; {changed} changed rankings; "
          f"MRR@10={result['bf16_full_cast']['mrr_at_10']:.4f}; "
          f"nDCG@10={result['bf16_full_cast']['ndcg_at_10']:.4f}")


if __name__ == "__main__":
    main()
