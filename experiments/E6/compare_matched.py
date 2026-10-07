"""Verify matched E6 exposure and summarize training precision trade-offs."""

import hashlib
import json
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent
PRECISIONS = ("fp32", "fp16", "bf16")
MATCHED = (
    "initial_checkpoint_sha256", "config_sha256", "training_seed",
    "training_pairs", "training_pair_ids_sha256", "validation_pairs",
    "validation_pair_ids_sha256", "batch_size", "max_length", "padding",
    "optimizer", "learning_rate", "microsteps", "optimizer_updates",
    "backend", "parameter_dtype", "loss_dtype", "validation_inference_precision",
)


def main() -> None:
    reports = {}
    summaries = {}
    for precision in PRECISIONS:
        report = json.loads((HERE / f"matched_{precision}.json").read_text(encoding="utf-8"))
        summary = json.loads((HERE / f"matched_{precision}_summary.json").read_text(encoding="utf-8"))
        checkpoint = HERE / report["checkpoint"]
        if report["precision"] != precision or \
                hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"] or \
                summary["checkpoint_sha256"] != report["checkpoint_sha256"] or \
                not report["full_ranking_evaluation_completed"]:
            raise RuntimeError(f"E6 {precision} checkpoint, report, or summary differs")
        reports[precision] = report
        summaries[precision] = summary
    reference = reports["fp32"]
    candidate_hash = summaries["fp32"]["candidate_sha256"]
    query_count = summaries["fp32"]["queries"]
    for precision in PRECISIONS[1:]:
        report = reports[precision]
        summary = summaries[precision]
        if any(report[field] != reference[field] for field in MATCHED) or \
                summary["candidate_sha256"] != candidate_hash or \
                summary["queries"] != query_count or \
                summary["evaluation_precision"] != "fp32":
            raise RuntimeError(f"E6 {precision} settings or candidate lists differ")
    changes = {precision: 0 for precision in PRECISIONS[1:]}
    max_score_difference = {precision: 0.0 for precision in PRECISIONS[1:]}
    count = 0
    with ExitStack() as stack:
        sources = [stack.enter_context((HERE / f"matched_{precision}_rankings.jsonl").open())
                   for precision in PRECISIONS]
        for lines in zip_longest(*sources):
            if any(line is None for line in lines):
                raise RuntimeError("E6 ranking files have unequal query counts")
            rows = [json.loads(line) for line in lines]
            base = rows[0]
            base_ids = [item["document_id"] for item in base["ranking"]]
            base_scores = {item["document_id"]: item["score"] for item in base["ranking"]}
            for precision, row in zip(PRECISIONS[1:], rows[1:]):
                if row["query_id"] != base["query_id"]:
                    raise RuntimeError("E6 query IDs differ")
                ids = [item["document_id"] for item in row["ranking"]]
                if {(item["document_id"], item["label"]) for item in row["ranking"]} != \
                        {(item["document_id"], item["label"]) for item in base["ranking"]}:
                    raise RuntimeError("E6 candidate documents or labels differ")
                changes[precision] += ids != base_ids
                scores = {item["document_id"]: item["score"] for item in row["ranking"]}
                max_score_difference[precision] = max(max_score_difference[precision], *(
                    abs(base_scores[key] - scores[key]) for key in base_scores
                ))
            count += 1
    if count != query_count:
        raise RuntimeError("E6 ranking file count differs from summary")
    results = {}
    for precision in PRECISIONS:
        report, summary = reports[precision], summaries[precision]
        results[precision] = {
            "optimizer_updates": report["optimizer_updates"],
            "skipped_optimizer_updates": report["skipped_optimizer_updates"],
            "training_seconds": report["training_time"]["seconds"],
            "training_time_scope": report["training_time"]["scope"],
            "sampled_max_current_tensor_bytes": report["memory"]["sampled_max_current_tensor_bytes"],
            "sampled_max_current_driver_bytes": report["memory"]["sampled_max_current_driver_bytes"],
            "memory_measurement": report["memory"]["measurement"],
            "validation_bce_128_pairs": report["validation_after"]["bce_loss"],
            "mrr_at_10": summary["mrr_at_10"],
            "ndcg_at_10": summary["ndcg_at_10"],
            "changed_ranking_orders_vs_fp32": changes.get(precision, 0),
            "max_absolute_score_difference_vs_fp32": max_score_difference.get(precision, 0.0),
        }
    result = {
        "experiment": "E6 matched training precision comparison",
        "matched_fields": list(MATCHED),
        "candidate_sha256": candidate_hash,
        "queries": count,
        "evaluation": summaries["fp32"]["evaluation"],
        "evaluation_precision": "fp32 for every checkpoint; isolates training precision",
        "single_training_run_per_mode": True,
        "timing_caveat": "One short epoch per mode; do not infer a stable speed ranking without repeats.",
        "memory_caveat": "Sampled current MPS allocations, not true peaks.",
        "results": results,
        "sampled_tensor_memory_saved_vs_fp32_bytes": {
            precision: results["fp32"]["sampled_max_current_tensor_bytes"] -
                       results[precision]["sampled_max_current_tensor_bytes"]
            for precision in PRECISIONS[1:]
        },
    }
    atomic_write_text(json.dumps(result, indent=2) + "\n", HERE / "matched_comparison.json")
    print(f"Verified {count:,} identical query/candidate lists")
    for precision in PRECISIONS:
        row = results[precision]
        print(f"{precision}: {row['training_seconds']:.2f}s; "
              f"sampled MPS tensor {row['sampled_max_current_tensor_bytes']:,} bytes; "
              f"MRR@10 {row['mrr_at_10']:.4f}; nDCG@10 {row['ndcg_at_10']:.4f}; "
              f"changed rankings {row['changed_ranking_orders_vs_fp32']}")


if __name__ == "__main__":
    main()
