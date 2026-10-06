"""Verify E4's fixed workload and summarize batching performance."""

import json
from pathlib import Path

from scripts.checkpoint_io import atomic_write_text


HERE = Path(__file__).resolve().parent
SIZES = (1, 4, 8)
MATCHED = (
    "checkpoint_sha256", "candidate_file_sha256",
    "selected_query_candidate_ids_sha256", "sample_rule", "query_count",
    "pair_count", "candidate_count_distribution", "query_group_size",
    "query_latency_definition", "timing_scope", "warmup_calls",
    "untimed_full_sample_warmup_passes", "repeats", "max_length",
    "precision", "backend", "memory_label",
)


def main() -> None:
    reports = {
        size: json.loads((HERE / f"batch_{size}.json").read_text(encoding="utf-8"))
        for size in SIZES
    }
    baseline = reports[1]
    for size, report in reports.items():
        if report["pair_batch_size"] != size or \
                any(report[key] != baseline[key] for key in MATCHED):
            raise RuntimeError(f"E4 batch size {size} differs in its workload or settings")
    ranking_changes = {}
    maximum_score_difference = {}
    for size in (4, 8):
        changed = 0
        max_difference = 0.0
        for a, b in zip(baseline["rankings"], reports[size]["rankings"], strict=True):
            if a["query_id"] != b["query_id"]:
                raise RuntimeError("E4 query IDs are not aligned")
            a_ids = [item["document_id"] for item in a["ranking"]]
            b_ids = [item["document_id"] for item in b["ranking"]]
            if set(a_ids) != set(b_ids):
                raise RuntimeError("E4 candidate documents are not aligned")
            changed += a_ids != b_ids
            scores_a = {item["document_id"]: item["score"] for item in a["ranking"]}
            scores_b = {item["document_id"]: item["score"] for item in b["ranking"]}
            max_difference = max(max_difference, *(abs(scores_a[key] - scores_b[key])
                                                    for key in scores_a))
        ranking_changes[str(size)] = changed
        maximum_score_difference[str(size)] = max_difference
    if any(ranking_changes.values()) or any(value > 1e-4 for value in maximum_score_difference.values()):
        raise RuntimeError("E4 outputs changed materially with pair batch size")
    comparison = {
        "experiment": "E4 inference pair batching",
        "matched_fields": list(MATCHED),
        "query_count": baseline["query_count"],
        "pair_count": baseline["pair_count"],
        "ranking_changes_against_batch_1": ranking_changes,
        "max_absolute_score_difference_against_batch_1": maximum_score_difference,
        "memory_label": baseline["memory_label"],
        "results": {
            str(size): {
                "repeat_seconds": [run["seconds"] for run in reports[size]["runs"]],
                "median_total_seconds": reports[size]["median_seconds"],
                "median_pairs_per_second": reports[size]["median_pairs_per_second"],
                "median_query_latency_ms": 1000 * reports[size]["median_query_latency_seconds"],
                "median_query_p95_ms": 1000 * reports[size]["median_query_p95_seconds"],
                "sampled_max_current_tensor_bytes": reports[size]["sampled_max_current_tensor_bytes"],
                "sampled_max_current_driver_bytes": reports[size]["sampled_max_current_driver_bytes"],
            }
            for size in SIZES
        },
        "scope_note": "Eight-query groups pool pairs across queries because the fixed validation lists each have two candidates.",
        "single_benchmark_session": True,
    }
    atomic_write_text(json.dumps(comparison, indent=2) + "\n", HERE / "comparison.json")
    print(f"All {baseline['query_count']} rankings agree; max score differences: "
          f"4={maximum_score_difference['4']:.3g}, 8={maximum_score_difference['8']:.3g}")
    for size in SIZES:
        row = comparison["results"][str(size)]
        print(f"{size}: {row['median_pairs_per_second']:.1f} pairs/s; "
              f"median query latency {row['median_query_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
