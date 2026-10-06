"""Evaluate a head-only E1 checkpoint on the fixed validation candidates."""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, check_metrics, sha256_file
from scripts.evaluate_validation import evaluate_model, score_pairs

from .evaluate_variant import EXPERIMENT, TrainedVariant, synchronize


VARIANTS = ("two_layer", "four_layer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    folder = EXPERIMENT / args.variant
    report_path = folder / "head_only.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["variant"] != args.variant or not report["frozen_backbone_unchanged"]:
        raise RuntimeError("Head-only training report is missing or inconsistent")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidate file differs from its manifest")
    reranker = TrainedVariant(folder, report)
    with CANDIDATES.open(encoding="utf-8") as source:
        example = json.loads(next(source))
    query = example["query"]
    document = example["candidates"][0]["text"]
    for _ in range(10):
        score_pairs(reranker, [query], [document])
    synchronize(reranker.device)
    started = time.perf_counter()
    name = f"{args.variant}_head_only"
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        f"E1 {args.variant}: trained backbone from epoch1.pt frozen; "
        "only 129 scoring-head parameters updated on 1,024 new training pairs "
        "with query IDs disjoint from the original E1 training sample.",
        extra_metadata={
            "model_id": f"scratch_encoder_e1_{args.variant}_head_only",
            "checkpoint": str((folder / report["checkpoint"]).relative_to(EXPERIMENT.parents[1])),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
            "head_only_training_pair_ids_sha256": report["new_training_pair_ids_sha256"],
            "trainable_parameter_count_during_head_training": 129,
        },
        result_dir=folder,
    )
    synchronize(reranker.device)
    seconds = time.perf_counter() - started
    summary["inference_timing"] = {
        "seconds": seconds,
        "scope": "full validation: tokenization, device transfer, scoring, sorting, JSONL writing; excludes loading",
        "warmup_calls": 10,
        "mps_synchronized_at_start_and_end": reranker.device.type == "mps",
    }
    if reranker.device.type == "mps":
        summary["inference_memory_snapshot"] = {
            "measurement": "MPS current allocation after full validation evaluation; not peak memory",
            "tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
    baseline = json.loads((folder / f"{args.variant}_epoch1_summary.json").read_text(encoding="utf-8"))
    if baseline["candidate_sha256"] != candidate_hash or baseline["queries"] != summary["queries"]:
        raise RuntimeError("Head-only validation list differs from the parent model's")
    summary["change_from_parent"] = {
        "mrr_at_10": summary["mrr_at_10"] - baseline["mrr_at_10"],
        "ndcg_at_10": summary["ndcg_at_10"] - baseline["ndcg_at_10"],
    }
    summary_path = folder / f"{name}_summary.json"
    atomic_write_text(json.dumps(summary, indent=2) + "\n", summary_path,
                      replace_existing=True)
    report["full_ranking_evaluation_completed"] = True
    report["full_ranking_summary"] = str(summary_path.relative_to(EXPERIMENT.parents[1]))
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path,
                      replace_existing=True)
    print(f"{args.variant}: head-only MRR@10={summary['mrr_at_10']:.4f} "
          f"({summary['change_from_parent']['mrr_at_10']:+.4f}); "
          f"nDCG@10={summary['ndcg_at_10']:.4f} "
          f"({summary['change_from_parent']['ndcg_at_10']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
