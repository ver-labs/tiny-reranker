"""Evaluate an E5 checkpoint on all fixed validation candidate lists."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, check_metrics, sha256_file
from scripts.evaluate_validation import evaluate_model, score_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import load_scratch_tokenizer


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOKENIZER_CONFIG = ROOT / "experiments" / "E1" / "four_layer" / "config.json"


class E5Reranker:
    def __init__(self, report: dict) -> None:
        checkpoint = HERE / report["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise RuntimeError("E5 checkpoint differs from its report")
        if hashlib.sha256(TOKENIZER_CONFIG.read_bytes()).hexdigest() != \
                report["source_config_sha256"]:
            raise RuntimeError("E5 tokenizer config differs from its report")
        if not torch.backends.mps.is_available():
            raise RuntimeError("E5 evaluation on this Mac requires MPS")
        self.device = torch.device("mps")
        self.tokenizer = load_scratch_tokenizer(TOKENIZER_CONFIG)
        config = ScratchConfig(**report["effective_config"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != report["effective_config"] or \
                saved["update_count"] != report["optimizer_updates"] or \
                saved["mode"] != report["mode"]:
            raise RuntimeError("E5 checkpoint metadata differs from its report")
        self.model = ScratchReranker(config)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "accumulate"), required=True)
    args = parser.parse_args()
    report_path = HERE / f"{args.mode}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["mode"] != args.mode:
        raise RuntimeError("E5 report differs from the requested run")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidates differ from their manifest")
    reranker = E5Reranker(report)
    with CANDIDATES.open(encoding="utf-8") as source:
        first = json.loads(next(source))
    for _ in range(10):
        score_pairs(reranker, [first["query"]], [first["candidates"][0]["text"]])
    torch.mps.synchronize()
    started = time.perf_counter()
    name = f"e5_{args.mode}"
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        f"E5 {args.mode}: four-layer scratch model, 1,024 training pairs, "
        "64 effective-batch-16 optimizer updates, dropout disabled for the "
        "accumulation comparison.",
        extra_metadata={
            "model_id": f"scratch_encoder_e5_{args.mode}",
            "checkpoint": str((HERE / report["checkpoint"]).relative_to(ROOT)),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_pair_ids_sha256": report["training_pair_ids_sha256"],
            "microbatch_size_during_training": report["microbatch_size"],
            "effective_batch_size_during_training": report["effective_batch_size"],
            "optimizer_updates": report["optimizer_updates"],
        },
        result_dir=HERE,
    )
    torch.mps.synchronize()
    summary["inference_timing"] = {
        "seconds": time.perf_counter() - started,
        "scope": "full validation: tokenization, scoring, sorting, writing; excludes loading",
        "warmup_calls": 10,
        "mps_synchronized_at_start_and_end": True,
    }
    summary_path = HERE / f"{name}_summary.json"
    atomic_write_text(json.dumps(summary, indent=2) + "\n", summary_path,
                      replace_existing=True)
    report["full_ranking_evaluation_completed"] = True
    report["full_ranking_summary"] = str(summary_path.relative_to(ROOT))
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path,
                      replace_existing=True)
    print(f"E5 {args.mode}: MRR@10={summary['mrr_at_10']:.4f}; "
          f"nDCG@10={summary['ndcg_at_10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
