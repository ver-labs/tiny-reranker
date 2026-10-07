"""Evaluate an E6 matched-training checkpoint on fixed validation candidates."""

import argparse
import hashlib
import json
import os
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
CONFIG_PATH = ROOT / "experiments" / "E1" / "four_layer" / "config.json"
PRECISIONS = ("fp32", "fp16", "bf16")


class MatchedReranker:
    def __init__(self, report: dict) -> None:
        checkpoint = HERE / report["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise RuntimeError("E6 matched checkpoint differs from its report")
        if hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() != report["config_sha256"]:
            raise RuntimeError("E6 model config differs from its report")
        if not torch.backends.mps.is_available() or \
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("MPS without CPU fallback is required for E6 evaluation")
        self.device = torch.device("mps")
        self.tokenizer = load_scratch_tokenizer(CONFIG_PATH)
        config = ScratchConfig.from_file(CONFIG_PATH)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != vars(config) or \
                saved["update_count"] != report["optimizer_updates"] or \
                saved["precision"] != report["precision"] or \
                saved.get("full_cast", False) != report.get("full_cast", False):
            raise RuntimeError("E6 checkpoint metadata differs from its report")
        self.model = ScratchReranker(config)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        dtype = torch.bfloat16 if report.get("full_cast", False) else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--full-cast", action="store_true",
                        help="Evaluate the separate BF16 full-cast control in BF16")
    args = parser.parse_args()
    if args.full_cast and args.precision != "bf16":
        parser.error("Only the BF16 full-cast control passed its one-step probe")
    run_name = f"full_cast_{args.precision}" if args.full_cast else f"matched_{args.precision}"
    report_path = HERE / f"{run_name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["precision"] != args.precision or report.get("full_cast", False) != args.full_cast:
        raise RuntimeError("Precision report differs from requested run")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidates differ from their manifest")
    reranker = MatchedReranker(report)
    with CANDIDATES.open(encoding="utf-8") as source:
        first = json.loads(next(source))
    for _ in range(10):
        score_pairs(reranker, [first["query"]], [first["candidates"][0]["text"]])
    torch.mps.synchronize()
    started = time.perf_counter()
    name = run_name
    training_history = (
        "E6 separate BF16 full-model-cast control, no autocast: 64 updates "
        "on 1,024 pairs; BF16 model weights and BF16 inference."
        if args.full_cast else
        f"E6 matched 4-layer scratch model trained for 64 updates on 1,024 "
        f"pairs with {args.precision} training compute; all three checkpoints "
        "evaluated with FP32 model parameters and inference compute."
    )
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        training_history,
        extra_metadata={
            "model_id": f"scratch_encoder_e6_{run_name}",
            "checkpoint": str((HERE / report["checkpoint"]).relative_to(ROOT)),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_pair_ids_sha256": report["training_pair_ids_sha256"],
            "training_precision": args.precision,
            "evaluation_precision": "bf16" if args.full_cast else "fp32",
            "autocast_used_during_training": not args.full_cast and args.precision != "fp32",
            "full_cast": args.full_cast,
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
        "not_used_for_training_precision_speed_comparison": True,
    }
    summary_path = HERE / f"{name}_summary.json"
    atomic_write_text(json.dumps(summary, indent=2) + "\n", summary_path,
                      replace_existing=True)
    report["full_ranking_evaluation_completed"] = True
    report["full_ranking_summary"] = str(summary_path.relative_to(ROOT))
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path,
                      replace_existing=True)
    print(f"{args.precision}: MRR@10={summary['mrr_at_10']:.4f}; "
          f"nDCG@10={summary['ndcg_at_10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
