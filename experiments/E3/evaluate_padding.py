"""Evaluate an E3 padding checkpoint on the fixed validation candidate lists."""

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
CONFIG = ROOT / "experiments" / "E1" / "four_layer" / "config.json"


class PaddingReranker:
    def __init__(self, report: dict) -> None:
        checkpoint = HERE / report["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise RuntimeError("Padding checkpoint differs from its report")
        if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != report["config_sha256"]:
            raise RuntimeError("Model config differs from the padding run")
        if not torch.backends.mps.is_available():
            raise RuntimeError("E3 evaluation on this Mac requires MPS")
        self.device = torch.device("mps")
        self.tokenizer = load_scratch_tokenizer(CONFIG)
        config = ScratchConfig.from_file(CONFIG)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != vars(config) or \
                saved["update_count"] != report["optimizer_updates"] or \
                saved["padding"] != report["padding"]:
            raise RuntimeError("Padding checkpoint metadata differs from its report")
        self.model = ScratchReranker(config)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding", choices=("fixed", "dynamic"), required=True)
    args = parser.parse_args()
    report_path = HERE / f"{args.padding}_padding.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["padding"] != args.padding:
        raise RuntimeError("Padding report differs from the requested run")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidates differ from their manifest")
    reranker = PaddingReranker(report)
    with CANDIDATES.open(encoding="utf-8") as source:
        first = json.loads(next(source))
    for _ in range(10):
        score_pairs(reranker, [first["query"]], [first["candidates"][0]["text"]])
    torch.mps.synchronize()
    started = time.perf_counter()
    name = f"{args.padding}_padding"
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        f"4-layer scratch encoder trained for 64 updates on 1,024 pairs "
        f"with {args.padding} training-time padding; evaluation uses the same "
        "dynamic-padding scoring routine as the other scratch checkpoints.",
        extra_metadata={
            "model_id": f"scratch_encoder_e3_{name}",
            "checkpoint": str((HERE / report["checkpoint"]).relative_to(ROOT)),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_pair_ids_sha256": report["training_pair_ids_sha256"],
            "training_padding": args.padding,
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
    print(f"{args.padding}: MRR@10={summary['mrr_at_10']:.4f}; "
          f"nDCG@10={summary['ndcg_at_10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
