"""Evaluate an E8 model on the fixed full validation candidate lists."""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, check_metrics, sha256_file
from scripts.evaluate_validation import evaluate_model, score_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import load_scratch_tokenizer

from .train_variant import E1, HERE, sha256


class TrainedVariant:
    def __init__(self, report: dict) -> None:
        checkpoint = HERE / report["checkpoint"]
        if sha256(checkpoint) != report["checkpoint_sha256"]:
            raise RuntimeError("E8 checkpoint differs from training report")
        config_path = E1 / "config.json"
        if sha256(config_path) != report["config_sha256"]:
            raise RuntimeError("E8 config differs from training report")
        if not torch.backends.mps.is_available():
            raise RuntimeError("E8 validation requires MPS")
        self.device = torch.device("mps")
        self.tokenizer = load_scratch_tokenizer(config_path)
        self.model = ScratchReranker(ScratchConfig.from_file(config_path))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != vars(self.model.config) or \
                saved["update_count"] != report["optimizer_updates"] or \
                saved["activation_checkpointing"] != (report["activation_checkpointing"] == "on"):
            raise RuntimeError("E8 checkpoint metadata differs from training report")
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpointing", choices=("off", "on"), required=True)
    args = parser.parse_args()
    report_path = HERE / f"{args.checkpointing}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["activation_checkpointing"] != args.checkpointing:
        raise RuntimeError("Training report has the wrong E8 variant")
    name = f"four_layer_checkpointing_{args.checkpointing}"
    if (HERE / f"{name}_summary.json").exists():
        parser.error(f"E8 {args.checkpointing} evaluation already exists")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidates differ from their manifest")
    reranker = TrainedVariant(report)
    with CANDIDATES.open(encoding="utf-8") as source:
        example = json.loads(next(source))
    query = example["query"]
    document = example["candidates"][0]["text"]
    for _ in range(10):
        score_pairs(reranker, [query], [document])
    torch.mps.synchronize()
    started = time.perf_counter()
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        "E8 scratch four-layer model: seed-42 initialization, one epoch on the "
        "same 1,024 training pairs and 64 AdamW updates; activation checkpointing "
        f"{args.checkpointing} during training.",
        extra_metadata={
            "model_id": f"scratch_encoder_e8_{args.checkpointing}",
            "checkpoint": str(Path("experiments/E8") / report["checkpoint"]),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_pair_ids_sha256": report["training_pair_ids_sha256"],
            "activation_checkpointing_during_training": args.checkpointing,
        },
        result_dir=HERE,
    )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    summary["inference_timing"] = {
        "seconds": elapsed,
        "scope": "full validation: tokenization, device transfer, scoring, sorting, JSONL writing; excludes loading",
        "warmup_calls": 10,
        "mps_synchronized_at_start_and_end": True,
    }
    atomic_write_text(json.dumps(summary, indent=2) + "\n",
                      HERE / f"{name}_summary.json", replace_existing=True)
    report["full_ranking_evaluation_completed"] = True
    report["full_ranking_summary"] = str(Path("experiments/E8") / f"{name}_summary.json")
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path,
                      replace_existing=True)
    print(f"checkpointing={args.checkpointing}: full validation in {elapsed:.2f}s; "
          f"MRR@10={summary['mrr_at_10']:.4f}; nDCG@10={summary['ndcg_at_10']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
