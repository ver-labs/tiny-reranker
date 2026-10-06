"""Score an E1 trained model on the fixed full validation candidate lists."""

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


EXPERIMENT = Path(__file__).resolve().parent
VARIANTS = ("two_layer", "four_layer")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


class TrainedVariant:
    def __init__(self, folder: Path, report: dict) -> None:
        checkpoint = folder / report["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise RuntimeError("E1 trained checkpoint differs from its report")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        config_path = folder / "config.json"
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != report["config_sha256"]:
            raise RuntimeError("E1 model config differs from its training report")
        self.tokenizer = load_scratch_tokenizer(config_path)
        self.model = ScratchReranker(ScratchConfig.from_file(config_path))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != vars(self.model.config) or \
                saved["update_count"] != report["optimizer_updates"]:
            raise RuntimeError("E1 checkpoint metadata differs from its training report")
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    folder = EXPERIMENT / args.variant
    training_report_path = folder / "epoch1.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report["variant"] != args.variant or not training_report["trained"]:
        raise RuntimeError("E1 variant has not completed its training run")
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidates differ from their manifest")
    reranker = TrainedVariant(folder, training_report)
    with CANDIDATES.open(encoding="utf-8") as source:
        example = json.loads(next(source))
    query = example["query"]
    document = example["candidates"][0]["text"]
    for _ in range(10):
        score_pairs(reranker, [query], [document])
    synchronize(reranker.device)
    started = time.perf_counter()
    name = f"{args.variant}_epoch1"
    summary = evaluate_model(
        name, reranker, expected, candidate_hash,
        f"E1 scratch {args.variant} model: seed-42 random initialization, "
        "one epoch on the same 1,024 training pairs, 64 AdamW updates.",
        extra_metadata={
            "model_id": f"scratch_encoder_e1_{args.variant}",
            "checkpoint": str((folder / "epoch1.pt").relative_to(EXPERIMENT.parents[1])),
            "checkpoint_sha256": training_report["checkpoint_sha256"],
            "training_pair_ids_sha256": training_report["training_pair_ids_sha256"],
            "parameter_count": training_report["parameter_count"],
        },
        result_dir=folder,
    )
    synchronize(reranker.device)
    seconds = time.perf_counter() - started
    summary["inference_timing"] = {
        "seconds": seconds,
        "scope": "full validation: tokenization, device transfer, scoring, sorting, and JSONL writing; excludes loading",
        "warmup_calls": 10,
        "mps_synchronized_at_start_and_end": reranker.device.type == "mps",
    }
    if reranker.device.type == "mps":
        summary["inference_memory_snapshot"] = {
            "measurement": "MPS current allocation after full validation evaluation; not peak memory",
            "tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
    summary_path = folder / f"{name}_summary.json"
    atomic_write_text(json.dumps(summary, indent=2) + "\n", summary_path,
                      replace_existing=True)
    training_report["full_ranking_evaluation_completed"] = True
    training_report["full_ranking_summary"] = str(summary_path.relative_to(EXPERIMENT.parents[1]))
    atomic_write_text(json.dumps(training_report, indent=2) + "\n",
                      training_report_path, replace_existing=True)
    print(f"{args.variant}: full validation in {seconds:.2f}s; "
          f"MRR@10={summary['mrr_at_10']:.4f}; nDCG@10={summary['ndcg_at_10']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
