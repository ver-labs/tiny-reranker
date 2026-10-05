"""Evaluate the saved short-epoch scratch checkpoint on all validation candidates."""

import hashlib
import json

import torch

from .evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, MAX_LENGTH, check_metrics, sha256_file
from .evaluate_validation import evaluate_model
from .model import ScratchConfig, ScratchReranker
from .scratch_tokenizer import load_scratch_tokenizer
from .short_epoch import CHECKPOINT, REPORT


class TrainedReranker:
    def __init__(self, expected_hash: str) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        if digest != report["checkpoint_sha256"] or digest != expected_hash:
            raise RuntimeError("Short-epoch checkpoint differs from the recorded run")
        self.checkpoint_sha256 = digest
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = load_scratch_tokenizer()
        self.model = ScratchReranker(ScratchConfig.from_file())
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if saved["optimizer_steps"] != report["optimizer_steps"]:
            raise RuntimeError("Checkpoint step count differs from the run report")
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()


def main() -> None:
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != expected["sha256"]:
        raise RuntimeError("Validation candidate file differs from its manifest")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    reranker = TrainedReranker(report["checkpoint_sha256"])
    evaluate_model(
        "short_epoch_1024", reranker, expected, candidate_hash,
        "Scratch encoder trained from seed-42 random weights for one epoch "
        "on 1,024 selected training pairs (64 optimizer steps); no test queries used.",
        extra_metadata={
            "model_id": "scratch_encoder_seed42_short_epoch_1024",
            "checkpoint": "checkpoints/" + CHECKPOINT.name,
            "checkpoint_sha256": reranker.checkpoint_sha256,
            "training_run_report": "results/" + REPORT.name,
            "training_pairs": report["train_pairs"],
            "optimizer_steps": report["optimizer_steps"],
            "training_pair_ids_sha256": report["train_pair_ids_sha256"],
            "truncation_max_tokens": MAX_LENGTH,
        },
    )


if __name__ == "__main__":
    main()
