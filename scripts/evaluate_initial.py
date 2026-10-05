"""Compare MiniLM with the saved random model on fixed validation candidates."""

import hashlib
import json
import math
import resource
import statistics
import time
from pathlib import Path

import torch

from .baselines import BaselineReranker, Document, MODEL_ID, MODEL_REVISION
from .model import ScratchConfig, ScratchReranker
from .scratch_tokenizer import encode_pairs, load_scratch_tokenizer


ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "data" / "processed" / "eval_validation_candidates.jsonl"
CANDIDATE_MANIFEST = ROOT / "results" / "evaluation_candidates.json"
SAMPLE = ROOT / "results" / "initial_validation_sample.json"
RESULT_DIR = ROOT / "results" / "initial_validation"
INITIAL_CHECKPOINT = ROOT / "checkpoints" / "initial_seed42.pt"
INITIAL_MANIFEST = INITIAL_CHECKPOINT.with_suffix(".json")
QUALITY_QUERIES = 128
TIMING_QUERIES = 16
PAIR_BATCH_SIZE = 2
MAX_LENGTH = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(labels: list[int]) -> tuple[float, float]:
    """MRR@10 and binary nDCG@10; zero for no relevant candidate."""
    top = labels[:10]
    first = next((rank for rank, label in enumerate(top, 1) if label == 1), None)
    mrr = 1.0 / first if first is not None else 0.0
    dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(top, 1))
    ideal = sum(1 / math.log2(rank + 1)
                for rank in range(1, min(sum(labels), 10) + 1))
    return mrr, dcg / ideal if ideal else 0.0


def check_metrics() -> None:
    for labels, expected in (([1, 0], (1.0, 1.0)),
                             ([0, 1], (0.5, 1 / math.log2(3))),
                             ([0, 0], (0.0, 0.0))):
        actual = ranking_metrics(labels)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(actual, expected))


def load_sample() -> tuple[list[dict], str]:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected_hash = manifest["splits"]["validation"]["sha256"]
    if sha256_file(CANDIDATES) != expected_hash:
        raise RuntimeError("Validation candidate file differs from its manifest")
    if SAMPLE.exists():
        saved = json.loads(SAMPLE.read_text(encoding="utf-8"))
        if saved["candidate_sha256"] != expected_hash:
            raise RuntimeError("Saved sample points to different candidate data")
        selected_ids = saved["query_ids"]
    else:
        with CANDIDATES.open(encoding="utf-8") as source:
            selected_ids = [json.loads(next(source))["query_id"]
                            for _ in range(QUALITY_QUERIES)]
        SAMPLE.write_text(json.dumps({
            "split": "validation",
            "selection": "first 128 query IDs in the saved, sorted candidate file",
            "candidate_sha256": expected_hash,
            "query_ids": selected_ids,
        }, indent=2) + "\n", encoding="utf-8")
    selected = set(selected_ids)
    rows = []
    with CANDIDATES.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["query_id"] in selected:
                rows.append(row)
    if len(rows) != len(selected_ids) or [row["query_id"] for row in rows] != selected_ids:
        raise RuntimeError("Saved validation query IDs are missing or out of order")
    return rows, expected_hash


class RandomReranker:
    def __init__(self) -> None:
        manifest = json.loads(INITIAL_MANIFEST.read_text(encoding="utf-8"))
        if sha256_file(INITIAL_CHECKPOINT) != manifest["sha256"]:
            raise RuntimeError("Initial random checkpoint differs from its manifest")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = load_scratch_tokenizer()
        self.model = ScratchReranker(ScratchConfig.from_file())
        saved = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def rank(self, query: str, documents: list[Document]) -> list[tuple[str, float]]:
        scored = []
        for start in range(0, len(documents), PAIR_BATCH_SIZE):
            batch = documents[start:start + PAIR_BATCH_SIZE]
            inputs = encode_pairs(
                self.tokenizer, [query] * len(batch), [item.text for item in batch],
                max_length=MAX_LENGTH,
            ).to(self.device)
            values = self.model(**inputs).detach().cpu().tolist()
            scored.extend((item.id, float(value)) for item, value in zip(batch, values))
        return sorted(scored, key=lambda item: (-item[1], item[0]))


def rank_query(reranker, row: dict) -> list[dict]:
    documents = [Document(item["document_id"], item["text"])
                 for item in row["candidates"]]
    raw = reranker.rank(row["query"], documents)
    ranked = sorted(((item.id, item.score) if hasattr(item, "id") else item
                     for item in raw), key=lambda item: (-item[1], item[0]))
    labels = {item["document_id"]: item["label"] for item in row["candidates"]}
    if len(ranked) != len(labels) or {doc_id for doc_id, _ in ranked} != set(labels):
        raise RuntimeError("Reranker lost or duplicated a candidate")
    return [{"document_id": doc_id, "label": labels[doc_id], "score": score}
            for doc_id, score in ranked]


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict:
    if device.type == "mps":
        return {
            "measurement": "MPS current allocation snapshot after timed call, bytes",
            "tensor_allocated": torch.mps.current_allocated_memory(),
            "driver_allocated": torch.mps.driver_allocated_memory(),
        }
    return {
        "measurement": "process peak RSS as reported by macOS, bytes",
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def evaluate(name: str, reranker, rows: list[dict], candidate_hash: str,
             training_history: str, model_metadata: dict) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = RESULT_DIR / f"{name}_rankings.jsonl"
    mrr_values, ndcg_values = [], []
    no_relevant = 0
    with ranking_path.open("w", encoding="utf-8") as output:
        for row in rows:
            ranking = rank_query(reranker, row)
            labels = [item["label"] for item in ranking]
            no_relevant += not any(labels)
            mrr, ndcg = ranking_metrics(labels)
            mrr_values.append(mrr)
            ndcg_values.append(ndcg)
            output.write(json.dumps({
                "query_id": row["query_id"], "mrr_at_10": mrr,
                "ndcg_at_10": ndcg, "ranking": ranking,
            }) + "\n")

    timing_rows = rows[:TIMING_QUERIES]
    for _ in range(10):
        rank_query(reranker, timing_rows[0])
    seconds = []
    memory = []
    for _ in range(3):
        synchronize(reranker.device)
        start = time.perf_counter()
        for row in timing_rows:
            rank_query(reranker, row)
        synchronize(reranker.device)
        seconds.append(time.perf_counter() - start)
        memory.append(memory_snapshot(reranker.device))

    summary = {
        "model": name,
        **model_metadata,
        "training_history": training_history,
        "evaluation": "constructed validation candidate lists; sampled, not leaderboard comparable",
        "candidate_sha256": candidate_hash,
        "sample_manifest": str(SAMPLE.relative_to(ROOT)),
        "rankings_file": str(ranking_path.relative_to(ROOT)),
        "queries": len(rows),
        "queries_without_relevant_candidate": no_relevant,
        "no_relevant_policy": "MRR@10 and nDCG@10 are zero",
        "mrr_at_10": statistics.mean(mrr_values),
        "ndcg_at_10": statistics.mean(ndcg_values),
        "backend": reranker.device.type,
        "precision": str(next(reranker.model.parameters()).dtype),
        "truncation_max_tokens": MAX_LENGTH,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "parameter_count": sum(p.numel() for p in reranker.model.parameters()),
        "timing": {
            "scope": "whole rerank call including tokenization, scoring, sorting, and CPU results",
            "warmup_calls": 10,
            "queries_per_repeat": len(timing_rows),
            "repeat_seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "range_seconds": [min(seconds), max(seconds)],
            "memory_snapshots": memory,
            "memory_note": "Snapshots after each run, not peak usage; driver allocation includes context and cache.",
        },
    }
    (RESULT_DIR / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: MRR@10={summary['mrr_at_10']:.4f}, "
          f"nDCG@10={summary['ndcg_at_10']:.4f}, "
          f"timing median={summary['timing']['median_seconds']:.3f}s/16 queries")
    return summary


def main() -> None:
    check_metrics()
    rows, candidate_hash = load_sample()
    baseline = BaselineReranker(max_length=MAX_LENGTH, batch_size=PAIR_BATCH_SIZE)
    evaluate("minilm", baseline, rows, candidate_hash,
             "Pretrained on MS MARCO; these validation queries are not established as unseen.",
             {"model_id": MODEL_ID, "revision": MODEL_REVISION,
              "input_format": "Tokenizer query/document pair with pair special tokens and attention mask; sequence-classification logit."})
    del baseline
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    random_model = RandomReranker()
    evaluate("random_scratch", random_model, rows, candidate_hash,
             "Random initial seed-42 weights; no training. Sanity-check reference only.",
             {"model_id": "scratch_encoder_seed42",
              "checkpoint": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
              "checkpoint_sha256": json.loads(INITIAL_MANIFEST.read_text(encoding="utf-8"))["sha256"],
              "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
              "input_format": "[CLS] query [SEP] document [SEP], token-type IDs 0/1, attention mask; raw scoring logit."})


if __name__ == "__main__":
    main()
