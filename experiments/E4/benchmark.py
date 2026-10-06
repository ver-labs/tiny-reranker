"""Benchmark one pair batch size on fixed grouped validation queries."""

import argparse
import hashlib
import json
import statistics
import time
from itertools import islice
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, sha256_file
from scripts.evaluate_validation import MAX_LENGTH
from scripts.scratch_tokenizer import encode_pairs

from experiments.E1.evaluate_variant import TrainedVariant


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_FOLDER = ROOT / "experiments" / "E1" / "four_layer"
QUERY_COUNT = 128
QUERY_GROUP_SIZE = 8
REPEATS = 3
WARMUPS = 10


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


@torch.inference_mode()
def score_batch(reranker: TrainedVariant, pairs: list[tuple[str, str]]) -> tuple[list[float], dict]:
    inputs = encode_pairs(
        reranker.tokenizer, [query for query, _ in pairs],
        [document for _, document in pairs], max_length=MAX_LENGTH,
    ).to(reranker.device)
    logits = reranker.model(**inputs)
    if logits.shape != (len(pairs),) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Invalid batched scores")
    memory = {
        "current_tensor_bytes": torch.mps.current_allocated_memory(),
        "current_driver_bytes": torch.mps.driver_allocated_memory(),
    }
    return logits.cpu().tolist(), memory


def rerank_group(reranker: TrainedVariant, rows: list[dict],
                 pair_batch_size: int) -> tuple[list[dict], list[float], list[dict]]:
    """Return each query as soon as its last pair score is CPU-readable."""
    flattened = [
        (query_index, candidate)
        for query_index, row in enumerate(rows)
        for candidate in row["candidates"]
    ]
    scores: list[list[tuple[dict, float]]] = [[] for _ in rows]
    completions: list[dict | None] = [None] * len(rows)
    latencies: list[float | None] = [None] * len(rows)
    memory_samples = []
    torch.mps.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(flattened), pair_batch_size):
        batch = flattened[offset:offset + pair_batch_size]
        values, memory = score_batch(reranker, [
            (rows[index]["query"], candidate["text"])
            for index, candidate in batch
        ])
        memory_samples.append(memory)
        for (index, candidate), value in zip(batch, values):
            scores[index].append((candidate, float(value)))
            if len(scores[index]) == len(rows[index]["candidates"]):
                ranking = sorted(({
                    "document_id": item["document_id"],
                    "label": item["label"],
                    "score": score,
                } for item, score in scores[index]),
                    key=lambda item: (-item["score"], item["document_id"]))
                completions[index] = {"query_id": rows[index]["query_id"],
                                      "ranking": ranking}
                latencies[index] = time.perf_counter() - started
    torch.mps.synchronize()
    if any(item is None for item in completions + latencies):
        raise RuntimeError("A query was not fully ranked")
    return completions, latencies, memory_samples


def run_once(reranker: TrainedVariant, rows: list[dict], pair_batch_size: int) -> dict:
    all_results = []
    all_latencies = []
    all_memory = []
    torch.mps.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(rows), QUERY_GROUP_SIZE):
        results, latencies, memory = rerank_group(
            reranker, rows[offset:offset + QUERY_GROUP_SIZE], pair_batch_size
        )
        all_results.extend(results)
        all_latencies.extend(latencies)
        all_memory.extend(memory)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    pairs = sum(len(row["candidates"]) for row in rows)
    return {
        "seconds": seconds,
        "pairs_per_second": pairs / seconds,
        "queries_per_second": len(rows) / seconds,
        "per_query_latency_median_seconds": statistics.median(all_latencies),
        "per_query_latency_p95_seconds": percentile(all_latencies, .95),
        "per_query_latency_max_seconds": max(all_latencies),
        "per_query_latency_seconds": all_latencies,
        "rankings": all_results,
        "memory": {
            "sampled_max_current_tensor_bytes": max(
                item["current_tensor_bytes"] for item in all_memory
            ),
            "sampled_max_current_driver_bytes": max(
                item["current_driver_bytes"] for item in all_memory
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-batch-size", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an earlier benchmark after changing the method")
    args = parser.parse_args()
    report_path = HERE / f"batch_{args.pair_batch_size}.json"
    if report_path.exists() and not args.overwrite:
        parser.error(f"Batch-size {args.pair_batch_size} output already exists")
    if not torch.backends.mps.is_available():
        raise RuntimeError("E4 MPS comparison requires MPS")
    parent = json.loads((MODEL_FOLDER / "epoch1.json").read_text(encoding="utf-8"))
    reranker = TrainedVariant(MODEL_FOLDER, parent)
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    candidate_hash = sha256_file(CANDIDATES)
    if candidate_hash != manifest["splits"]["validation"]["sha256"]:
        raise RuntimeError("Candidate lists differ from their manifest")
    with CANDIDATES.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in islice(source, QUERY_COUNT)]
    if len(rows) != QUERY_COUNT or any(not row["candidates"] for row in rows):
        raise RuntimeError("Incomplete E4 query sample")
    sample_hash = hashlib.sha256(json.dumps(
        [(row["query_id"], [candidate["document_id"]
                            for candidate in row["candidates"]]) for row in rows],
        separators=(",", ":")
    ).encode()).hexdigest()
    # Model/tokenizer loading and warmups are outside every timed repetition.
    warmup_pairs = [
        (row["query"], candidate["text"])
        for row in rows[:QUERY_GROUP_SIZE]
        for candidate in row["candidates"]
    ][:args.pair_batch_size]
    for _ in range(WARMUPS):
        score_batch(reranker, warmup_pairs)
    # Warm every query and its actual token shape once before timed repetitions.
    run_once(reranker, rows, args.pair_batch_size)
    torch.mps.synchronize()
    runs = []
    for repeat in range(1, REPEATS + 1):
        run = run_once(reranker, rows, args.pair_batch_size)
        runs.append(run)
        print(f"batch={args.pair_batch_size} repeat={repeat}: "
              f"{run['pairs_per_second']:.1f} pairs/s; "
              f"query median={run['per_query_latency_median_seconds'] * 1000:.2f} ms",
              flush=True)
    first = runs[0]["rankings"]
    for run in runs[1:]:
        if [row["query_id"] for row in run["rankings"]] != \
                [row["query_id"] for row in first]:
            raise RuntimeError("E4 query order changed between repeats")
        if [[item["document_id"] for item in row["ranking"]]
            for row in run["rankings"]] != \
                [[item["document_id"] for item in row["ranking"]]
                 for row in first]:
            raise RuntimeError("E4 ranking order changed between repeats")
    pair_count = sum(len(row["candidates"]) for row in rows)
    # Keep one CPU-readable ranking per query, plus timing for all repetitions.
    report = {
        "experiment": "E4 inference pair batching",
        "pair_batch_size": args.pair_batch_size,
        "checkpoint": "experiments/E1/four_layer/epoch1.pt",
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "candidate_file_sha256": candidate_hash,
        "selected_query_candidate_ids_sha256": sample_hash,
        "sample_rule": "first 128 nonempty validation candidate lists, original file order",
        "query_count": QUERY_COUNT,
        "pair_count": pair_count,
        "candidate_count_distribution": {
            str(n): sum(len(row["candidates"]) == n for row in rows)
            for n in sorted({len(row["candidates"]) for row in rows})
        },
        "query_group_size": QUERY_GROUP_SIZE,
        "query_latency_definition": "time from start of an eight-query group until that query has CPU-readable sorted results",
        "timing_scope": "query and document tokenization, device transfer, model scoring, CPU score transfer, sorting, returning results; excludes model loading",
        "warmup_calls": WARMUPS,
        "untimed_full_sample_warmup_passes": 1,
        "repeats": REPEATS,
        "max_length": MAX_LENGTH,
        "precision": "float32",
        "backend": "mps",
        "memory_label": "maximum sampled MPS current allocation while input and score tensors are live; not a true peak",
        "runs": [{key: value for key, value in run.items() if key != "rankings"} for run in runs],
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "median_pairs_per_second": statistics.median(run["pairs_per_second"] for run in runs),
        "median_query_latency_seconds": statistics.median(
            run["per_query_latency_median_seconds"] for run in runs
        ),
        "median_query_p95_seconds": statistics.median(
            run["per_query_latency_p95_seconds"] for run in runs
        ),
        "sampled_max_current_tensor_bytes": max(
            run["memory"]["sampled_max_current_tensor_bytes"] for run in runs
        ),
        "sampled_max_current_driver_bytes": max(
            run["memory"]["sampled_max_current_driver_bytes"] for run in runs
        ),
        "rankings": first,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", report_path,
                      replace_existing=args.overwrite)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
