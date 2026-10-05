"""Score every fixed validation candidate list with MiniLM and random scratch."""

import json
from pathlib import Path

import torch

from .baselines import BaselineReranker, MODEL_ID, MODEL_REVISION
from .evaluate_initial import (
    CANDIDATES, CANDIDATE_MANIFEST, MAX_LENGTH, RandomReranker,
    check_metrics, ranking_metrics, sha256_file,
)
from .scratch_tokenizer import encode_pairs


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "full_validation"
PAIR_BATCH_SIZE = 64
QUERY_BUFFER_SIZE = 32


@torch.inference_mode()
def score_pairs(reranker, queries: list[str], documents: list[str]) -> list[float]:
    if isinstance(reranker, BaselineReranker):
        inputs = reranker.tokenizer(
            queries, documents, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(reranker.device)
        values = reranker.model(**inputs).logits[:, 0]
    else:
        inputs = encode_pairs(reranker.tokenizer, queries, documents,
                              max_length=MAX_LENGTH).to(reranker.device)
        values = reranker.model(**inputs)
    if values.shape != (len(queries),) or not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite score or incorrect score count")
    return [float(value) for value in values.cpu().tolist()]


def evaluate_buffer(reranker, rows: list[dict], output) -> tuple[int, float, float, int]:
    pairs = [(row_index, candidate) for row_index, row in enumerate(rows)
             for candidate in row["candidates"]]
    scores: list[list[float]] = [[] for _ in rows]
    for start in range(0, len(pairs), PAIR_BATCH_SIZE):
        chunk = pairs[start:start + PAIR_BATCH_SIZE]
        values = score_pairs(
            reranker,
            [rows[row_index]["query"] for row_index, _ in chunk],
            [candidate["text"] for _, candidate in chunk],
        )
        for (row_index, _), value in zip(chunk, values):
            scores[row_index].append(value)

    mrr_total = ndcg_total = 0.0
    no_relevant = 0
    for row, row_scores in zip(rows, scores):
        if len(row_scores) != len(row["candidates"]):
            raise RuntimeError("A candidate is missing a score")
        ranking = sorted(({
            "document_id": candidate["document_id"],
            "label": candidate["label"],
            "score": score,
        } for candidate, score in zip(row["candidates"], row_scores)),
            key=lambda item: (-item["score"], item["document_id"]))
        if len({item["document_id"] for item in ranking}) != len(ranking):
            raise RuntimeError("Duplicate document IDs in a candidate list")
        labels = [item["label"] for item in ranking]
        no_relevant += not any(labels)
        mrr, ndcg = ranking_metrics(labels)
        mrr_total += mrr
        ndcg_total += ndcg
        output.write(json.dumps({
            "query_id": row["query_id"], "mrr_at_10": mrr,
            "ndcg_at_10": ndcg, "ranking": ranking,
        }) + "\n")
    return len(rows), mrr_total, ndcg_total, no_relevant


def evaluate_model(name: str, reranker, expected: dict,
                   candidate_hash: str, training_history: str,
                   extra_metadata: dict | None = None) -> dict:
    expected_queries = expected["evaluable_queries"]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rankings = RESULT_DIR / f"{name}_rankings.jsonl"
    temporary = rankings.with_suffix(".jsonl.tmp")
    count = no_relevant = 0
    mrr_total = ndcg_total = 0.0
    with CANDIDATES.open(encoding="utf-8") as source, \
         temporary.open("w", encoding="utf-8") as output:
        buffered: list[dict] = []
        for line in source:
            buffered.append(json.loads(line))
            if len(buffered) >= QUERY_BUFFER_SIZE:
                n, mrr, ndcg, missing = evaluate_buffer(reranker, buffered, output)
                count += n
                mrr_total += mrr
                ndcg_total += ndcg
                no_relevant += missing
                buffered.clear()
                if count % 4096 == 0:
                    print(f"{name}: scored {count:,}/{expected_queries:,} queries", flush=True)
        if buffered:
            n, mrr, ndcg, missing = evaluate_buffer(reranker, buffered, output)
            count += n
            mrr_total += mrr
            ndcg_total += ndcg
            no_relevant += missing
    if count != expected_queries:
        raise RuntimeError(f"Expected {expected_queries} queries, scored {count}")
    temporary.replace(rankings)
    summary = {
        "model": name,
        "model_id": MODEL_ID if name == "minilm" else "scratch_encoder_seed42",
        "revision": MODEL_REVISION if name == "minilm" else None,
        "training_history": training_history,
        "evaluation": "constructed MS MARCO mined-triplet validation candidate lists; not leaderboard comparable",
        "candidate_sha256": candidate_hash,
        "rankings_file": str(rankings.relative_to(ROOT)),
        "queries": count,
        "source_query_ids": expected["source_query_ids"],
        "metric_denominator": count,
        "queries_excluded_no_clean_pairs": expected["queries_with_no_clean_pairs"],
        "queries_excluded_empty_after_document_dedup": expected["queries_empty_after_document_dedup"],
        "zero_candidate_policy": "Cannot rank an empty list; exclude from metric denominator and report the count.",
        "queries_without_relevant_candidate": no_relevant,
        "no_relevant_policy": "Include nonempty candidate lists; MRR@10 and nDCG@10 are zero when no candidate is relevant.",
        "mrr_at_10": mrr_total / count,
        "ndcg_at_10": ndcg_total / count,
        "tie_break": "document ID ascending when scores tie",
        "backend": reranker.device.type,
        "precision": str(next(reranker.model.parameters()).dtype),
        "truncation_max_tokens": MAX_LENGTH,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "input_format": (
            "Tokenizer query/document pair with pair special tokens and attention mask"
            if name == "minilm" else
            "[CLS] query [SEP] document [SEP] with token-type IDs and attention mask"
        ),
    }
    if extra_metadata:
        summary.update(extra_metadata)
    (RESULT_DIR / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: {count:,} queries; MRR@10={summary['mrr_at_10']:.4f}; "
          f"nDCG@10={summary['ndcg_at_10']:.4f}", flush=True)
    return summary


def main() -> None:
    check_metrics()
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["splits"]["validation"]
    if sha256_file(CANDIDATES) != expected["sha256"]:
        raise RuntimeError("Validation candidate file differs from its manifest")
    baseline = BaselineReranker(max_length=MAX_LENGTH, batch_size=PAIR_BATCH_SIZE)
    evaluate_model("minilm", baseline, expected, expected["sha256"],
                   "Pretrained on MS MARCO; validation queries may have been seen.")
    del baseline
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    random_model = RandomReranker()
    evaluate_model("random_scratch", random_model, expected,
                   expected["sha256"],
                   "Saved random initial weights; no training. Sanity-check reference only.")


if __name__ == "__main__":
    main()
