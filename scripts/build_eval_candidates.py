"""Build fixed candidate lists for a constructed reranker evaluation."""

import hashlib
import json
from pathlib import Path

from .data import document_id, query_id, split_for_query


ROOT = Path(__file__).resolve().parent.parent
CLEAN_PAIRS = ROOT / "data" / "processed" / "clean_pairs.jsonl"
SPLIT_MANIFEST = ROOT / "data" / "manifests" / "query_split.json"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "results" / "evaluation_candidates.json"
EVAL_SPLITS = ("validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not CLEAN_PAIRS.exists():
        raise FileNotFoundError("Run python -m scripts.clean_pairs first")

    groups: dict[str, dict[str, dict]] = {split: {} for split in EVAL_SPLITS}
    pair_count = 0
    duplicate_doc_ids = 0
    conflicting_doc_ids: dict[str, set[str]] = {}
    with CLEAN_PAIRS.open(encoding="utf-8") as source:
        for line in source:
            pair_count += 1
            pair = json.loads(line)
            split = pair["split"]
            if split not in EVAL_SPLITS:
                continue
            qid = pair["query_id"]
            if qid != query_id(pair["query"]) or split != split_for_query(pair["query"]):
                raise RuntimeError("Clean pair has an inconsistent query ID or split")
            group = groups[split].setdefault(qid, {
                "query": pair["query"], "candidates": {}
            })
            # Select a stable display form if the same normalized query has variants.
            group["query"] = min(group["query"], pair["query"])
            doc_id = document_id(pair["document"])
            previous = group["candidates"].get(doc_id)
            if previous is None:
                group["candidates"][doc_id] = {
                    "document_id": doc_id,
                    "text": pair["document"],
                    "label": pair["label"],
                }
            else:
                if previous["text"] != pair["document"]:
                    raise RuntimeError("Document hash collision")
                duplicate_doc_ids += 1
                if previous["label"] != pair["label"]:
                    conflicting_doc_ids.setdefault(qid, set()).add(doc_id)

    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    report = {
        "evaluation_kind": "constructed MS MARCO mined-triplet candidate lists",
        "dataset_revision": json.loads(
            (ROOT / "data" / "manifests" / "msmarco_bm25_triplet.json").read_text(
                encoding="utf-8"
            )
        )["revision"],
        "input": str(CLEAN_PAIRS.relative_to(ROOT)),
        "input_sha256": file_sha256(CLEAN_PAIRS),
        "document_id": "SHA-256 of exact UTF-8 document text",
        "candidate_order": "ascending document ID",
        "use_same_saved_files_for_all_models": True,
        "scope": "validation and test queries with at least one unambiguous clean candidate; zero-relevant lists are included",
        "baseline_training_history": "External baseline was already trained on MS MARCO; unseen queries are not established for it.",
        "scratch_training_rule": "Do not train on validation or test query IDs.",
        "archived_checkpoint_limit": "The archived all-data scratch checkpoint saw held-out queries and is not eligible for held-out evaluation.",
        "comparison_limit": "These constructed-list results are not directly comparable to published leaderboard scores.",
        "clean_pairs_read": pair_count,
        "duplicate_document_id_occurrences": duplicate_doc_ids,
        "ambiguous_document_ids_excluded": sum(map(len, conflicting_doc_ids.values())),
        "splits": {},
    }

    for split in EVAL_SPLITS:
        path = OUTPUT_DIR / f"eval_{split}_candidates.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        query_count = candidate_count = empty_after_dedup = 0
        single_label_queries = no_relevant_queries = 0
        with temporary.open("w", encoding="utf-8") as output:
            for qid in sorted(groups[split]):
                group = groups[split][qid]
                ambiguous = conflicting_doc_ids.get(qid, set())
                candidates = [
                    group["candidates"][doc_id]
                    for doc_id in sorted(group["candidates"])
                    if doc_id not in ambiguous
                ]
                positive_count = sum(item["label"] == 1 for item in candidates)
                negative_count = len(candidates) - positive_count
                if not candidates:
                    empty_after_dedup += 1
                    continue
                single_label_queries += positive_count == 0 or negative_count == 0
                no_relevant_queries += positive_count == 0
                output.write(json.dumps({
                    "query_id": qid,
                    "query": group["query"],
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "candidates": candidates,
                }, ensure_ascii=False) + "\n")
                query_count += 1
                candidate_count += len(candidates)
        temporary.replace(path)
        source_queries = split_manifest["verified_unique_query_ids"][split]
        report["splits"][split] = {
            "candidate_file": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
            "source_query_ids": source_queries,
            "queries_with_clean_pairs": len(groups[split]),
            "queries_with_no_clean_pairs": source_queries - len(groups[split]),
            "queries_empty_after_document_dedup": empty_after_dedup,
            "single_label_queries_included": single_label_queries,
            "queries_without_relevant_candidate_included": no_relevant_queries,
            "evaluable_queries": query_count,
            "candidates": candidate_count,
        }
        if source_queries != query_count + empty_after_dedup + report["splits"][split]["queries_with_no_clean_pairs"]:
            raise RuntimeError("Evaluation query counts do not reconcile")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
