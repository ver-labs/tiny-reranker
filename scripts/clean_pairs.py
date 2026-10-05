"""Deduplicate exact query-document pairs and report conflicting labels."""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from .data import MANIFEST, query_id, split_for_query


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "results" / "pair_cleaning.json"


def source_rows(manifest: dict):
    source = load_dataset(
        manifest["dataset_id"],
        manifest["subset"],
        split=manifest["split"],
        revision=manifest["revision"],
        streaming=True,
    )
    if set(source.features) != set(manifest["columns"]):
        raise RuntimeError("Dataset schema differs from the pinned manifest")
    return source


def pair_key(query: str, document: str) -> bytes:
    # Preserve the source text exactly; normalization is only for query IDs.
    raw = json.dumps([query, document], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).digest()


def valid_triplet(row: dict) -> bool:
    return all(isinstance(row[field], str) and row[field].strip()
               for field in ("query", "positive", "negative"))


def pairs(row: dict):
    yield row["positive"], 1
    yield row["negative"], 0


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labels_by_key: dict[bytes, int] = {}
    source_count = valid_count = duplicate_occurrences = conflict_occurrences = 0

    # First pass finds every conflict before writing any cleaned examples.
    for row in source_rows(manifest):
        source_count += 1
        if not valid_triplet(row):
            continue
        valid_count += 1
        for document, label in pairs(row):
            key = pair_key(row["query"], document)
            bit = 1 << label
            previous = labels_by_key.get(key, 0)
            if previous & bit:
                duplicate_occurrences += 1
            elif previous:
                conflict_occurrences += 1
            labels_by_key[key] = previous | bit
        if source_count % 100_000 == 0:
            print(f"Audited {source_count:,} source rows", flush=True)

    if source_count != manifest["rows_in_split"]:
        raise RuntimeError("Source row count differs from the pinned manifest")

    conflicting_keys = {key for key, labels in labels_by_key.items() if labels == 3}
    clean_expected = len(labels_by_key) - len(conflicting_keys)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED / "clean_pairs.jsonl"
    conflict_path = PROCESSED / "conflicting_pairs.jsonl"
    clean_tmp = clean_path.with_suffix(".jsonl.tmp")
    conflict_tmp = conflict_path.with_suffix(".jsonl.tmp")
    written_clean: set[bytes] = set()
    written_conflicts: set[bytes] = set()
    clean_counts = {name: 0 for name in ("train", "validation", "test")}

    # Second pass writes each consistent pair once and flags each conflict once.
    with clean_tmp.open("w", encoding="utf-8") as clean_file, \
         conflict_tmp.open("w", encoding="utf-8") as conflict_file:
        for row in source_rows(manifest):
            if not valid_triplet(row):
                continue
            split = split_for_query(row["query"])
            qid = query_id(row["query"])
            for document, label in pairs(row):
                key = pair_key(row["query"], document)
                item = {
                    "pair_id": key.hex(),
                    "query_id": qid,
                    "split": split,
                    "query": row["query"],
                    "document": document,
                }
                if key in conflicting_keys:
                    if key not in written_conflicts:
                        conflict_file.write(json.dumps(
                            {**item, "labels": [0, 1]}, ensure_ascii=False
                        ) + "\n")
                        written_conflicts.add(key)
                elif key not in written_clean:
                    clean_file.write(json.dumps(
                        {**item, "label": label}, ensure_ascii=False
                    ) + "\n")
                    written_clean.add(key)
                    clean_counts[split] += 1

    if len(written_clean) != clean_expected or written_conflicts != conflicting_keys:
        raise RuntimeError("Cleaned output does not match the pair audit")
    clean_tmp.replace(clean_path)
    conflict_tmp.replace(conflict_path)

    report = {
        "dataset_id": manifest["dataset_id"],
        "subset": manifest["subset"],
        "revision": manifest["revision"],
        "pair_key": "SHA-256 of exact source query and document strings as a JSON array",
        "source_triplets": source_count,
        "valid_triplets": valid_count,
        "pair_occurrences": valid_count * 2,
        "same_label_duplicate_occurrences_removed": duplicate_occurrences,
        "conflicting_pairs_flagged_and_excluded": len(conflicting_keys),
        "opposite_label_occurrences": conflict_occurrences,
        "clean_pairs": len(written_clean),
        "clean_pairs_by_split": clean_counts,
        "clean_file": str(clean_path.relative_to(ROOT)),
        "conflict_file": str(conflict_path.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
