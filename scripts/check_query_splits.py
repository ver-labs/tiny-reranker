"""Audit query-level split assignment across the pinned triplet dataset."""

import json

from datasets import load_dataset

from .data import MANIFEST, query_id, split_for_query


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = load_dataset(
        manifest["dataset_id"],
        manifest["subset"],
        split=manifest["split"],
        revision=manifest["revision"],
        streaming=True,
    )
    if set(source.features) != set(manifest["columns"]):
        raise RuntimeError("Dataset schema differs from the pinned manifest")

    ids = {name: set() for name in ("train", "validation", "test")}
    triplets = {name: 0 for name in ids}
    total = 0
    for row in source:
        total += 1
        if not all(isinstance(row[field], str) and row[field].strip()
                   for field in ("query", "positive", "negative")):
            continue
        split = split_for_query(row["query"])
        ids[split].add(query_id(row["query"]))
        triplets[split] += 1
        if total % 100_000 == 0:
            print(f"Checked {total:,} source rows", flush=True)

    overlaps = {
        "train/validation": len(ids["train"] & ids["validation"]),
        "train/test": len(ids["train"] & ids["test"]),
        "validation/test": len(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Query IDs overlap: {overlaps}")
    print(f"Source rows checked: {total:,}")
    for name in ids:
        print(f"{name}: {triplets[name]:,} triplets, {len(ids[name]):,} unique query IDs")
    print(f"Query ID overlaps: {overlaps}")


if __name__ == "__main__":
    main()
