"""Stream the pinned MS MARCO triplets and prepare query-document pairs."""

import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from .scratch_tokenizer import encode_pairs


MANIFEST = Path(__file__).resolve().parent.parent / "data" / "manifests" / "msmarco_bm25_triplet.json"


def normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def query_id(query: str) -> str:
    return hashlib.sha256(normalized_query(query).encode("utf-8")).hexdigest()


def split_for_query(query: str) -> str:
    bucket = int(query_id(query)[:16], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


class TripletStream(IterableDataset):
    def __init__(self, split: str, *, shuffle: bool = False, seed: int = 42,
                 skip_triplets: int = 0) -> None:
        if split not in {"train", "validation", "test", "heldout"}:
            raise ValueError("split must be train, validation, test, or heldout")
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.skip_triplets = skip_triplets
        self.source = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def __iter__(self) -> Iterator[dict[str, str]]:
        source = load_dataset(
            self.source["dataset_id"],
            self.source["subset"],
            split=self.source["split"],
            revision=self.source["revision"],
            streaming=True,
        )
        if set(source.features) != set(self.source["columns"]):
            raise RuntimeError("Dataset schema differs from the pinned manifest")
        if self.shuffle:
            source = source.shuffle(seed=self.seed, buffer_size=10_000)

        seen: set[str] = set()
        accepted = 0
        for row in source:
            query, positive, negative = (row[name] for name in ("query", "positive", "negative"))
            if not all(isinstance(value, str) and value.strip()
                       for value in (query, positive, negative)):
                continue
            assigned_split = split_for_query(query)
            if self.split == "heldout" and assigned_split == "train":
                continue
            if self.split != "heldout" and assigned_split != self.split:
                continue
            row_id = hashlib.sha256(
                json.dumps([query, positive, negative], ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if row_id in seen:
                continue
            seen.add(row_id)
            if accepted < self.skip_triplets:
                accepted += 1
                continue
            accepted += 1
            yield {
                "row_id": row_id,
                "query_id": query_id(query),
                "query": query,
                "positive": positive,
                "negative": negative,
            }


def collate_triplets(rows: list[dict[str, str]], tokenizer, max_length: int) -> dict:
    queries = [row["query"] for row in rows for _ in range(2)]
    documents = [document for row in rows
                 for document in (row["positive"], row["negative"])]
    labels = torch.tensor([label for _ in rows for label in (1.0, 0.0)], dtype=torch.float32)
    return {
        "inputs": encode_pairs(tokenizer, queries, documents, max_length=max_length),
        "labels": labels,
        "query_ids": [row["query_id"] for row in rows],
        "row_ids": [row["row_id"] for row in rows],
    }
