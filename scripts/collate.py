"""Tokenize cleaned query-document pairs and pad each batch dynamically."""

import torch
from torch.utils.data import DataLoader

from .model import ScratchConfig
from .scratch_tokenizer import encode_pairs, load_scratch_tokenizer


class PairCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        if max_length < 3:
            raise ValueError("max_length must allow the pair's special tokens")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict]) -> dict:
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        labels = [row["label"] for row in rows]
        if any(label not in (0, 1) for label in labels):
            raise ValueError("Each pair label must be 0 or 1")
        return {
            "inputs": encode_pairs(
                self.tokenizer,
                [row["query"] for row in rows],
                [row["document"] for row in rows],
                max_length=self.max_length,
            ),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "query_ids": [row["query_id"] for row in rows],
            "document_ids": [row["document_id"] for row in rows],
            "pair_ids": [row["pair_id"] for row in rows],
        }


def make_pair_loader(dataset, tokenizer, *, batch_size: int, max_length: int,
                     shuffle: bool = False) -> DataLoader:
    """Start with data loading in the main process for straightforward debugging."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairCollator(tokenizer, max_length),
        num_workers=0,
    )


def main() -> None:
    config = ScratchConfig.from_file()
    tokenizer = load_scratch_tokenizer()
    rows = [
        {
            "query": "What is a reranker?",
            "document": "A model that sorts candidate passages.",
            "label": 1,
            "query_id": "example-query",
            "document_id": "short-document",
            "pair_id": "short-pair",
        },
        {
            "query": "What is a reranker?",
            "document": "A longer passage with extra words explaining how a model scores "
                        "query and document pairs before sorting them by relevance.",
            "label": 0,
            "query_id": "example-query",
            "document_id": "long-document",
            "pair_id": "long-pair",
        },
    ]
    loader = make_pair_loader(rows, tokenizer, batch_size=2,
                              max_length=config.max_length)
    assert loader.num_workers == 0
    batch = next(iter(loader))
    inputs = batch["inputs"]
    lengths = inputs["attention_mask"].sum(dim=1).tolist()
    assert inputs["input_ids"].shape[0] == 2
    assert inputs["input_ids"].shape[1] == max(lengths) < config.max_length
    assert (inputs["attention_mask"] == 0).any()
    for index, row in enumerate(rows):
        expected_ids = encode_pairs(
            tokenizer, [row["query"]], [row["document"]],
            max_length=config.max_length,
        )["input_ids"][0]
        assert torch.equal(inputs["input_ids"][index, :len(expected_ids)], expected_ids)
        assert batch["labels"][index].item() == float(row["label"])
        assert batch["query_ids"][index] == row["query_id"]
        assert batch["document_ids"][index] == row["document_id"]
        assert batch["pair_ids"][index] == row["pair_id"]
    print(f"Batch shape: {tuple(inputs['input_ids'].shape)}")
    print(f"Token lengths before padding: {lengths}")
    print(f"Padding tokens added: {int((inputs['attention_mask'] == 0).sum())}")
    print(f"DataLoader workers: {loader.num_workers}")
    print("Tokenized pairs, labels, and IDs stay aligned: True")
    print(f"Labels: {batch['labels'].tolist()}; IDs retained: {batch['pair_ids']}")


if __name__ == "__main__":
    main()
