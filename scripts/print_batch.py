"""Print one complete, small batch made from real cleaned training pairs."""

import json
from itertools import islice

from .collate import PairCollator
from .data import iter_clean_pairs
from .model import ScratchConfig
from .scratch_tokenizer import load_scratch_tokenizer


def main() -> None:
    tokenizer = load_scratch_tokenizer()
    max_length = ScratchConfig.from_file().max_length
    sample = list(islice(iter_clean_pairs("train"), 500))

    def token_count(row: dict) -> int:
        return len(tokenizer(
            row["query"], row["document"],
            truncation=True, max_length=max_length,
        )["input_ids"])

    # Pick one compact example of each label so the full batch fits on screen.
    rows = [min((row for row in sample if row["label"] == label), key=token_count)
            for label in (1, 0)]
    batch = PairCollator(tokenizer, max_length)(rows)
    printable = {
        "inputs": {name: values.tolist() for name, values in batch["inputs"].items()},
        "labels": batch["labels"].tolist(),
        "query_ids": batch["query_ids"],
        "document_ids": batch["document_ids"],
        "pair_ids": batch["pair_ids"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
