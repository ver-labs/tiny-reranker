"""Load tokenizer assets for the scratch model without loading model weights."""

import json
from pathlib import Path
from typing import Sequence

from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "scratch.json"


def load_scratch_tokenizer(config_path: Path = DEFAULT_CONFIG) -> BertTokenizer:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = BertTokenizer.from_pretrained(
        config["tokenizer_id"], revision=config["tokenizer_revision"]
    )
    if tokenizer.vocab_size != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size does not match scratch config")
    if tokenizer.pad_token_id != config["pad_token_id"]:
        raise ValueError("Tokenizer padding ID does not match scratch config")
    return tokenizer


def encode_pairs(
    tokenizer: BertTokenizer,
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    max_length: int,
) -> BatchEncoding:
    """Encode [CLS] query [SEP] document [SEP] pairs as PyTorch tensors."""
    if not queries or len(queries) != len(documents):
        raise ValueError("Provide matching, nonempty query and document lists")
    return tokenizer(
        list(queries),
        list(documents),
        add_special_tokens=True,
        padding=True,
        truncation="longest_first",
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )


if __name__ == "__main__":
    tokenizer = load_scratch_tokenizer()
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Padding ID: {tokenizer.pad_token_id}")
