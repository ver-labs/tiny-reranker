"""Run the published MS MARCO cross-encoder as a comparison baseline."""

from dataclasses import dataclass
from math import isclose
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class ScoredDocument:
    id: str
    score: float


class BaselineReranker:
    def __init__(self, *, max_length: int = 128, batch_size: int = 2) -> None:
        if max_length < 3 or batch_size < 1:
            raise ValueError("max_length must be at least 3 and batch_size must be positive")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rank(self, query: str, documents: Sequence[Document]) -> list[ScoredDocument]:
        """Return document IDs and raw relevance scores, highest score first."""
        if not query:
            raise ValueError("query must not be empty")

        scored = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                [document.text for document in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits
            if logits.shape != (len(batch), 1):
                raise RuntimeError(f"Expected one score per document, got {tuple(logits.shape)}")
            scored.extend(
                ScoredDocument(document.id, float(score))
                for document, score in zip(batch, logits[:, 0].cpu().tolist())
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)


if __name__ == "__main__":
    reranker = BaselineReranker()
    documents = [
        Document("A", "A guide to saving and loading model state dictionaries."),
        Document("B", "A guide to growing tomatoes."),
        Document("C", "An introduction to Python lists."),
    ]
    print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
    print(f"Device: {reranker.device}")
    first = reranker.rank("How do I save a PyTorch model?", documents)
    second = reranker.rank("How do I save a PyTorch model?", documents)
    if [item.id for item in first] != [item.id for item in second] or not all(
        isclose(a.score, b.score, rel_tol=0, abs_tol=1e-4)
        for a, b in zip(first, second)
    ):
        raise RuntimeError("Baseline results changed between repeated calls")

    for result in first:
        print(f"{result.id}: {result.score:.6f}")
    print("Repeat check: same ranking and scores within 1e-4")
