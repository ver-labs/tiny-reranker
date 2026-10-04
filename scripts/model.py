"""Small bidirectional reranker with all model weights initialized from scratch."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "scratch.json"


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int
    pad_token_id: int
    max_length: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    feed_forward_size: int
    dropout: float

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CONFIG) -> "ScratchConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})

    def __post_init__(self) -> None:
        if self.vocab_size < 1 or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("Invalid vocabulary size or padding ID")
        if min(self.max_length, self.num_layers, self.hidden_size, self.num_attention_heads,
               self.feed_forward_size) < 1:
            raise ValueError("Model dimensions and layer count must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Hidden size must be divisible by attention-head count")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class ScratchReranker(nn.Module):
    """Score tokenized query-document pairs without any pretrained model weights."""

    def __init__(self, config: ScratchConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Construct each layer separately so it gets its own initialized weights.
        self.encoder_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feed_forward_size,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.scoring_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> Tensor:
        """Return one raw relevance score for each query-document pair."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        if token_type_ids.shape != input_ids.shape:
            raise ValueError("token_type_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds configured maximum length")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(positions)[None, :, :]
            + self.segment_embeddings(token_type_ids)
        )
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        padding_mask = attention_mask == 0
        for layer in self.encoder_layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask, is_causal=False)
        hidden = self.final_norm(hidden)
        return self.scoring_head(hidden[:, 0, :]).squeeze(-1)


if __name__ == "__main__":
    from .scratch_tokenizer import encode_pairs, load_scratch_tokenizer

    torch.manual_seed(42)
    config = ScratchConfig.from_file()
    tokenizer = load_scratch_tokenizer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ScratchReranker(config).to(device).eval()
    inputs = encode_pairs(
        tokenizer,
        ["How do I save a PyTorch model?"] * 2,
        ["A guide to saving model state dictionaries.", "A guide to growing tomatoes."],
        max_length=config.max_length,
    ).to(device)
    padding = inputs["attention_mask"] == 0
    changed_padding = {name: tensor.clone() for name, tensor in inputs.items()}
    changed_padding["input_ids"][padding] = tokenizer.unk_token_id
    changed_padding["token_type_ids"][padding] = 1
    with torch.inference_mode():
        scores = model(**inputs)
        changed_scores = model(**changed_padding)
    finite = bool(torch.isfinite(scores).all())
    padding_ignored = bool(torch.allclose(scores, changed_scores, atol=1e-5, rtol=0))
    assert scores.shape == (2,) and finite and padding.any() and padding_ignored
    print(f"Device: {device}")
    print(f"Input IDs shape: {tuple(inputs['input_ids'].shape)}")
    print(f"Attention mask shape: {tuple(inputs['attention_mask'].shape)}")
    print(f"Segment IDs shape: {tuple(inputs['token_type_ids'].shape)}")
    print(f"Output shape: {tuple(scores.shape)}")
    print(f"All scores finite: {finite}")
    print(f"Padded positions: {int(padding.sum())}; ignored by scores: {padding_ignored}")
    print(f"Scores: {scores.tolist()} (random weights, not trained)")
