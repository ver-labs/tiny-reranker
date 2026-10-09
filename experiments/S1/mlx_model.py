"""FP32 inference port of ScratchReranker, with exact GELU and pre-norm layers."""
import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, config):
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.heads = config.num_attention_heads

    def __call__(self, x, mask):
        batch, length, hidden = x.shape
        q, k, v = mx.split(self.in_proj(x), 3, axis=-1)
        q, k, v = [a.reshape(batch, length, self.heads, hidden // self.heads)
                   .transpose(0, 2, 1, 3) for a in (q, k, v)]
        x = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(hidden // self.heads) ** -0.5, mask=mask)
        return self.out_proj(x.transpose(0, 2, 1, 3).reshape(batch, length, hidden))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = Attention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.linear2 = nn.Linear(config.feed_forward_size, config.hidden_size)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        z = self.linear1(self.norm2(x))
        z = 0.5 * z * (1.0 + mx.erf(z / (2.0 ** 0.5)))
        return x + self.linear2(z)


class MLXReranker(nn.Module):
    def __init__(self, config):
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.encoder_layers = [EncoderLayer(config) for _ in range(config.num_layers)]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.scoring_head = nn.Linear(config.hidden_size, 1)

    def __call__(self, input_ids, attention_mask, token_type_ids, *, trace=False):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or \
                input_ids.shape != token_type_ids.shape:
            raise ValueError("Inputs must have matching [batch, length] shapes")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds model maximum")
        x = self.embedding_norm(self.token_embeddings(input_ids) +
            self.position_embeddings(mx.arange(input_ids.shape[1]))[None] +
            self.segment_embeddings(token_type_ids))
        states = {"embedding_norm": x}
        mask = (attention_mask != 0)[:, None, None, :]
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)
            if trace:
                states[f"encoder_layers.{i}"] = x
        x = self.final_norm(x)
        scores = self.scoring_head(x[:, 0])[:, 0]
        if trace:
            states["final_norm"] = x
            states["scores"] = scores
            return states
        return scores
