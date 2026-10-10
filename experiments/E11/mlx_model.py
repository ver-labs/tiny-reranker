"""Epoch-3 MLX architecture with an explicit optional inference-only fused path."""
import mlx.core as mx

from experiments.S1.mlx_model import MLXReranker
from .kernel import residual_layer_norm


class FusedMLXReranker(MLXReranker):
    def __call__(self, input_ids, attention_mask, token_type_ids, *, trace=False, use_fused=False):
        if not use_fused:
            return super().__call__(input_ids, attention_mask, token_type_ids, trace=trace)
        if self.training:
            raise ValueError('The custom Metal path is inference-only; call eval() first')
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape != token_type_ids.shape:
            raise ValueError('Inputs must have matching [batch, length] shapes')
        if not 0 < input_ids.shape[1] <= self.config.max_length:
            raise ValueError('Sequence length is outside the supported model range')
        x = self.embedding_norm(self.token_embeddings(input_ids) +
                                self.position_embeddings(mx.arange(input_ids.shape[1]))[None] +
                                self.segment_embeddings(token_type_ids))
        states = {'embedding_norm': x}
        mask = (attention_mask != 0)[:, None, None, :]
        norm_attention = self.encoder_layers[0].norm1(x)
        for i, layer in enumerate(self.encoder_layers):
            branch = layer.self_attn(norm_attention, mask)
            x, norm_ff = residual_layer_norm(x, branch, layer.norm2.weight,
                                             layer.norm2.bias, layer.norm2.eps)
            z = layer.linear1(norm_ff)
            z = .5 * z * (1. + mx.erf(z / (2. ** .5)))
            branch = layer.linear2(z)
            next_norm = self.encoder_layers[i + 1].norm1 if i + 1 < len(self.encoder_layers) else self.final_norm
            x, norm_attention = residual_layer_norm(x, branch, next_norm.weight,
                                                    next_norm.bias, next_norm.eps)
            if trace:
                states[f'encoder_layers.{i}'] = x
        x = norm_attention
        scores = self.scoring_head(x[:, 0])[:, 0]
        if trace:
            states['final_norm'] = x
            states['scores'] = scores
            return states
        return scores
