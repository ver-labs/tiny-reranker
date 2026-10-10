"""Same scratch weights with matched explicit PyTorch and Triton inference graphs."""
import torch
import torch.nn.functional as F

from scripts.model import ScratchReranker
from .kernel import reference, residual_layer_norm


class TritonReranker(ScratchReranker):
    def forward(self, input_ids, attention_mask, token_type_ids, *, kernel_mode='original', num_warps=4):
        if kernel_mode == 'original':
            return super().forward(input_ids, attention_mask, token_type_ids)
        if kernel_mode not in ('reference', 'triton'):
            raise ValueError('kernel_mode must be original, reference or triton')
        if self.training:
            raise RuntimeError('Explicit kernel paths support inference only; call eval()')
        if kernel_mode == 'triton' and torch.is_grad_enabled():
            raise RuntimeError('Use torch.inference_mode() for the Triton model')
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or token_type_ids.shape != input_ids.shape:
            raise ValueError('Model inputs must have matching [batch, length] shapes')
        if not 0 < input_ids.shape[1] <= self.config.max_length:
            raise ValueError('Sequence exceeds supported model length')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embedding_dropout(self.embedding_norm(self.token_embeddings(input_ids) +
            self.position_embeddings(positions)[None] + self.segment_embeddings(token_type_ids)))
        padding_mask = attention_mask == 0
        normalized = self.encoder_layers[0].norm1(x)
        for i, layer in enumerate(self.encoder_layers):
            attention = layer.self_attn(normalized, normalized, normalized,
                                        key_padding_mask=padding_mask, need_weights=False)[0]
            # Batch-first MHA can return a transposed, noncontiguous view. Both
            # matched paths include packing; stock-original timing stays separate.
            attention = layer.dropout1(attention).contiguous()
            if kernel_mode == 'triton':
                x, ff_input = residual_layer_norm(x, attention, layer.norm2.weight,
                    layer.norm2.bias, layer.norm2.eps, fallback=False, num_warps=num_warps)
            else:
                x, ff_input = reference(x, attention, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
            branch = layer.dropout2(layer.linear2(layer.dropout(F.gelu(layer.linear1(ff_input), approximate='none')))).contiguous()
            next_norm = self.encoder_layers[i + 1].norm1 if i + 1 < len(self.encoder_layers) else self.final_norm
            if kernel_mode == 'triton':
                x, normalized = residual_layer_norm(x, branch, next_norm.weight,
                    next_norm.bias, next_norm.eps, fallback=False, num_warps=num_warps)
            else:
                x, normalized = reference(x, branch, next_norm.weight, next_norm.bias, next_norm.eps)
        return self.scoring_head(normalized[:, 0]).squeeze(-1)
