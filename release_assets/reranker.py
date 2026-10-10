"""Inspectable local PyTorch loader for the original scratch-trained reranker."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import BertTokenizer


class ScratchReranker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config['hidden_size']
        self.token_embeddings = nn.Embedding(config['vocab_size'], h, padding_idx=config['pad_token_id'])
        self.position_embeddings = nn.Embedding(config['max_length'], h)
        self.segment_embeddings = nn.Embedding(config['type_vocab_size'], h)
        self.embedding_norm = nn.LayerNorm(h, eps=config['layer_norm_eps'])
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.encoder_layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=h, nhead=config['num_attention_heads'],
            dim_feedforward=config['feed_forward_size'], dropout=config['dropout'],
            activation=config['activation'], layer_norm_eps=config['layer_norm_eps'],
            batch_first=config['batch_first'], norm_first=config['norm_first'], bias=True,
        ) for _ in range(config['num_layers']))
        self.final_norm = nn.LayerNorm(h, eps=config['layer_norm_eps'])
        self.scoring_head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or token_type_ids.shape != input_ids.shape:
            raise ValueError('All inputs must have shape [batch, length]')
        if input_ids.shape[1] > self.config['max_length']:
            raise ValueError('Sequence exceeds maximum length')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)[None, :, :] + self.segment_embeddings(token_type_ids)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        for layer in self.encoder_layers:
            hidden = layer(hidden, src_key_padding_mask=attention_mask == 0, is_causal=False)
        return self.scoring_head(self.final_norm(hidden)[:, 0, :]).squeeze(-1)


def encode_pairs(tokenizer, queries, documents, max_length=128, padding=True):
    if not queries or len(queries) != len(documents):
        raise ValueError('Matching nonempty query and document lists required')
    return tokenizer(list(queries), list(documents), add_special_tokens=True,
                     padding=padding, truncation='longest_first', max_length=max_length,
                     return_attention_mask=True, return_token_type_ids=True, return_tensors='pt')


def sort_results(documents, scores):
    """Descending raw logit, then original index ascending for exact ties."""
    if len(documents) != len(scores):
        raise ValueError('Every document needs one score')
    return sorted([{'index': i, 'document': d, 'score': float(s)}
                   for i, (d, s) in enumerate(zip(documents, scores))],
                  key=lambda row: (-row['score'], row['index']))


class Reranker:
    def __init__(self, path, device='cpu'):
        path = Path(path)
        manifest = json.loads((path / 'release_manifest.json').read_text())
        for name, expected in manifest['file_sha256'].items():
            with (path / name).open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError(f'Release hash mismatch: {name}')
        self.config = json.loads((path / 'config.json').read_text())
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda', 'mps'):
            raise ValueError('Supported explicit devices: cpu, cuda, mps')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        if self.device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable; no CPU fallback')
        if self.device.type == 'mps':
            import os
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                raise RuntimeError('Disable MPS CPU fallback for explicit GPU inference')
        self.tokenizer = BertTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
        if self.tokenizer.vocab_size != self.config['vocab_size'] or self.tokenizer.pad_token_id != self.config['pad_token_id']:
            raise RuntimeError('Tokenizer/config mismatch')
        self.model = ScratchReranker(self.config)
        self.model.load_state_dict(load_file(str(path / 'model.safetensors'), device='cpu'), strict=True)
        self.model.to(self.device).eval()

    def rank(self, query, documents, batch_size=16):
        """Return original-order raw logits and descending-score ranking."""
        if not isinstance(query, str) or any(not isinstance(d, str) for d in documents):
            raise TypeError('Query and documents must be strings')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        documents = list(documents)
        scores = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                chunk = documents[start:start + batch_size]
                inputs = encode_pairs(self.tokenizer, [query] * len(chunk), chunk,
                                      self.config['max_length']).to(self.device)
                values = self.model(**inputs)
                if values.shape != (len(chunk),) or not torch.isfinite(values).all():
                    raise RuntimeError('Expected one finite scalar per pair')
                scores.extend(values.cpu().tolist())
        return {'scores': scores, 'ranking': sort_results(documents, scores)}
