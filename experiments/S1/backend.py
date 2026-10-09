"""One selected checkpoint, common tokenization and ranking for both backends."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FOLDER = ROOT / "experiments/E1/four_layer"
REFERENCE = ROOT / "runs/full_train"
CHECKPOINT = REFERENCE / "epoch_0003.pt"
WEIGHTS = HERE / "epoch_0003_mlx.safetensors"


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    from scripts.checkpoint_io import atomic_write_text
    atomic_write_text(json.dumps(data, indent=2) + '\n', path, replace_existing=True)


def source_metadata():
    report = json.loads((REFERENCE / "epoch_0003_summary.json").read_text())
    provenance = json.loads((REFERENCE / "config.json").read_text())
    report["config_sha256"] = provenance["config_sha256"]
    if sha256(CHECKPOINT) != report["checkpoint_sha256"] or \
            sha256(FOLDER / "config.json") != report["config_sha256"]:
        raise RuntimeError("Selected checkpoint/config hash mismatch")
    return report


def convert():
    import torch
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from scripts.model import ScratchConfig
    from .mlx_model import MLXReranker
    report = source_metadata()
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    config = ScratchConfig.from_file(FOLDER / 'config.json')
    if saved['provenance_sha256'] != sha256(REFERENCE / 'config.json'):
        raise RuntimeError('Checkpoint config mismatch')
    model = MLXReranker(config)
    expected = dict(tree_flatten(model.parameters()))
    mapped, mapping = {}, []
    for name, tensor in saved['model_state_dict'].items():
        target = name.replace('self_attn.in_proj_weight', 'self_attn.in_proj.weight') \
                     .replace('self_attn.in_proj_bias', 'self_attn.in_proj.bias')
        if target not in expected or tuple(tensor.shape) != expected[target].shape:
            raise RuntimeError(f'Unexpected weight name/shape: {name}')
        mapped[target] = mx.array(tensor.float().numpy())
        mapping.append({'source': name, 'target': target, 'shape': list(tensor.shape),
                        'transform': 'identity; both Linear weights are [out, in]'})
    if set(mapped) != set(expected):
        raise RuntimeError('Incomplete weight mapping')
    mx.eval(mapped)
    mx.save_safetensors(str(WEIGHTS), mapped)
    write_json(HERE / 'conversion.json', {
        'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'checkpoint_sha256': report['checkpoint_sha256'],
        'config_sha256': report['config_sha256'],
        'weights_sha256': sha256(WEIGHTS), 'precision': 'float32', 'mapping': mapping})


class Backend:
    def __init__(self, name):
        started = time.perf_counter()
        if name not in ('pytorch', 'mlx'):
            raise ValueError('Unknown backend')
        self.name = name
        report = source_metadata()
        from scripts.model import ScratchConfig
        self.config = ScratchConfig.from_file(FOLDER / 'config.json')
        # Verify provenance and bytes of the shared local tokenizer assets.
        bundle = ROOT / 'artifacts/short_epoch_1024'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        config_data = json.loads((FOLDER / 'config.json').read_text())
        if manifest['tokenizer_source_id'] != config_data['tokenizer_id'] or \
                manifest['tokenizer_source_revision'] != config_data['tokenizer_revision']:
            raise RuntimeError('Tokenizer source mismatch')
        for relative, expected in manifest['file_sha256'].items():
            if relative.startswith('tokenizer/') and sha256(bundle / relative) != expected:
                raise RuntimeError('Tokenizer asset hash mismatch')
        self.tokenizer = BertTokenizer.from_pretrained(
            ROOT / 'artifacts/short_epoch_1024/tokenizer', local_files_only=True)
        if self.tokenizer.vocab_size != self.config.vocab_size or \
                self.tokenizer.pad_token_id != self.config.pad_token_id:
            raise RuntimeError('Tokenizer/config mismatch')
        if name == 'pytorch':
            import torch
            from scripts.model import ScratchReranker
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS required')
            self.model = ScratchReranker(self.config)
            saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
            if saved['provenance_sha256'] != sha256(REFERENCE / 'config.json'):
                raise RuntimeError('Checkpoint config mismatch')
            self.model.load_state_dict(saved['model_state_dict'], strict=True)
            self.model.to('mps').eval()
        else:
            import mlx.core as mx
            from .mlx_model import MLXReranker
            if not mx.metal.is_available():
                raise RuntimeError('MLX Metal required')
            manifest = json.loads((HERE / 'conversion.json').read_text())
            if manifest['checkpoint_sha256'] != report['checkpoint_sha256'] or \
                    manifest['config_sha256'] != report['config_sha256'] or \
                    sha256(WEIGHTS) != manifest['weights_sha256']:
                raise RuntimeError('Converted weight provenance mismatch')
            self.model = MLXReranker(self.config)
            self.model.load_weights(str(WEIGHTS), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
        self.synchronize()
        self.loading_seconds = time.perf_counter() - started

    def encode(self, queries, documents):
        return dict(self.tokenizer(queries, documents, padding=True,
            truncation='longest_first', max_length=self.config.max_length,
            return_attention_mask=True, return_token_type_ids=True, return_tensors='np'))

    def prepare(self, inputs):
        if self.name == 'pytorch':
            import torch
            return {k: torch.from_numpy(v).to('mps') for k, v in inputs.items()}
        import mlx.core as mx
        return {k: mx.array(v.astype(np.int32)) for k, v in inputs.items()}

    def forward(self, inputs):
        if self.name == 'pytorch':
            import torch
            with torch.inference_mode():
                return self.model(**inputs).cpu().numpy()
        import mlx.core as mx
        values = self.model(**inputs)
        mx.eval(values)  # Timers must include execution, not only graph construction.
        return np.array(values)

    def score(self, queries, documents):
        values = self.forward(self.prepare(self.encode(queries, documents)))
        if values.shape != (len(queries),) or not np.isfinite(values).all():
            raise RuntimeError('Invalid scores')
        return values.tolist()

    def rerank(self, query, documents, batch_size=8):
        if not documents or not 1 <= batch_size <= 64:
            raise ValueError('Provide documents and a pair batch size in [1, 64]')
        scores = []
        for i in range(0, len(documents), batch_size):
            chunk = documents[i:i + batch_size]
            scores.extend(self.score([query] * len(chunk), chunk))
        return sorted([{'index': i, 'score': value} for i, value in enumerate(scores)],
                      key=lambda item: (-item['score'], item['index']))

    def synchronize(self):
        if self.name == 'pytorch':
            import torch
            torch.mps.synchronize()
        else:
            import mlx.core as mx
            mx.synchronize()

    def memory(self):
        if self.name == 'pytorch':
            import torch
            return {'current_tensor_bytes': torch.mps.current_allocated_memory(),
                    'current_driver_bytes': torch.mps.driver_allocated_memory()}
        import mlx.core as mx
        return {'current_tensor_bytes': mx.get_active_memory(),
                'cache_bytes': mx.get_cache_memory(), 'peak_tensor_bytes': mx.get_peak_memory()}


if __name__ == '__main__':
    convert()
    print('Converted selected epoch-3 checkpoint with explicit shape/name checks.')
