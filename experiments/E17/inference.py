"""Named FP32 execution paths; GPU required, no CPU benchmark fallback."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import time
import numpy as np
from .common import ROOT, CHECKPOINT


class Inference:
    def __init__(self, device, path):
        import torch
        self.device, self.path = device, path
        if path == 'mlx':
            if device != 'mps': raise ValueError('MLX requires Apple')
            from experiments.S1.backend import Backend
            self.backend = Backend('mlx')
            self.model = self.backend.model
            self.tokenizer = self.backend.tokenizer
            return
        if device == 'mps':
            if path not in ('strict','stock','compiled'): raise ValueError('CUDA/Triton require NVIDIA')
            from experiments.S1.backend import Backend
            self.backend = Backend('pytorch')
            self.model, self.tokenizer = self.backend.model, self.backend.tokenizer
        elif device == 'cuda':
            if not torch.cuda.is_available() or torch.version.hip is not None:
                raise RuntimeError('NVIDIA GPU required')
            from scripts.model import ScratchConfig
            from experiments.E14.model import CudaReranker
            from transformers import BertTokenizer
            self.model = CudaReranker(ScratchConfig.from_file(ROOT/'experiments/E1/four_layer/config.json'))
            saved = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
            self.model.load_state_dict(saved['model_state_dict'],strict=True)
            self.model.to('cuda').eval()
            self.tokenizer = BertTokenizer.from_pretrained(ROOT/'artifacts/short_epoch_1024/tokenizer',local_files_only=True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
        else:
            raise ValueError('Only GPU execution supported')
        self.model.eval()
        torch.backends.mha.set_fastpath_enabled(path in ('stock','compiled'))
        if path == 'compiled':
            self.model = torch.compile(self.model, backend='inductor',fullgraph=True,dynamic=False)
        self.sync()

    def encode(self, pairs):
        return dict(self.tokenizer([q for q,d in pairs],[d for q,d in pairs],padding=True,
                     truncation='longest_first',max_length=128,return_attention_mask=True,
                     return_token_type_ids=True,return_tensors='np'))

    def prepare(self, encoded):
        if self.path == 'mlx': return self.backend.prepare(encoded)
        import torch
        return {k:torch.from_numpy(v).to(self.device) for k,v in encoded.items()}

    def forward(self, inputs):
        if self.path == 'mlx':
            import mlx.core as mx
            values = self.model(**inputs)
            mx.eval(values)
            return values
        import torch
        torch.backends.mha.set_fastpath_enabled(self.path in ('stock','compiled'))
        with torch.inference_mode():
            if self.device == 'cuda' and self.path in ('cuda','triton'):
                return self.model(**inputs,kernel_mode=self.path,num_warps=4)
            return self.model(**inputs)

    def cpu(self, values):
        if self.path == 'mlx': return np.array(values).astype(float).tolist()
        return values.cpu().tolist()

    def score(self, pairs):
        values = self.cpu(self.forward(self.prepare(self.encode(pairs))))
        if len(values) != len(pairs) or not np.isfinite(values).all(): raise RuntimeError('Invalid model scores')
        return values

    def request(self, row, k, batch):
        # Exactly the existing shared worker's document-ID scoring order.
        docs = sorted(zip(row['document_ids'][:k],row['texts'][:k]))
        scores = []
        for i in range(0,len(docs),batch):
            scores += self.score([(row['query'],t) for d,t in docs[i:i+batch]])
        mapped = dict(zip([d for d,t in docs],scores))
        return [mapped[d] for d in row['document_ids'][:k]]

    def sync(self):
        if self.path == 'mlx': self.backend.synchronize()
        else:
            import torch
            if self.device == 'mps': torch.mps.synchronize()
            else: torch.cuda.synchronize()

    def dispatch(self, pairs):
        if self.path == 'mlx': return {'implementation':'MLX explicit transformer; eval materializes GPU arrays'}
        import torch
        inputs = self.prepare(self.encode(pairs))
        for _ in range(10): self.forward(inputs)
        self.sync()
        with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            self.forward(inputs); self.sync()
        ops = {e.key:e.count for e in p.key_averages() if any(s in e.key for s in ('attention','transformer','layer_norm'))}
        fused = ops.get('aten::_transformer_encoder_layer_fwd',0)
        return {'mha_fastpath_flag':torch.backends.mha.get_fastpath_enabled(), 'operators':ops,
                'fused_encoder_executed':bool(fused), 'fused_encoder_calls':fused,
                'interpretation': 'Fused stock encoder observed' if fused else 'Fused encoder not observed; flag alone provides no optimization evidence',
                'custom_kernel':self.path if self.path in ('cuda','triton') else None,
                'custom_fallback':False if self.path in ('cuda','triton') else None}
