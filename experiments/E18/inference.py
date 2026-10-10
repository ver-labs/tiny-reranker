"""Same pinned tokenizer; GPU-only timing paths with explicit stock SDPA."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
import numpy as np
from .common import HERE,ASSETS,verify_assets


class Inference:
    def __init__(self,device,path='stock',length=512):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        verify_assets()
        self.device,self.path,self.length = device,path,length
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(ASSETS,local_files_only=True,trust_remote_code=False,use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError('Pinned fast tokenizer required')
        if device=='mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Apple GPU required; CPU fallback prohibited')
        if device=='cuda' and (not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name()):
            raise RuntimeError('NVIDIA L40S required; CPU fallback prohibited')
        if device not in ('mps','cuda','cpu'):
            raise ValueError(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        self.custom_calls = 0
        if path in ('mlx','mlx_compiled','metal','metal_compiled'):
            if device!='mps':
                raise ValueError('MLX paths require Apple GPU')
            import mlx.core as mx
            from mlx.utils import tree_flatten
            from .mlx_model import Model
            from safetensors.numpy import load_file
            self.model = Model(json.loads((ASSETS/'config.json').read_text()))
            weights = load_file(ASSETS/'model.safetensors')
            expected = dict(tree_flatten(self.model.parameters()))
            # Old HF serialization may include deterministic position-ID buffers.
            buffers = {k:v for k,v in weights.items() if k.endswith('.position_ids')}
            for k,v in buffers.items():
                assert np.array_equal(v,np.arange(512)[None]), k
            weights = {k:v for k,v in weights.items() if k not in buffers}
            assert set(weights)==set(expected), (set(weights)-set(expected),set(expected)-set(weights))
            assert all(weights[k].shape==expected[k].shape for k in weights)
            self.model.load_weights([(k,mx.array(v)) for k,v in weights.items()],strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
            if path in ('metal','metal_compiled'):
                for layer in self.model.bert.encoder.layer:
                    layer.attention.output.custom = layer.output.custom = True
            if path in ('mlx_compiled','metal_compiled'):
                self.call = mx.compile(self.model)
            else:
                self.call = self.model
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(ASSETS,local_files_only=True,
                trust_remote_code=False,attn_implementation='eager' if path=='eager_attention' else 'sdpa').to(device).eval()
            if path in ('cuda','triton','explicit'):
                if device!='cuda':
                    raise ValueError('CUDA custom paths require NVIDIA')
                from .torch_kernel import install
                install(self.model,path,self)
            if path=='compiled':
                self.model = torch.compile(self.model,backend='inductor',fullgraph=True,dynamic=True)
            self.call = self.model
        self.sync()

    def encode(self,pairs):
        return dict(self.tokenizer([q for q,d in pairs],[d for q,d in pairs],padding=True,
                    truncation='longest_first',max_length=self.length,return_tensors='np'))

    def prepare(self,encoded):
        if self.path in ('mlx','mlx_compiled','metal','metal_compiled'):
            import mlx.core as mx
            return {k:mx.array(v.astype(np.int32)) for k,v in encoded.items()}
        import torch
        return {k:torch.from_numpy(v).to(self.device) for k,v in encoded.items()}

    def forward(self,inputs):
        if self.path in ('mlx','mlx_compiled','metal','metal_compiled'):
            import mlx.core as mx
            values = self.call(**inputs)
            mx.eval(values)
            return values
        import torch
        with torch.inference_mode():
            return self.call(**inputs).logits[:,0]

    def cpu(self,values):
        result = np.array(values) if self.path in ('mlx','mlx_compiled','metal','metal_compiled') else values.cpu().numpy()
        assert np.isfinite(result).all()
        return result.astype(float).tolist()

    def score(self,pairs):
        return self.cpu(self.forward(self.prepare(self.encode(pairs))))

    def request(self,row,k,batch=16):
        docs = sorted(zip(row['document_ids'][:k],row['texts'][:k]))
        scores = []
        for offset in range(0,k,batch):
            scores += self.score([(row['query'],t) for d,t in docs[offset:offset+batch]])
        mapped = dict(zip((d for d,t in docs),scores))
        return [mapped[d] for d in row['document_ids'][:k]]

    def sync(self):
        if self.path in ('mlx','mlx_compiled','metal','metal_compiled'):
            import mlx.core as mx
            mx.synchronize()
        elif self.device!='cpu':
            import torch
            if self.device=='mps':torch.mps.synchronize()
            else:torch.cuda.synchronize()

    def memory(self):
        if self.path in ('mlx','mlx_compiled','metal','metal_compiled'):
            import mlx.core as mx
            return {'active_bytes':mx.get_active_memory(),'peak_bytes':mx.get_peak_memory(),
                    'scope':'MLX allocator counters, not process RSS or system peak'}
        import torch
        if self.device=='mps':
            return {'active_bytes':torch.mps.current_allocated_memory(),'driver_bytes':torch.mps.driver_allocated_memory(),
                    'scope':'PyTorch MPS allocator snapshot'}
        if self.device=='cuda':
            return {'active_bytes':torch.cuda.memory_allocated(),'peak_bytes':torch.cuda.max_memory_allocated(),
                    'scope':'PyTorch CUDA allocator counters'}
        return {'scope':'CPU correctness diagnostic only'}

    def dispatch(self,pairs):
        inputs = self.prepare(self.encode(pairs))
        for _ in range(3):self.forward(inputs)
        self.sync()
        if self.path in ('mlx','mlx_compiled','metal','metal_compiled'):
            from . import metal_kernel
            metal_kernel.CALLS=0
            values=self.call(**inputs);mx.eval(values);self.sync()
            graph=HERE/'traces'/f'{self.device}_{self.path}_dispatch.dot';graph.parent.mkdir(exist_ok=True)
            mx.export_to_dot(str(graph),values)
            text=graph.read_text()
            import re
            primitives=re.findall(r'label ="([^"]+)"',text)
            from collections import Counter
            custom_count=sum(v=='CustomKernel' for v in primitives)
            attention_count=sum('ScaledDotProductAttention' in v for v in primitives)
            math_attention=sum(v=='Softmax' for v in primitives)
            if attention_count+math_attention!=6:raise RuntimeError('Attention graph does not contain six attention sites')
            if self.path in ('metal','metal_compiled') and custom_count!=12:raise RuntimeError(f'Observed {custom_count} custom sites')
            if self.path in ('mlx_compiled','metal_compiled') and not any(v.startswith('Compiled') for v in primitives):raise RuntimeError('No compiled primitives observed')
            return {'framework':'MLX','attention':'mx.fast.scaled_dot_product_attention',
                    'compiled':self.path in ('mlx_compiled','metal_compiled'),'custom_sites':custom_count,
                    'observed_fused_attention_sites':attention_count,'observed_math_attention_softmax_sites':math_attention,'graph_file':str(graph.relative_to(HERE)),
                    'observed_primitives':dict(Counter(primitives)),'python_custom_calls':metal_kernel.CALLS}
        import torch
        self.custom_calls = 0
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            self.forward(inputs)
            self.sync()
        names = {e.key:e.count for e in profiler.key_averages()}
        return {'operators':names,'stock_attention_requested':'sdpa' if self.path!='eager_attention' else 'eager',
                'custom_calls':self.custom_calls,'compiled':self.path=='compiled'}
