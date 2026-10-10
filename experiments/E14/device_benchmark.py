"""CUDA-event graph replay: amortize host dispatch over 100 resident operations.

Measures the complete two-output operation inside a CUDA graph, not HTTP or
allocation/dispatch wall latency, and not an isolated profiler kernel timestamp.
"""
import json
import torch
from .benchmark import HERE, environment, load_fixture, save, summary, sha256
from experiments.E13.kernel import reference, residual_layer_norm as triton
from .kernel import residual_layer_norm as cuda


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer, rows, checkpoint, manifest = load_fixture()
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    result = {'environment': environment(), 'checkpoint_sha256': sha256(checkpoint),
              'scope': 'CUDA events around graph replay / 100 operations; resident FP32 inputs, 30 rotated repeats',
              'operations_per_graph': 100, 'rows': [],
              'code_sha256': sha256(__import__('pathlib').Path(__file__))}
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        for size in (1, 2, 8, 16, 64, 128):
            encoded = tokenizer([q for q, _ in flat[:size]], [d for _, d in flat[:size]],
                padding=True, truncation='longest_first', max_length=model.config.max_length,
                return_tensors='pt')
            inputs = {k: v.cuda() for k, v in encoded.items()}
            positions = torch.arange(inputs['input_ids'].shape[1], device='cuda')
            x = model.embedding_norm(model.token_embeddings(inputs['input_ids']) +
                model.position_embeddings(positions)[None] + model.segment_embeddings(inputs['token_type_ids']))
            layer = model.encoder_layers[0]; norm = layer.norm1(x)
            branch = layer.self_attn(norm, norm, norm, key_padding_mask=inputs['attention_mask'] == 0,
                                    need_weights=False)[0].contiguous()
            values = (x.contiguous(), branch, layer.norm2.weight, layer.norm2.bias)
            functions = {'reference': lambda: reference(*values), 'cuda': lambda: cuda(*values, fallback=False)}
            for w in (1, 2, 4):
                functions[f'triton_{w}'] = lambda w=w: triton(*values, fallback=False, num_warps=w)
            graphs = {}; retained = {}; expected = reference(*values)
            for mode, fn in functions.items():
                warm_stream = torch.cuda.Stream(); warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream):
                    for _ in range(10): fn()
                torch.cuda.current_stream().wait_stream(warm_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100): output = fn()
                graphs[mode] = graph; retained[mode] = output
                graph.replay(); torch.cuda.synchronize()
                for actual, target in zip(output, expected):
                    torch.testing.assert_close(actual, target, atol=2e-5, rtol=2e-5)
            samples = {mode: [] for mode in functions}; modes = list(functions)
            for repeat in range(30):
                shift = repeat % len(modes)
                for mode in modes[shift:] + modes[:shift]:
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[mode].replay(); end.record(); end.synchronize()
                    samples[mode].append(start.elapsed_time(end) / 100)
            result['rows'].append({'batch_size': size, 'shape': list(x.shape),
                                  'operation_ms': {mode: summary(samples[mode]) for mode in modes}})
            print(f'Device graph batch={size} complete', flush=True)
    save(HERE / 'device_benchmark.json', result)


if __name__ == '__main__': main()
