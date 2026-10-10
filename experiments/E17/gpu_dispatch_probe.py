"""Supplemental CUDA dispatch evidence; run after timings, never concurrently."""
import json
import torch
from .common import HERE, rows, save, sha, verify_plan
from .inference import Inference


def main():
    verify_plan()
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name():
        raise RuntimeError('L40S required')
    import experiments.E14.model as module
    row = rows()[0]
    pairs = [(row['query'], text) for text in row['texts'][:32]]
    report = {'plan_sha256': sha(HERE/'plan.json'), 'scope': 'Post-timing dispatch probe; no latency claims',
              'gpu': torch.cuda.get_device_name(), 'paths': {}}
    for path in ('strict', 'stock', 'cuda', 'triton'):
        model = Inference('cuda', path)
        inputs = model.prepare(model.encode(pairs))
        for _ in range(10):
            model.forward(inputs)
        model.sync()
        calls = []
        name = 'cuda_layer_norm' if path == 'cuda' else 'residual_layer_norm'
        original = getattr(module, name)
        def counted(*args, **kwargs):
            calls.append({'fallback': kwargs.get('fallback'), 'num_warps': kwargs.get('num_warps')})
            return original(*args, **kwargs)
        setattr(module, name, counted)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                values = model.forward(inputs)
                model.sync()
            scores = model.cpu(values)
        finally:
            setattr(module, name, original)
        kernels = sorted({event.name for event in profiler.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        if path in ('cuda', 'triton'):
            assert len(calls) == 8 and all(c['fallback'] is False for c in calls)
        else:
            assert not calls
        report['paths'][path] = {'custom_operation_calls': calls, 'cuda_kernel_names': kernels,
                                'scores_finite': bool(torch.isfinite(torch.tensor(scores)).all()),
                                'fused_encoder_calls': sum(event.count for event in profiler.key_averages()
                                                          if event.key == 'aten::_transformer_encoder_layer_fwd')}
        del model
    save(HERE/'cuda_dispatch_probe.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
