"""CPU contract checks plus explicitly skipped NVIDIA correctness tests on a Mac."""
import importlib.util

import pytest
import torch

from experiments.E13.kernel import reference, residual_layer_norm
from experiments.E13.model import TritonReranker
from scripts.model import ScratchConfig


def gpu_available():
    return (torch.cuda.is_available() and torch.version.hip is None and
            torch.cuda.get_device_capability() >= (8, 0) and importlib.util.find_spec('triton') is not None)


nvidia = pytest.mark.skipif(not gpu_available(), reason='Requires NVIDIA CUDA compute capability >=8.0 and Triton; not executed on this Mac')


def args(device='cpu', hidden=128, shape=(3, 128)):
    return (torch.randn(shape, device=device), torch.randn(shape, device=device),
            torch.randn(hidden, device=device), torch.randn(hidden, device=device))


def test_cpu_fallback_and_strict_rejection():
    values = args()
    actual = residual_layer_norm(*values)
    expected = reference(*values)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='NVIDIA'):
        residual_layer_norm(*values, fallback=False)


@pytest.mark.parametrize('epsilon', [0., -1., float('nan'), float('inf'), 1e-45, 1e40])
def test_bad_epsilon(epsilon):
    with pytest.raises(ValueError): residual_layer_norm(*args(), eps=epsilon)


def test_bad_shapes_and_launch():
    a, b, w, bias = args()
    with pytest.raises(ValueError): residual_layer_norm(a, b[:1], w, bias)
    with pytest.raises(ValueError): residual_layer_norm(a, b, w[:64], bias)
    with pytest.raises(ValueError): residual_layer_norm(a, b, w, bias, num_warps=3)


def model():
    torch.manual_seed(42)
    return TritonReranker(ScratchConfig(vocab_size=256, pad_token_id=0, dropout=.1,
        max_length=128, num_layers=4, hidden_size=128, num_attention_heads=4, feed_forward_size=512)).eval()


@pytest.mark.parametrize('length', [1, 3, 37, 128])
def test_explicit_reference_matches_original_cpu(length):
    m = model()
    ids = torch.randint(0, 256, (2, length)); mask = torch.ones_like(ids)
    if length > 1: mask[1, length // 2:] = 0
    with torch.inference_mode():
        original = m(ids, mask, torch.zeros_like(ids))
        previous = torch.backends.mha.get_fastpath_enabled()
        try:
            for fastpath in (False, True):
                torch.backends.mha.set_fastpath_enabled(fastpath)
                explicit = m(ids, mask, torch.zeros_like(ids), kernel_mode='reference')
                torch.testing.assert_close(explicit, original, atol=2e-5, rtol=2e-5)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)


def test_training_and_mode_guards():
    m = model(); ids = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(RuntimeError, match='inference_mode'):
        m(ids, ids, torch.zeros_like(ids), kernel_mode='triton')
    m.train()
    with pytest.raises(RuntimeError, match='inference only'):
        m(ids, ids, torch.zeros_like(ids), kernel_mode='reference')
    with pytest.raises(ValueError): m(ids, ids, ids, kernel_mode='bad')


def test_benchmark_missing_gpu_records_pending(monkeypatch, tmp_path):
    import sys
    from experiments.E13 import benchmark
    monkeypatch.setattr(benchmark, 'HERE', tmp_path)
    monkeypatch.setattr(benchmark, 'environment', lambda: {'cuda_available': False, 'triton_installed': False})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--require-gpu'])
    with pytest.raises(SystemExit) as error: benchmark.main()
    assert error.value.code == 2
    import json
    report = json.loads((tmp_path / 'hardware_status.json').read_text())
    assert report['status'] == 'pending_nvidia_execution'
    assert not report['gpu_correctness_verified'] and not report['performance_measured']
    assert not (tmp_path / 'benchmark.json').exists()


@nvidia
@pytest.mark.parametrize('hidden', [1, 32, 96, 128, 129, 256, 1024])
@pytest.mark.parametrize('prefix', [(1,), (5,), (2, 37)])
@pytest.mark.parametrize('warps', [1, 2, 4])
def test_cuda_kernel_against_reference_and_float64(hidden, prefix, warps):
    torch.manual_seed(42)
    values = args('cuda', hidden, (*prefix, hidden))
    with torch.inference_mode():
        actual = residual_layer_norm(*values, fallback=False, num_warps=warps)
        expected = reference(*values)
        oracle_sum = actual[0].double()
        oracle = (oracle_sum - oracle_sum.mean(-1, keepdim=True)) / torch.sqrt(oracle_sum.var(-1, unbiased=False, keepdim=True) + 1e-5)
        oracle = oracle * values[2].double() + values[3].double()
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[1].double(), oracle, atol=2e-5, rtol=2e-5)


@nvidia
@pytest.mark.parametrize('kind', ['zero', 'constant', 'large', 'cancellation'])
def test_cuda_edge_values(kind):
    a, b, w, bias = args('cuda')
    if kind == 'zero': a.zero_(); b.zero_()
    if kind == 'constant': a.fill_(3); b.fill_(-1)
    if kind == 'large': a.mul_(1e15); b.mul_(1e15)
    if kind == 'cancellation': b.copy_(-a)
    with torch.inference_mode():
        actual = residual_layer_norm(a, b, w, bias, fallback=False)
        expected = reference(a, b, w, bias)
        assert torch.isfinite(actual[1]).all()
        for x, y in zip(actual, expected): torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)


@nvidia
def test_cuda_strides_dtype_fallback_and_autograd_guard():
    values = args('cuda', 128, (3, 256))
    a, b = values[0][:, ::2], values[1][:, ::2]
    for vals in [(a, b, values[2], values[3]), tuple(v.half() for v in args('cuda'))]:
        actual = residual_layer_norm(*vals); expected = reference(*vals)
        for x, y in zip(actual, expected): torch.testing.assert_close(x, y, atol=0, rtol=0)
        with pytest.raises(RuntimeError): residual_layer_norm(*vals, fallback=False)
    values = args('cuda'); values[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match='inference-only'): residual_layer_norm(*values, fallback=False)


@nvidia
@pytest.mark.parametrize('length', [1, 3, 37, 128])
def test_cuda_complete_model(length):
    m = model().cuda(); ids = torch.randint(0, 256, (2, length), device='cuda')
    mask = torch.ones_like(ids)
    if length > 1: mask[1, length // 2:] = 0
    with torch.inference_mode():
        original = m(ids, mask, torch.zeros_like(ids))
        standard = m(ids, mask, torch.zeros_like(ids), kernel_mode='reference')
        fused = m(ids, mask, torch.zeros_like(ids), kernel_mode='triton')
    torch.testing.assert_close(standard, original, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(fused, standard, atol=2e-4, rtol=1e-4)
