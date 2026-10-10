"""Actual Metal correctness tests; require an accessible Apple GPU."""
import numpy as np
import pytest

try:
    from experiments.S1 import backend as _policy
    import mlx.core as mx
except ImportError:
    pytest.skip('MLX / Metal is unavailable', allow_module_level=True)
if not mx.metal.is_available():
    pytest.skip('Apple GPU is unavailable', allow_module_level=True)

from experiments.E11.kernel import reference, residual_layer_norm
from experiments.E11.mlx_model import FusedMLXReranker
from scripts.model import ScratchConfig


@pytest.mark.parametrize('hidden', [32, 64, 128, 256])
@pytest.mark.parametrize('shape_prefix', [(1,), (5,), (1, 3), (3, 37)])
def test_matches_mlx_and_numpy(hidden, shape_prefix):
    rng = np.random.default_rng(hidden + sum(shape_prefix))
    shape = (*shape_prefix, hidden)
    a, b = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
    w, bias = (rng.normal(size=hidden).astype(np.float32) for _ in range(2))
    inputs = list(map(mx.array, (a, b, w, bias)))
    got = residual_layer_norm(*inputs, fallback=False)
    expected = reference(*inputs)
    mx.eval(got, expected)
    np.testing.assert_array_equal(np.array(got[0]), a + b)
    np.testing.assert_allclose(np.array(got[1]), np.array(expected[1]), atol=2e-5, rtol=2e-5)
    summed = (a + b).astype(np.float64)
    oracle = (summed - summed.mean(axis=-1, keepdims=True)) / np.sqrt(summed.var(axis=-1, keepdims=True) + 1e-5)
    oracle = oracle * w + bias
    np.testing.assert_allclose(np.array(got[1]), oracle, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('kind', ['zero', 'constant', 'large_constant', 'extreme', 'cancellation'])
def test_numerical_edges(kind):
    rng = np.random.default_rng(42)
    shape = (1, 7, 128)
    if kind == 'zero':
        a = b = np.zeros(shape, np.float32)
    elif kind == 'constant':
        a = np.full(shape, 3., np.float32); b = np.full(shape, -1., np.float32)
    elif kind == 'large_constant':
        a = np.full(shape, 1e20, np.float32); b = np.zeros(shape, np.float32)
    elif kind == 'extreme':
        a = (rng.normal(size=shape) * 1e15).astype(np.float32); b = -a * .25
    else:
        a = (rng.normal(size=shape) * 1e5).astype(np.float32); b = -a
    w = rng.normal(size=128).astype(np.float32); bias = rng.normal(size=128).astype(np.float32)
    inputs = list(map(mx.array, (a, b, w, bias)))
    got = residual_layer_norm(*inputs, fallback=False)
    expected = reference(*inputs)
    mx.eval(got, expected)
    assert np.isfinite(np.array(got[1])).all()
    np.testing.assert_array_equal(np.array(got[0]), a + b)
    np.testing.assert_allclose(np.array(got[1]), np.array(expected[1]), atol=2e-5, rtol=2e-5)


def test_strided_inputs_are_packed_correctly():
    a = mx.arange(5 * 128 * 2, dtype=mx.float32).reshape(5, 256)[:, ::2]
    b = mx.ones((128, 5)).transpose(1, 0)
    w = mx.ones(256)[::2]; bias = mx.zeros(256)[::2]
    got = residual_layer_norm(a, b, w, bias, fallback=False)
    expected = reference(a, b, w, bias)
    mx.eval(got, expected)
    for actual, target in zip(got, expected):
        np.testing.assert_allclose(np.array(actual), np.array(target), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('dtype,hidden', [(mx.float16, 128), (mx.float32, 96)])
def test_unsupported_inputs_fall_back_or_raise(dtype, hidden):
    args = (mx.ones((3, hidden), dtype=dtype), mx.zeros((3, hidden), dtype=dtype),
            mx.ones(hidden, dtype=dtype), mx.zeros(hidden, dtype=dtype))
    got = residual_layer_norm(*args)
    expected = reference(*args)
    mx.eval(got, expected)
    for actual, target in zip(got, expected):
        np.testing.assert_array_equal(np.array(actual), np.array(target))
    with pytest.raises(ValueError):
        residual_layer_norm(*args, fallback=False)


@pytest.mark.parametrize('eps', [0., -1., float('nan'), float('inf'), 1e-45, 1e40])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        residual_layer_norm(mx.ones((1, 128)), mx.zeros((1, 128)), mx.ones(128), mx.zeros(128), eps)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        residual_layer_norm(mx.ones((1, 128)), mx.zeros((2, 128)), mx.ones(128), mx.zeros(128))
    with pytest.raises(ValueError):
        residual_layer_norm(mx.ones((1, 128)), mx.zeros((1, 128)), mx.ones(64), mx.zeros(128))


@pytest.mark.parametrize('length', [1, 3, 37, 128])
def test_complete_model_states_and_training_guard(length):
    config = ScratchConfig(vocab_size=256, pad_token_id=0, dropout=0.1, max_length=128, hidden_size=128,
                           num_attention_heads=4, feed_forward_size=512, num_layers=4)
    mx.random.seed(42)
    model = FusedMLXReranker(config)
    model.eval()
    ids = mx.random.randint(0, 256, (2, length))
    mask = mx.ones_like(ids)
    if length > 1:
        mask = mask.at[1, length // 2:].add(-1)
    segments = mx.zeros_like(ids)
    standard = model(ids, mask, segments, trace=True)
    fused = model(ids, mask, segments, trace=True, use_fused=True)
    mx.eval(standard, fused)
    assert standard.keys() == fused.keys()
    for name in standard:
        np.testing.assert_allclose(np.array(fused[name]), np.array(standard[name]), atol=3e-5, rtol=3e-5)
    model.train()
    with pytest.raises(ValueError, match='inference-only'):
        model(ids, mask, segments, use_fused=True)
