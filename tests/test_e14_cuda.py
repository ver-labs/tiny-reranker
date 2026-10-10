import pytest
import torch
from experiments.E14.kernel import residual_layer_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA GPU required')
@pytest.mark.parametrize('hidden', [32, 64, 128, 256])
@pytest.mark.parametrize('prefix', [(1,), (5,), (2, 37), (128, 128)])
@pytest.mark.parametrize('case', ['random', 'constant', 'cancellation'])
def test_cuda_float64_oracle(hidden, prefix, case):
    torch.manual_seed(17)
    x = torch.randn((*prefix, hidden), device='cuda')
    branch = torch.randn_like(x)
    if case == 'constant': x.fill_(10000); branch.fill_(-9999)
    if case == 'cancellation':
        x.mul_(1e5); branch = -x + torch.randn_like(x)
    weight = torch.randn(hidden, device='cuda'); bias = torch.randn_like(weight)
    with torch.inference_mode():
        summed, actual = residual_layer_norm(x, branch, weight, bias, fallback=False)
    torch.testing.assert_close(summed, x + branch, atol=0, rtol=0)
    expected = torch.nn.functional.layer_norm((x + branch).double(), (hidden,), weight.double(), bias.double(), 1e-5)
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)


def test_cpu_fallback_and_strict_guard():
    x = torch.randn(2, 128); weight = torch.ones(128); bias = torch.zeros(128)
    actual = residual_layer_norm(x, x, weight, bias)
    torch.testing.assert_close(actual[1], torch.nn.functional.layer_norm(x + x, (128,), weight, bias))
    with pytest.raises(RuntimeError): residual_layer_norm(x, x, weight, bias, fallback=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA GPU required')
def test_gradient_and_stride_guard():
    x = torch.randn(2, 128, device='cuda', requires_grad=True)
    weight = torch.ones(128, device='cuda'); bias = torch.zeros_like(weight)
    with pytest.raises(RuntimeError): residual_layer_norm(x, x, weight, bias, fallback=False)
    with torch.inference_mode():
        strided = torch.randn(2, 256, device='cuda')[:, ::2]
        with pytest.raises(RuntimeError): residual_layer_norm(strided, strided, weight, bias, fallback=False)
