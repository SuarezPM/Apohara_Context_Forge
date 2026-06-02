"""Tests for Fast Walsh-Hadamard Transform — V7 codec work."""
import numpy as np
import pytest
import torch

from apohara_context_forge.quantization.fwht import fwht, ifwht


HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)


def test_fwht_shape_preserved():
    x = torch.randn(8, dtype=torch.float32)
    y = fwht(x)
    assert y.shape == x.shape


def test_fwht_power_of_two_no_pad():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32)
    y = ifwht(fwht(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_fwht_non_power_of_two_padding():
    x = torch.randn(6, dtype=torch.float32)
    y = fwht(x)
    assert y.shape[-1] == 8


def test_fwht_orthogonality():
    # FWHT applied row-wise to I_4 yields the normalized Hadamard matrix H_4 / sqrt(4).
    I4 = torch.eye(4, dtype=torch.float32)
    Y = fwht(I4).numpy()
    expected = HADAMARD_4 / np.sqrt(4.0)
    assert np.allclose(Y, expected, atol=1e-5)


def test_ifwht_round_trip_identity():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float32)
    y = ifwht(fwht(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_fwht_batched():
    torch.manual_seed(2)
    x = torch.randn(2, 8, dtype=torch.float32)
    y = fwht(x)
    # Each batch row transformed independently → equals fwht applied per-row.
    y0 = fwht(x[0])
    y1 = fwht(x[1])
    assert torch.allclose(y[0], y0, atol=1e-6)
    assert torch.allclose(y[1], y1, atol=1e-6)


def test_fwht_dtype_preservation():
    x32 = torch.randn(8, dtype=torch.float32)
    assert fwht(x32).dtype == torch.float32

    x16 = torch.randn(8, dtype=torch.float16)
    y16 = fwht(x16)
    assert y16.dtype == torch.float16
    # fp16 round-trip within fp16 epsilon (~1e-3 for d=8).
    rt = ifwht(fwht(x16))
    assert torch.allclose(rt.float(), x16.float(), atol=5e-3)


def test_fwht_zero_input():
    x = torch.zeros(8, dtype=torch.float32)
    y = fwht(x)
    assert torch.allclose(y, torch.zeros(8, dtype=torch.float32))


def test_fwht_fp16_native():
    """Default fwht() on fp16 input runs in fp16 — no fp32 intermediate.

    V7.0.0-alpha.5/.6 measurement on MI300X: fp16-only butterfly is 2x faster
    and 60% lower peak alloc than the fp32-upcast path. This test pins the
    fp16-native execution so a future regression can't silently re-introduce
    the upcast.
    """
    torch.manual_seed(3)
    x = torch.randn(64, dtype=torch.float16)

    # Patched .to() asserts no fp16->fp32 promotion during the butterfly.
    orig_to = torch.Tensor.to
    upcasts = []

    def tracking_to(self, *args, **kwargs):
        out = orig_to(self, *args, **kwargs)
        if self.dtype == torch.float16 and out.dtype == torch.float32:
            upcasts.append(True)
        return out

    torch.Tensor.to = tracking_to
    try:
        y = fwht(x)
    finally:
        torch.Tensor.to = orig_to

    assert y.dtype == torch.float16
    assert y.shape == x.shape
    assert upcasts == [], f"fp16-default path leaked {len(upcasts)} fp16->fp32 upcasts"


def test_fwht_fp32_upcast_opt_in():
    """fwht(x, fp32_upcast=True) preserves shape/dtype and matches legacy result."""
    torch.manual_seed(4)
    x = torch.randn(64, dtype=torch.float16)
    y = fwht(x, fp32_upcast=True)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    # The upcast path is the legacy V7.0.0-alpha.4 default; reproduce its
    # output by hand and compare.
    ref32 = x.to(torch.float32)
    d = ref32.shape[-1]
    h = 1
    while h < d:
        view = ref32.view(d // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    ref32 = (ref32 / (d ** 0.5)).to(torch.float16)
    assert torch.equal(y, ref32)
