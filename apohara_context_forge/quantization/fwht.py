"""Fast Walsh-Hadamard Transform (FWHT).

Orthonormal FWHT applied along the last dimension via in-place butterfly
recursion in O(d log d). Self-inverse under the sqrt(d) normalization used
here, so ``ifwht`` is exposed for clarity but performs the same operation.

Math reference: Walsh 1923, Hadamard 1893. The Hadamard matrix H_d is
constructed recursively as
    H_1 = [[1]]
    H_{2n} = [[H_n,  H_n],
              [H_n, -H_n]]
and the FWHT computes ``y = H_d @ x / sqrt(d)``.

Non-power-of-two inputs are zero-padded along the last dim to the next
power of two; output keeps the padded shape (the caller is responsible
for slicing back if needed).

Backend selection (Sprint 2 / AUDIT #320a): the ``_select_fwht_impl``
dispatcher prefers the in-tree ``turboquant_turing.fwht_inplace``
Rust wheel when the wheel is importable. The numpy / torch paths
remain the source of truth (the Rust kernel mirrors the numpy
butterfly byte-for-byte) and the fallback is automatic when the
wheel is not built in the active venv.
"""
from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Callable

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _torch = None
    _HAS_TORCH = False

import numpy as _np

if TYPE_CHECKING:
    import torch


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht_butterfly_torch(x: "torch.Tensor") -> "torch.Tensor":
    """In-place butterfly on a torch tensor; last dim must be a power of two."""
    d = x.shape[-1]
    h = 1
    while h < d:
        # Reshape last dim into (..., d/(2h), 2, h) so a and b are contiguous slices.
        view = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    return x


def _fwht_butterfly_torch_inplace(x: "torch.Tensor") -> "torch.Tensor":
    """In-place butterfly without fp32 upcast; clones only one half per stage.

    Matches the V7.0.0-alpha.6 fwht_inplace pattern: native dtype, single
    clone of the smaller slice per stage (60% lower peak alloc on MI300X
    vs the fp32-upcast path). Caller must own x (i.e., already cloned).
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        view = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = view[..., 0, :]
        b = view[..., 1, :]
        t = b.clone()
        b.copy_(a)
        b.sub_(t)
        a.add_(t)
        h *= 2
    return x


def _fwht_butterfly_numpy(x: _np.ndarray) -> _np.ndarray:
    d = x.shape[-1]
    h = 1
    while h < d:
        view = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    return x


def _select_fwht_impl(allow_rust: bool = True) -> str:
    """Pick the FWHT backend.

    Returns one of ``"rust"``, ``"torch"``, ``"numpy"``. The Rust
    backend is selected only when ``allow_rust`` is True and the
    ``turboquant_turing`` wheel is importable in the current
    process. The torch backend is selected when the input is a
    torch tensor; numpy is the universal fallback. The returned
    name is the *path* the dispatcher would take for a hypothetical
    numpy input — the actual tensor / numpy branching happens
    inside :func:`fwht` based on ``isinstance(x, torch.Tensor)``.
    """
    if allow_rust and importlib.util.find_spec("turboquant_turing") is not None:
        return "rust"
    return "numpy"


def _fwht_butterfly_rust_inplace(buf: _np.ndarray) -> _np.ndarray:
    """Rust-backed in-place butterfly (Sprint 2 / AUDIT #320a).

    Imports the in-tree ``turboquant_turing`` wheel lazily, calls
    ``fwht_inplace(buf)`` (1-D, contiguous, f32), and returns the
    same buffer. The wheel is built by
    ``apohara_context_forge/serving/turboquant_turing/build.sh``
    which chains ``cargo test --release && maturin develop
    --release``. The Rust kernel mirrors the numpy butterfly in
    :func:`_fwht_butterfly_numpy` byte-for-byte; the parity is
    asserted in ``tests/test_quantization_fwht.py``.
    """
    from apohara_context_forge.serving.turboquant_turing import (
        fwht_inplace as _rust_fwht_inplace,
    )
    _rust_fwht_inplace(buf)
    return buf


def fwht(x, *, fp32_upcast: bool = False, allow_rust: bool = True):
    """Apply orthonormal FWHT along last dim.

    By default runs in fp16/native dtype (2x faster, 60% less peak alloc on
    MI300X, precision error < INT4 noise floor per V7.0.0-alpha.5 measurements).
    Pass fp32_upcast=True for the legacy precision-conservative path.

    Args:
        x: torch.Tensor or np.ndarray; last dim is transformed. If the last
           dim is not a power of two, x is zero-padded to the next power of two.
        fp32_upcast: If True, run the butterfly in fp32 and cast back at the
           end (legacy behaviour). If False (default), run in the input dtype
           via the strided in-place butterfly.
        allow_rust: If True (default) and the ``turboquant_turing`` wheel is
           importable, dispatch the numpy path through the Rust kernel
           (``_fwht_butterfly_rust_inplace``). Set False to force the pure
           numpy path even when the wheel is present (useful for cross-checking
           numerics during AUDIT audits).

    Returns:
        Tensor of the same backend as input. Shape equals input shape with the
        last dim replaced by next_pow2(last_dim). Dtype is preserved.
    """
    rust_impl = _select_fwht_impl(allow_rust=allow_rust) if allow_rust else "numpy"

    if _HAS_TORCH and isinstance(x, _torch.Tensor):
        d = x.shape[-1]
        d_pad = _next_pow2(d)
        if d_pad != d:
            pad = _torch.zeros(*x.shape[:-1], d_pad - d, dtype=x.dtype, device=x.device)
            x = _torch.cat([x, pad], dim=-1)
        else:
            x = x.clone()
        orig_dtype = x.dtype

        if fp32_upcast:
            work = x.to(_torch.float32) if orig_dtype != _torch.float32 else x
            _fwht_butterfly_torch(work)
            work = work / (d_pad ** 0.5)
            return work.to(orig_dtype) if orig_dtype != _torch.float32 else work

        _fwht_butterfly_torch_inplace(x)
        x.mul_(1.0 / (d_pad ** 0.5))
        return x

    arr = _np.asarray(x)
    d = arr.shape[-1]
    d_pad = _next_pow2(d)
    if d_pad != d:
        pad_shape = (*arr.shape[:-1], d_pad - d)
        arr = _np.concatenate([arr, _np.zeros(pad_shape, dtype=arr.dtype)], axis=-1)
    else:
        arr = arr.copy()
    orig_dtype = arr.dtype

    if fp32_upcast:
        work = arr.astype(_np.float32, copy=False) if orig_dtype != _np.float32 else arr
        _fwht_butterfly_numpy(work)
        work = work / _np.sqrt(d_pad)
        return work.astype(orig_dtype, copy=False) if orig_dtype != _np.float32 else work

    if rust_impl == "rust":
        try:
            # Flatten all leading dims into a 1-D view of the last
            # axis. The Rust kernel operates on a contiguous 1-D f32
            # buffer. We reshape back after the call.
            flat = arr.reshape(-1).astype(_np.float32, copy=False)
            _fwht_butterfly_rust_inplace(flat)
            arr = flat.reshape(arr.shape).astype(orig_dtype, copy=False)
            arr = arr / _np.sqrt(d_pad)
            return arr
        except (ImportError, RuntimeError, ValueError):
            # Fall back to numpy if the wheel is broken or the buffer
            # shape is incompatible (e.g. fp16 — the Rust kernel is
            # f32-only and the cast could expose a numerics drift we
            # do not want to ship silently).
            pass

    _fwht_butterfly_numpy(arr)
    arr = arr / _np.sqrt(d_pad)
    return arr.astype(orig_dtype, copy=False) if arr.dtype != orig_dtype else arr


def ifwht(x, *, fp32_upcast: bool = False, allow_rust: bool = True):
    """Inverse FWHT. Orthonormal FWHT is self-inverse, so this just calls fwht."""
    return fwht(x, fp32_upcast=fp32_upcast, allow_rust=allow_rust)
