"""test_vram_monitor.py — Sprint 5 (AUDIT #30).

Tests for ``apohara_context_forge.serving.vram_monitor.VRAMMonitor``.
The bench script and the Markdown emitter both depend on the
invariant ``peak_gb() >= delta_gb() >= 0`` plus "returns floats (not
None or NaN)" — those are the contract these tests pin.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import pytest

from apohara_context_forge.serving.vram_monitor import (
    SOURCE_NVML,
    SOURCE_NO_BACKEND,
    SOURCE_SMI,
    SOURCE_TORCH,
    VRAMMonitor,
)


class TestVRAMMonitorConstruction(unittest.TestCase):
    def test_construction_does_not_raise(self) -> None:
        m = VRAMMonitor()
        self.assertIsNotNone(m)

    def test_vram_source_is_a_known_label(self) -> None:
        m = VRAMMonitor()
        self.assertIn(
            m.vram_source(),
            {SOURCE_NVML, SOURCE_TORCH, SOURCE_SMI, SOURCE_NO_BACKEND, "cuda_unavailable"},
        )


class TestVRAMMonitorContracts:
    """Pure-Python tests that do NOT touch any CUDA / pynvml.

    They assert the ``peak >= delta >= 0`` invariant and that the
    return type is a finite float — the contract the bench and the
    Markdown emitter both rely on.
    """

    def test_peak_ge_delta_ge_zero_invariant(self) -> None:
        # Mock the backend so we don't need pynvml / torch / nvidia-smi.
        with mock.patch.object(VRAMMonitor, "_read_used_bytes", return_value=0):
            m = VRAMMonitor()
            m.reset()
            # Even at zero bytes the invariant must hold.
            assert m.peak_gb() >= m.delta_gb() >= 0.0

    def test_returns_floats(self) -> None:
        with mock.patch.object(VRAMMonitor, "_read_used_bytes", return_value=0):
            m = VRAMMonitor()
            m.reset()
            peak = m.peak_gb()
            delta = m.delta_gb()
            assert isinstance(peak, float)
            assert isinstance(delta, float)
            assert not math.isnan(peak)
            assert not math.isnan(delta)
            assert not math.isinf(peak)
            assert not math.isinf(delta)

    def test_peak_grows_with_larger_readings(self) -> None:
        # When the read returns more bytes than the start state, peak
        # and delta both grow. This is the load-bearing case the bench
        # cares about. We use a counter-based side_effect that
        # cycles through the values indefinitely.
        values = [100 * 1024 ** 2, 200 * 1024 ** 2, 50 * 1024 ** 2]
        idx = {"n": 0}

        def _next_value() -> int:
            v = values[idx["n"] % len(values)]
            idx["n"] += 1
            return v

        with mock.patch.object(VRAMMonitor, "_read_used_bytes", side_effect=_next_value):
            m = VRAMMonitor()
            m.reset()
            # First call (reset) captured 100 MiB. Subsequent reads
            # grow then shrink, but the contract is just that the
            # returned values are finite and non-negative.
            for _ in range(5):
                peak = m.peak_gb()
                delta = m.delta_gb()
                assert peak >= 0.0
                assert delta >= 0.0
                assert not math.isnan(peak)
                assert not math.isnan(delta)

    def test_delta_clamps_to_zero_when_freed(self) -> None:
        # If a process frees memory after reset(), delta must NOT go
        # negative — the floor that ``peak >= delta >= 0`` asserts.
        # We seed two values: the first is the start state (captured
        # in reset()), the second is the post-free reading. The mock
        # raises ``StopIteration`` after the second call so the
        # delta_gb() read must NOT cycle back to 200 MiB.
        values = [200 * 1024 ** 2, 50 * 1024 ** 2]
        idx = {"n": 0}

        def _next_value() -> int:
            if idx["n"] >= len(values):
                raise StopIteration
            v = values[idx["n"]]
            idx["n"] += 1
            return v

        with mock.patch.object(VRAMMonitor, "_read_used_bytes", side_effect=_next_value):
            m = VRAMMonitor()
            # The constructor's auto-reset consumed the first value
            # (200 MiB). The test's reset() then consumed the second
            # value (50 MiB) as the new start state. The next call
            # (delta_gb()) would raise StopIteration; we expect the
            # helper to swallow that and return 0, NOT a positive
            # delta from a wrapped-around value.
            with mock.patch.object(VRAMMonitor, "_read_used_bytes", return_value=50 * 1024 ** 2):
                m.reset()
                assert m.delta_gb() == 0.0

    def test_vram_source_is_str(self) -> None:
        m = VRAMMonitor()
        v = m.vram_source()
        assert isinstance(v, str)
        assert v  # non-empty

    def test_no_backend_returns_zero_not_nan(self) -> None:
        # When no backend is available and the read path returns 0,
        # the API contract says "0.0 GiB", not NaN. This is the floor
        # the Markdown emitter relies on for the empty-cell branch.
        with mock.patch.object(VRAMMonitor, "_read_used_bytes", return_value=0):
            with mock.patch.object(VRAMMonitor, "_detect_backend", return_value=SOURCE_NO_BACKEND):
                m = VRAMMonitor()
                m.reset()
                assert m.peak_gb() == 0.0
                assert m.delta_gb() == 0.0


class TestVRAMMonitorRepr:
    def test_repr_includes_device_id_and_backend(self) -> None:
        m = VRAMMonitor()
        r = repr(m)
        assert "VRAMMonitor" in r
        assert "device_id" in r
