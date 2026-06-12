"""VRAMMonitor — Sprint 5 (AUDIT #30).

Zero-overhead VRAM probe for the "WOW 8 GB" bench. Wraps the read paths
in order of reliability and exposes a tiny surface:

    monitor = VRAMMonitor()
    monitor.reset()            # record start state
    ...                        # do work that allocates VRAM
    monitor.peak_gb()          # max in GiB
    monitor.delta_gb()         # current - start in GiB

Backend preference:

    1. ``pynvml`` (device-wide used bytes; ground truth on NVIDIA).
    2. ``torch.cuda.memory_reserved`` (process-reserved; honestly
       labelled, NOT device-wide).
    3. ``nvidia-smi --query-gpu=memory.used --format=csv,noheader``
       (subprocess fallback, only if neither of the above is
       available).

All values are measured, never fabricated. If no reader works the
monitor returns ``0.0`` from both ``peak_gb()`` and ``delta_gb()`` and
``vram_source()`` is one of the honest ``unavailable`` labels — never
a default like 192 GB.

The AMD / ROCm path is intentionally NOT covered here: the spec
targets the local RTX 2060 SUPER (8 GB), and the existing
``apohara_context_forge/metrics/vram_monitor.py`` already does the
ROCm path with a different surface (``get_used_gb()``, pressure,
eviction mode). This module is the slim, bench-friendly variant.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Honest vram_source labels — never invent a default like 192 GB.
SOURCE_NVML = "cuda_pynvml"
SOURCE_TORCH = "cuda_torch_reserved"
SOURCE_SMI = "cuda_nvidia_smi"
SOURCE_NONE = "cuda_unavailable"
SOURCE_NO_BACKEND = "no_cuda_backend"


class VRAMMonitor:
    """Tiny VRAM probe. Backend-preference: pynvml -> torch -> nvidia-smi.

    Construction never raises: if no backend is available the monitor
    still instantiates and reports 0.0 GiB with an honest source label.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._backend = self._detect_backend()
        self._vram_source = (
            self._backend if self._backend in (SOURCE_NVML, SOURCE_TORCH)
            else (SOURCE_SMI if self._backend == SOURCE_SMI else SOURCE_NONE)
        )
        self._start_used_bytes: int = 0
        self._peak_used_bytes: int = 0
        self._last_used_bytes: int = 0
        self._lock = threading.Lock()
        # Best-effort start state. We don't want construction to raise;
        # the bench can always call reset() explicitly to be sure.
        try:
            self.reset()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("VRAMMonitor initial reset failed: %s", e)

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Record current usage as the start state and reset the peak."""
        used = self._read_used_bytes()
        with self._lock:
            self._start_used_bytes = used
            self._last_used_bytes = used
            self._peak_used_bytes = used

    def peak_gb(self) -> float:
        """Max in-process used VRAM in GiB since the last ``reset()``."""
        current = self._read_used_bytes()
        with self._lock:
            if current > self._peak_used_bytes:
                self._peak_used_bytes = current
            self._last_used_bytes = current
            return self._peak_used_bytes / (1024 ** 3)

    def delta_gb(self) -> float:
        """Current used VRAM minus the start state, in GiB (>= 0)."""
        current = self._read_used_bytes()
        with self._lock:
            self._last_used_bytes = current
            delta_bytes = current - self._start_used_bytes
            if delta_bytes < 0:
                # A process that freed memory after reset() shouldn't
                # report a negative peak; clamp to 0 so callers that
                # do ``delta <= peak`` hold.
                delta_bytes = 0
            return delta_bytes / (1024 ** 3)

    def vram_source(self) -> str:
        """Honest label of the backend that produced the last byte read."""
        with self._lock:
            return self._vram_source

    def current_gb(self) -> float:
        """Current used VRAM in GiB (a snapshot, not a peak)."""
        current = self._read_used_bytes()
        with self._lock:
            self._last_used_bytes = current
            return current / (1024 ** 3)

    # ------------------------------------------------------------- backends

    def _detect_backend(self) -> str:
        """Pick the best backend available on this host. Never raises."""
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            if count > self._device_id:
                return SOURCE_NVML
        except Exception as e:
            logger.debug("pynvml probe failed: %s", e)
        try:
            import torch  # type: ignore
            if torch.cuda.is_available() and torch.cuda.device_count() > self._device_id:
                return SOURCE_TORCH
        except Exception as e:
            logger.debug("torch cuda probe failed: %s", e)
        if shutil.which("nvidia-smi") is not None:
            return SOURCE_SMI
        return SOURCE_NO_BACKEND

    def _read_used_bytes(self) -> int:
        """Read used bytes from the selected backend. 0 on failure."""
        if self._backend == SOURCE_NVML:
            used = self._nvml_used_bytes()
            if used is not None:
                return used
            # Fall through to torch / smi. We don't fail closed; the
            # remaining paths are still ground truth on the same host.
        if self._backend in (SOURCE_NVML, SOURCE_TORCH):
            used = self._torch_used_bytes()
            if used is not None:
                return used
        if self._backend in (SOURCE_NVML, SOURCE_TORCH, SOURCE_SMI):
            used = self._smi_used_bytes()
            if used is not None:
                return used
        return 0

    def _nvml_used_bytes(self) -> Optional[int]:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._vram_source = SOURCE_NVML
                return int(mem.used)
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("pynvml read failed: %s", e)
            return None

    def _torch_used_bytes(self) -> Optional[int]:
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                return None
            if torch.cuda.device_count() <= self._device_id:
                return None
            # Process-reserved is the honest label: it's NOT
            # device-wide used bytes. The orchestrator surfaces this
            # via vram_source().
            self._vram_source = SOURCE_TORCH
            return int(torch.cuda.memory_reserved(self._device_id))
        except Exception as e:
            logger.debug("torch.cuda.memory_reserved failed: %s", e)
            return None

    def _smi_used_bytes(self) -> Optional[int]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._device_id}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
            line = proc.stdout.strip().splitlines()
            if not line:
                return None
            used_mib = float(line[0].strip())
            self._vram_source = SOURCE_SMI
            return int(round(used_mib * (1024 ** 2)))
        except Exception as e:
            logger.debug("nvidia-smi read failed: %s", e)
            return None

    def __repr__(self) -> str:
        return (
            f"VRAMMonitor(device_id={self._device_id}, "
            f"backend={self._backend!r}, source={self._vram_source!r})"
        )


__all__ = [
    "VRAMMonitor",
    "SOURCE_NVML",
    "SOURCE_TORCH",
    "SOURCE_SMI",
    "SOURCE_NONE",
    "SOURCE_NO_BACKEND",
]
