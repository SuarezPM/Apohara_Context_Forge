"""Build the vLLM invocation that loads LMCache as the OFFICIAL KVConnector.

ROMY Fase 3: cross-worker KV sharing is config-driven. vLLM never
exposed an attention-hook registry (see ``LMCACHE.md`` / ``AUDIT.md``
item 18), so the only honest way to intercept KV is vLLM's
``--kv-transfer-config`` flag pointing at LMCache's *built-in* connector
``LMCacheConnectorV1Dynamic``. This module does NOT subclass any
connector and does NOT touch the ContextForge ``LMCacheConnectorV2`` —
it only emits the strings/dicts that the launcher and the A/B harness
hand to the ``vllm`` CLI.

Everything here is PURE: no vLLM, no lmcache, no torch import. The
functions return plain dicts/strings and are unit-testable with no
server (see ``tests/test_vllm_launch_config.py``).

Two invariants this module owns:

* ``chunk_size == block_size`` — LMCache chunk boundaries must line up
  with vLLM PagedAttention block boundaries or cross-worker reuse
  degrades. :func:`build_kv_transfer_config` raises ``ValueError`` if a
  caller tries to launch with them misaligned.
* ``PYTHONHASHSEED=0`` on EVERY worker — vLLM's APC keys prefix blocks
  by ``hash(cache_salt, token_ids, ...)``; Python's per-process hash
  randomisation would make those keys disagree across workers, so the
  shared-prefix salts from :mod:`prefix_salt_planner` would never
  collide cross-worker. Pinning the seed is mandatory for cross-worker
  reuse to work at all.
"""
from __future__ import annotations

import json
from pathlib import Path

# vLLM PagedAttention block size (tokens). Mirrors VLLM_BLOCK_SIZE in
# the dedup/registry modules; LMCache chunk_size MUST equal this.
DEFAULT_BLOCK_SIZE = 16

# LMCache's OFFICIAL built-in dynamic connector + the module path vLLM
# imports it from. These strings are the public LMCache/vLLM contract;
# they are asserted verbatim by the tests.
LMCACHE_KV_CONNECTOR = "LMCacheConnectorV1Dynamic"
LMCACHE_KV_CONNECTOR_MODULE_PATH = "lmcache.integration.vllm.lmcache_connector_v1"
# "kv_both" = this worker both stores (producer) and retrieves (consumer)
# KV chunks, the role every symmetric ContextForge worker plays.
LMCACHE_KV_ROLE = "kv_both"


def build_kv_transfer_config(
    block_size: int = DEFAULT_BLOCK_SIZE,
    chunk_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Return the dict vLLM expects for ``--kv-transfer-config``.

    The shape matches LMCache's documented invocation of
    ``LMCacheConnectorV1Dynamic`` exactly. ``block_size``/``chunk_size``
    are validated for alignment but are NOT part of this dict — vLLM
    receives ``block_size`` via ``--block-size`` and LMCache receives
    ``chunk_size`` via its own config file (``deploy/lmcache_config.yaml``);
    they are passed here only so this function is the single place that
    enforces the alignment invariant.

    Raises:
        ValueError: if ``chunk_size != block_size`` (misalignment would
            silently degrade cross-worker reuse).
    """
    if chunk_size != block_size:
        raise ValueError(
            f"LMCache chunk_size ({chunk_size}) must equal vLLM block_size "
            f"({block_size}); misaligned chunks straddle PagedAttention "
            "block boundaries and break cross-worker KV reuse."
        )
    return {
        "kv_connector": LMCACHE_KV_CONNECTOR,
        "kv_role": LMCACHE_KV_ROLE,
        "kv_connector_module_path": LMCACHE_KV_CONNECTOR_MODULE_PATH,
    }


def build_kv_transfer_config_json(
    block_size: int = DEFAULT_BLOCK_SIZE,
    chunk_size: int = DEFAULT_BLOCK_SIZE,
) -> str:
    """Same as :func:`build_kv_transfer_config` but JSON-serialised.

    This is the exact string handed to ``vllm serve --kv-transfer-config``.
    Keys are emitted in the documented order (not sorted) so the string
    matches the canonical LMCache invocation byte-for-byte.
    """
    return json.dumps(build_kv_transfer_config(block_size, chunk_size))


def worker_env(extra: dict | None = None) -> dict:
    """Environment that MUST be set on EVERY cross-worker vLLM process.

    ``PYTHONHASHSEED=0`` is non-negotiable: without it, prefix-block
    hashes differ per worker and the shared salts never collide
    cross-worker, so LMCache stores chunks no other worker can find.
    ``LMCACHE_USE_EXPERIMENTAL=True`` selects LMCache's v1 engine (the
    one ``LMCacheConnectorV1Dynamic`` drives).

    Args:
        extra: optional overrides/additions merged on top.

    Returns:
        A new dict (does not mutate ``os.environ``; the caller decides
        how to apply it to the subprocess).
    """
    env = {
        "PYTHONHASHSEED": "0",
        "LMCACHE_USE_EXPERIMENTAL": "True",
    }
    if extra:
        env.update(extra)
    return env


def build_vllm_serve_args(
    model: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    chunk_size: int = DEFAULT_BLOCK_SIZE,
) -> list[str]:
    """Build the ``vllm serve`` CLI args for an LMCache-backed worker.

    Pure: returns the arg list, launches nothing. The caller runs
    ``["vllm", *args]`` with :func:`worker_env` applied. Alignment is
    enforced via :func:`build_kv_transfer_config_json`.
    """
    return [
        "serve",
        model,
        "--block-size",
        str(block_size),
        "--kv-transfer-config",
        build_kv_transfer_config_json(block_size, chunk_size),
    ]


def build_lmcache_config_yaml(
    *,
    remote_url: str,
    chunk_size: int = DEFAULT_BLOCK_SIZE,
    remote_serde: str = "naive",
    local_cpu: bool = False,
    max_local_cpu_size: float = 2.0,
) -> str:
    """Return the LMCache config-file body that points a worker at a shared store.

    This is the YAML ``remote_url`` form PROVEN in ``local_cross_worker_smoke.py``'s
    cross-process reuse result — NOT the ``LMCACHE_REDIS_URL`` env (which the smoke
    run did not validate). ``chunk_size`` MUST equal vLLM's ``--block-size`` so the
    LMCache chunk boundaries line up with PagedAttention block boundaries (same
    invariant :func:`build_kv_transfer_config` enforces); ``local_cpu: false`` makes
    worker-2 start with an EMPTY local cache so a hit can only come from the remote.

    Pure: returns the string; it does not touch the filesystem. The caller decides
    where to write it (see :func:`write_lmcache_config`).
    """
    return (
        f"chunk_size: {chunk_size}\n"
        f"local_cpu: {str(local_cpu).lower()}\n"
        f"max_local_cpu_size: {max_local_cpu_size}\n"
        f'remote_url: "{remote_url}"\n'
        f'remote_serde: "{remote_serde}"\n'
    )


def write_lmcache_config(
    out_path: str,
    *,
    remote_url: str,
    chunk_size: int = DEFAULT_BLOCK_SIZE,
    remote_serde: str = "naive",
    local_cpu: bool = False,
    max_local_cpu_size: float = 2.0,
) -> str:
    """Write the LMCache config-file body (:func:`build_lmcache_config_yaml`) to
    ``out_path`` and return that path.

    Parameterized on the output path so callers (the cross-worker harness, the smoke
    script) pick the location — NO ``/tmp`` hardcode here. The caller sets
    ``LMCACHE_CONFIG_FILE`` to the returned path in the worker env.
    """
    body = build_lmcache_config_yaml(
        remote_url=remote_url,
        chunk_size=chunk_size,
        remote_serde=remote_serde,
        local_cpu=local_cpu,
        max_local_cpu_size=max_local_cpu_size,
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return str(p)
