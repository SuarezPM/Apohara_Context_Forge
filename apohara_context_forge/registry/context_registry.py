"""ContextRegistry v3.0 - Wired to LSH + FAISS + VRAMAwareCache.

Replaces the old Python-loop dedup and static TTLCache with:
- LSHTokenMatcher: SimHash on actual Qwen3 token IDs, PagedAttention block alignment
- FAISSContextIndex: O(log n) ANN search vs O(n) linear scan
- VRAMAwareCache: 5-mode LRU/LFU hybrid with VRAM-pressure-responsive eviction

Dependency injection - no hardcoded imports of stale modules.
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from apohara_context_forge.dedup.faiss_index import FAISSContextIndex, FAISSMatch
from apohara_context_forge.dedup.lsh_engine import LSHTokenMatcher, TokenBlockMatch
from apohara_context_forge.dedup.prefix_dedup import PrefixDedup
from apohara_context_forge.embeddings.embedding_engine import EmbeddingEngine
from apohara_context_forge.kv_offset.anchor_pool import AnchorPool
from apohara_context_forge.metrics.prometheus_metrics import (
    cache_hits,
    cache_misses,
    cache_evictions_total,
)
from apohara_context_forge.models import ContextEntry, ContextMatch
from apohara_context_forge.registry.vram_aware_cache import VRAMAwareCache
from apohara_context_forge.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# vLLM PagedAttention block size
VLLM_BLOCK_SIZE = 16


@dataclass
class SharedContextResult:
    """Result of get_shared_context() - contains reusable blocks with metadata."""
    agent_id: str
    shared_blocks: list[TokenBlockMatch]
    faiss_matches: list[FAISSMatch]
    total_tokens_saved: int
    reuse_confidence: float  # 0.0-1.0 weighted by hamming distance
    offset_hints: dict[str, list[float]] = field(default_factory=dict)  # agent_id -> offset vector


@dataclass
class RegisteredAgent:
    """Internal record of a registered agent."""
    agent_id: str
    system_prompt: str
    role_prompt: str
    token_count: int
    block_hashes: list[int]  # LSH block hashes for this agent


class ContextRegistry:
    """
    Production-grade context registry with LSH + FAISS + VRAM-aware cache.

    Usage:
        registry = ContextRegistry(
            lsh_matcher=LSHTokenMatcher(),
            vram_cache=VRAMAwareCache(max_token_budget=50_000_000),
            faiss_index=FAISSContextIndex(dim=384),
        )
        await registry.start()

        # Register agents with shared system prompt
        await registry.register_agent("agent1", system_prompt, "retriever role")
        await registry.register_agent("agent2", system_prompt, "summarizer role")

        # Query for reusable context across agents
        result = await registry.get_shared_context(["agent1", "agent2"])

        await registry.stop()

    Key design decisions:
    - Dependency injection for all core components (testable, swappable)
    - LSH operates on token IDs, not text - aligns to vLLM PagedAttention blocks
    - FAISS provides ANN candidates; LSH filters for actual token-level reuse
    - VRAMAwareCache manages eviction based on real GPU memory pressure
    """

    def __init__(
        self,
        lsh_matcher: Optional[LSHTokenMatcher] = None,
        vram_cache: Optional[VRAMAwareCache] = None,
        faiss_index: Optional[FAISSContextIndex] = None,
        token_counter: Optional[TokenCounter] = None,
        anchor_pool: Optional[AnchorPool] = None,
        vram_budget_tokens: int = 50_000_000,
        block_size: int = VLLM_BLOCK_SIZE,
        hamming_threshold: int = 8,
        faiss_nlist: int = 100,
        dedup: Any = None,
    ):
        # Dependency injection with lazy defaults
        self._lsh = lsh_matcher or LSHTokenMatcher(
            block_size=block_size,
            hamming_threshold=hamming_threshold,
        )
        self._vram_cache = vram_cache or VRAMAwareCache(max_token_budget=vram_budget_tokens)
        # FAISS index dim must match the EmbeddingEngine output dimension
        # (we instantiate EmbeddingEngine with dim=512 in register_agent).
        # A 384-dim default crashes faiss.IndexFlatIP.add() at runtime —
        # see the cascade of test_integration failures pre-fix.
        self._faiss = faiss_index or FAISSContextIndex(dim=512)
        self._token_counter = token_counter or TokenCounter.get()
        self._anchor_pool = anchor_pool or AnchorPool()
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._block_size = block_size

        # `dedup` is a hermetic-test escape hatch — when set, register() short-
        # circuits the LSH+FAISS+ANN heavy path and uses the provided engine
        # instead. The engine only needs `embed`, `similarity`,
        # `find_shared_prefix`, and `count_prefix_tokens` — see FakeDedupEngine
        # in tests/test_mcp_server.py for the contract.
        # Default to a real PrefixDedup so the CompressionCoordinator (which
        # reads `.dedup`) has a working token-count + shared-prefix helper in
        # production; tests can still inject a fake via this kwarg.
        self._dedup = dedup if dedup is not None else PrefixDedup()

        # Lightweight in-memory store for `register(agent_id, context)`. This
        # is independent from `register_agent(...)` (which exercises the full
        # KV-aware pipeline) — it backs the simple MCP /tools/register_context
        # endpoint and the test_full_flow scenario.
        self._simple_entries: dict[str, ContextEntry] = {}

        # Internal state
        self._agents: dict[str, RegisteredAgent] = {}
        self._system_prompt_hash: Optional[str] = None
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Start background VRAM monitor and cache."""
        if self._started:
            return
        await self._vram_cache.start()
        self._started = True
        logger.info("ContextRegistry started with LSH+FAISS+VRAM cache")

    async def stop(self) -> None:
        """Stop background monitoring and flush cache."""
        if not self._started:
            return
        await self._vram_cache.stop()
        self._started = False
        logger.info("ContextRegistry stopped")

    async def register_agent(
        self,
        agent_id: str,
        system_prompt: str,
        role_prompt: str,
    ) -> ContextEntry:
        """
        Register an agent with tokenization and LSH indexing.

        Args:
            agent_id: Unique agent identifier
            system_prompt: Shared system prompt (must be byte-identical across agents)
            role_prompt: Agent-specific role/instruction text

        Returns:
            ContextEntry with accurate token count
        """
        loop = asyncio.get_event_loop()

        # Tokenize full context
        full_context = f"{system_prompt}\n\n{role_prompt}"
        token_ids = await loop.run_in_executor(
            None, self._token_counter.encode, full_context
        )
        token_count = len(token_ids)

        # Index system prompt for LSH (critical for prefix caching)
        await self._lsh.index_prompt(
            f"{agent_id}:system",
            system_prompt
        )

        # Index full prompt for cross-agent dedup
        full_block_hashes = await self._lsh.index_prompt(
            agent_id,
            full_context
        )

        # Generate real embedding via EmbeddingEngine (replaces pseudo-embedding)
        if self._embedding_engine is None:
            self._embedding_engine = await EmbeddingEngine.get_instance(dim=512, use_onnx=True)
        embedding = await self._embedding_engine.encode(full_context)

        # Update AnchorPool — use embedding as kv_offset_approx until
        # LMCacheConnectorV1 bridge (TASK-007) provides real KV offset vectors
        await self._anchor_pool.update_pool(
            token_ids=token_ids,
            agent_id=agent_id,
            real_kv_offset=embedding.copy(),
            neighbor_prefix_offset=None,  # populated by TASK-007
        )

        # Store in VRAM-aware cache
        cache_key = f"context:{agent_id}"
        cache_value = {
            "system_prompt": system_prompt,
            "role_prompt": role_prompt,
            "full_context": full_context,
            "token_ids": token_ids,
        }
        stored = await self._vram_cache.set(
            cache_key,
            cache_value,
            token_count=token_count,
        )
        if not stored:
            logger.warning(f"VRAM cache blocked registration for {agent_id}")

        # Add to FAISS index for ANN search
        # Use real embedding from EmbeddingEngine (replaces pseudo-embedding)
        await self._faiss.add(agent_id, embedding.tolist())

        # Track registered agent
        async with self._lock:
            # Validate system prompt consistency (byte-identical for vLLM prefix caching)
            if self._system_prompt_hash is None:
                self._system_prompt_hash = self._sha256_prefix(system_prompt)
            else:
                incoming_hash = self._sha256_prefix(system_prompt)
                if incoming_hash != self._system_prompt_hash:
                    logger.warning(
                        f"Agent {agent_id} has DIFFERENT system prompt hash. "
                        f"vLLM prefix caching will NOT work. "
                        f"Expected {self._system_prompt_hash[:16]}, got {incoming_hash[:16]}"
                    )

            self._agents[agent_id] = RegisteredAgent(
                agent_id=agent_id,
                system_prompt=system_prompt,
                role_prompt=role_prompt,
                token_count=token_count,
                block_hashes=full_block_hashes,
            )

        logger.debug(f"Registered agent {agent_id}, tokens={token_count}, blocks={len(full_block_hashes)}")

        return ContextEntry(
            agent_id=agent_id,
            context=full_context,
            token_count=token_count,
            compressed_token_count=None,
            ttl_seconds=0,  # VRAM cache handles TTL
        )

    async def get_shared_context(
        self,
        agent_ids: list[str],
        target_agent_id: Optional[str] = None,
    ) -> list[SharedContextResult]:
        """
        Query for reusable context across multiple agents.

        Uses FAISS ANN to find candidate matches, then LSH to validate
        actual token-level reuse at PagedAttention block granularity.

        Args:
            agent_ids: Agents whose context to search
            target_agent_id: Optional target for offset hints

        Returns:
            List of SharedContextResult sorted by reuse confidence
        """
        if len(agent_ids) < 2:
            return []

        # Gather all registered agents
        agents_to_search = []
        async with self._lock:
            for aid in agent_ids:
                if aid in self._agents:
                    agents_to_search.append(self._agents[aid])

        if not agents_to_search:
            return []

        # Pre-initialize embedding engine to avoid race condition in gather
        if self._embedding_engine is None:
            self._embedding_engine = await EmbeddingEngine.get_instance(dim=512, use_onnx=True)

        async def _process_agent(agent: RegisteredAgent) -> Optional[SharedContextResult]:
            # Get full context for LSH matching
            cache_key = f"context:{agent.agent_id}"
            cache_val = await self._vram_cache.get(cache_key)
            if not cache_val:
                return None

            full_context = cache_val["full_context"]
            system_prompt = cache_val["system_prompt"]

            # Find reusable blocks via LSH
            matches = await self._lsh.find_reusable_blocks(
                full_context,
                exclude_agent=agent.agent_id,
            )

            # Filter matches by hamming threshold and compute confidence
            valid_matches = []
            total_hamming = 0
            for match in matches:
                if match.hamming_distance <= self._lsh._hamming_threshold:
                    valid_matches.append(match)
                    total_hamming += match.hamming_distance

            if not valid_matches:
                cache_misses.labels(agent_id=agent.agent_id).inc()
                return None

            avg_hamming = total_hamming / len(valid_matches)
            reuse_confidence = 1.0 - (avg_hamming / self._lsh._hash_bits)

            # Get FAISS ANN candidates for the system prompt
            # Use real embedding from EmbeddingEngine (replaces pseudo-embedding)
            system_embedding = await self._embedding_engine.encode(system_prompt)
            faiss_matches = await self._faiss.search(
                system_embedding.tolist(),
                k=5,
                threshold=0.7,
            )

            # Compute total tokens saved.
            # Bug fix: previously this was
            #   blocks_per_match * self._block_size * len(valid_matches)
            # which equals len(valid_matches)**2 * self._block_size — a
            # quadratic over-count that inflated any dashboard or log
            # consuming `total_tokens_saved`.  See AUDIT.md item 12.
            blocks_per_match = len(valid_matches)
            tokens_saved = blocks_per_match * self._block_size

            # AnchorPool shareability prediction
            is_shareable = await self._anchor_pool.predict_shareable(
                token_ids=cache_val["token_ids"],
                target_agent_id=target_agent_id or agent.agent_id,
            )

            offset_vector = None
            if is_shareable:
                offset_result = await self._anchor_pool.approximate_offset(
                    token_ids=cache_val["token_ids"],
                    target_agent_id=target_agent_id or agent.agent_id,
                )
                if offset_result is not None:
                    offset_vector = offset_result.placeholder_offset

            # Populate offset_hints — this field was ALWAYS empty in V3
            result = SharedContextResult(
                agent_id=agent.agent_id,
                shared_blocks=valid_matches,
                faiss_matches=faiss_matches,
                total_tokens_saved=tokens_saved,
                reuse_confidence=reuse_confidence,
            )
            if offset_vector is not None:
                result.offset_hints[agent.agent_id] = offset_vector.tolist()

            cache_hits.labels(
                agent_id=agent.agent_id,
                segment_type="system_prompt",
            ).inc()

            return result

        # Gather all agents concurrently
        tasks = [_process_agent(agent) for agent in agents_to_search]
        gathered_results = await asyncio.gather(*tasks)

        results: list[SharedContextResult] = [r for r in gathered_results if r is not None]

        # Sort by reuse confidence descending
        results.sort(key=lambda r: r.reuse_confidence, reverse=True)
        return results

    async def find_similar(
        self,
        context: str,
        threshold: Optional[float] = None,
    ) -> list[ContextMatch]:
        """Find prior registered agents whose context is similar to `context`.

        Embeds the raw context, runs FAISS ANN over registered agents, then
        builds a typed ContextMatch per candidate (similarity + longest shared
        prefix). Returns matches at/above `threshold` (default:
        settings.contextforge_dedup_threshold), sorted by similarity desc.

        This is the raw-context query the CompressionCoordinator needs;
        get_shared_context() serves the agent-id query used by the pipeline.
        """
        from apohara_context_forge.config import settings

        if threshold is None:
            threshold = settings.contextforge_dedup_threshold

        if self._embedding_engine is None:
            self._embedding_engine = await EmbeddingEngine.get_instance(
                dim=512, use_onnx=True
            )
        embedding = await self._embedding_engine.encode(context)
        faiss_matches = await self._faiss.search(
            embedding.tolist(), k=5, threshold=threshold
        )

        loop = asyncio.get_running_loop()

        async def _process_match(fm) -> Optional[ContextMatch]:
            candidate_context = await self.get_agent_context(fm.agent_id)
            if candidate_context is None:
                return None
            # Offload heavy string matching to the executor so large contexts
            # do not block the event loop, while fanning out across matches.
            shared_prefix = await loop.run_in_executor(
                None, self._dedup.find_shared_prefix, context, candidate_context
            )
            shared_prefix_tokens = await loop.run_in_executor(
                None, self._dedup.count_prefix_tokens, shared_prefix
            )
            return ContextMatch(
                agent_id=fm.agent_id,
                similarity=fm.similarity,
                shared_prefix=shared_prefix,
                shared_prefix_tokens=shared_prefix_tokens,
            )

        gathered = await asyncio.gather(*(_process_match(fm) for fm in faiss_matches))
        results: list[ContextMatch] = [r for r in gathered if r is not None]

        results.sort(key=lambda m: m.similarity, reverse=True)
        return results

    async def get_agent_context(self, agent_id: str) -> Optional[str]:
        """Get the full context for an agent."""
        cache_key = f"context:{agent_id}"
        cache_val = await self._vram_cache.get(cache_key)
        if cache_val:
            return cache_val["full_context"]
        return None

    async def register(self, agent_id: str, context: str) -> ContextEntry:
        """Lightweight register used by the MCP /tools/register_context endpoint.

        This is intentionally separate from `register_agent(...)`, which also
        indexes the system prompt for cross-agent KV reuse. The MCP endpoint
        deals with single opaque contexts, so we tokenize via TokenCounter,
        keep a `ContextEntry` in `_simple_entries`, and stop there.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        loop = asyncio.get_event_loop()
        try:
            token_count = await loop.run_in_executor(
                None, self._token_counter.count, context
            )
        except Exception:
            token_count = max(1, len(context.split()))

        now = _dt.now(_tz.utc)
        entry = ContextEntry(
            agent_id=agent_id,
            context=context,
            token_count=token_count,
            created_at=now,
            expires_at=now + _td(seconds=300),
        )
        async with self._lock:
            self._simple_entries[agent_id] = entry
        return entry

    async def clear(self) -> None:
        """Drop all simple-register state. Called by the MCP server lifespan
        on shutdown so a fresh process starts from a clean registry. We do
        NOT touch LSH/FAISS here — those have their own lifecycle hooks."""
        async with self._lock:
            self._simple_entries.clear()

    async def clear_agent(self, agent_id: str) -> bool:
        """Clear an agent's context from all stores."""
        async with self._lock:
            if agent_id not in self._agents:
                return False

        # Remove from LSH
        await self._lsh.clear_agent(agent_id)
        await self._lsh.clear_agent(f"{agent_id}:system")

        # Remove from FAISS
        await self._faiss.remove(agent_id)

        # Remove from VRAM cache
        cache_key = f"context:{agent_id}"
        await self._vram_cache.delete(cache_key)

        # Remove from agents dict
        async with self._lock:
            del self._agents[agent_id]

        cache_evictions_total.labels(reason="manual").inc()
        return True

    async def get_all_agents(self) -> list[str]:
        """Get list of all registered agent IDs."""
        async with self._lock:
            return list(self._agents.keys())

    async def get_vram_mode(self) -> str:
        """Get current VRAM eviction mode."""
        return self._vram_cache.mode.value

    async def get_vram_pressure(self) -> float:
        """Get current VRAM pressure (0.0-1.0)."""
        return self._vram_cache._vram.get_pressure()

    @staticmethod
    def _sha256_prefix(text: str) -> str:
        """SHA256 of text for prefix validation."""
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def lsh_matcher(self) -> LSHTokenMatcher:
        """Direct access to LSH matcher for advanced queries."""
        return self._lsh

    @property
    def faiss_index(self) -> FAISSContextIndex:
        """Direct access to FAISS index for advanced queries."""
        return self._faiss

    @property
    def vram_cache(self) -> VRAMAwareCache:
        """Direct access to VRAM cache for advanced queries."""
        return self._vram_cache

    @property
    def dedup(self) -> Any:
        """Token-count + shared-prefix helper used by the CompressionCoordinator.

        Defaults to PrefixDedup; tests may inject a fake via the constructor.
        """
        return self._dedup

    @property
    def registry_size(self) -> int:
        """Number of registered agents."""
        return len(self._agents)

    @property
    def is_started(self) -> bool:
        """Whether the registry is running."""
        return self._started