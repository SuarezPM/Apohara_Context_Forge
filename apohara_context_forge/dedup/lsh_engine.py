"""LSH Token-Level Matching Engine - IMPROVEMENT-001.

Token-level fuzzy matching using SimHash for KV cache block reuse.
Operates on actual token IDs from Qwen3 tokenizer, not word-level strings.
Aligns to vLLM PagedAttention block boundaries (default block_size=16).

Architecture:
    Incoming prompt (text)
        │
        ▼
   Qwen3 Tokenizer         ← Real token IDs, not word splits
        │
        ▼
  LSH Block Hashing        ← SimHash on token blocks
        │
        ▼
  Block Alignment          ← Align to PagedAttention blocks (16 tokens)
        │
        ▼
  Match Candidates         ← Find blocks with hamming distance < threshold
        │
        ▼
  Reuse Decision           → List of reusable block indices

Usage:
    matcher = LSHTokenMatcher()
    await matcher.index_prompt("agent1", "shared system prompt...")
    matches = await matcher.find_reusable_blocks("new incoming prompt...")
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from apohara_context_forge.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# vLLM PagedAttention default block size
VLLM_BLOCK_SIZE = 16


@dataclass
class TokenBlockMatch:
    """A matching block found in the LSH index."""

    block_index: int  # Which block position in the new prompt
    cached_block_hash: int  # 64-bit SimHash of the matching cached block
    hamming_distance: int  # Lower = more similar (0 = identical)
    reuse_confidence: float  # 0.0-1.0 derived from hamming distance
    cached_agent_id: str  # Which agent owns the cached block


class LSHTokenMatcher:
    """
    Token-level fuzzy matching using SimHash for KV cache block reuse.
    Operates on actual token IDs from Qwen3 tokenizer.

    Key insight: vLLM PagedAttention shares KV cache for identical token blocks.
    Two prompts with 95% SBERT similarity but different wording may share ZERO cache.
    LSH finds actual token-level matches at block boundaries.

    Usage:
        matcher = LSHTokenMatcher()
        await matcher.index_prompt("agent1", system_prompt)
        matches = await matcher.find_reusable_blocks(new_prompt)
    """

    def __init__(
        self,
        block_size: int = VLLM_BLOCK_SIZE,
        hash_bits: int = 64,
        hamming_threshold: int = 8,  # <8 bits different = high confidence
    ):
        self._block_size = block_size
        self._hash_bits = hash_bits
        self._hamming_threshold = hamming_threshold
        self._token_counter = TokenCounter.get()
        # hash → list of (tokens, agent_id). A list (not a single tuple) so
        # that multiple agents sharing the same prefix do not overwrite each
        # other — the last writer would otherwise erase the earlier owners
        # and `find_reusable_blocks` would miss legitimate cross-agent reuse.
        self._block_store: dict[int, dict[str, tuple[int, ...]]] = {}
        self._agent_blocks: dict[str, list[int]] = {}  # agent_id → list of block hashes
        self._lock = asyncio.Lock()

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        """Compute Hamming distance between two 64-bit integers."""
        return (a ^ b).bit_count()

    async def index_prompt(
        self,
        agent_id: str,
        text: str,
    ) -> list[int]:
        """
        Tokenize, blockify, and index a prompt for future reuse.
        Stores block hashes in LSH index.

        Args:
            agent_id: Owner of this prompt
            text: Full prompt text

        Returns:
            List of block hashes that were indexed
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(None, self._token_counter.encode, text)

        hashes = []
        blocks = []

        # Create blocks aligned to vLLM PagedAttention boundaries
        for i in range(0, len(token_ids), self._block_size):
            block = tuple(token_ids[i : i + self._block_size])

            # Skip partial blocks (no cache guarantee for < block_size)
            if len(block) < self._block_size:
                continue

            block_hash = self._simhash_block(block)
            owners = self._block_store.setdefault(block_hash, {})
            # Avoid duplicating the same owner if index_prompt is called
            # repeatedly for an agent (idempotent re-index).
            if agent_id not in owners:
                owners[agent_id] = block
            hashes.append(block_hash)
            blocks.append(block_hash)

        async with self._lock:
            self._agent_blocks[agent_id] = hashes

        logger.debug(f"Indexed {len(hashes)} blocks for agent {agent_id}")
        return hashes

    async def find_reusable_blocks(
        self,
        text: str,
        exclude_agent: Optional[str] = None,
    ) -> list[TokenBlockMatch]:
        """
        Find cached blocks that can be reused for this prompt.

        Args:
            text: New prompt text
            exclude_agent: Optionally exclude blocks from a specific agent

        Returns:
            List of TokenBlockMatch sorted by hamming distance (best first)
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(None, self._token_counter.encode, text)

        matches = []

        for i in range(0, len(token_ids), self._block_size):
            block = tuple(token_ids[i : i + self._block_size])

            if len(block) < self._block_size:
                continue

            new_hash = self._simhash_block(block)

            # Search for similar blocks. Each entry in the store may have
            # multiple owners (agents that all indexed the same block).
            # Exclusion matches both the bare agent_id ("agent1") and any
            # role-suffixed variant ("agent1:system") because the registry
            # indexes the system prompt under "<agent_id>:system" — without
            # this an agent finds matches against its own system blocks and
            # the cross-agent dedup path looks artificially busy.
            exclude_prefix = f"{exclude_agent}:" if exclude_agent else None
            for cached_hash, owners in self._block_store.items():
                hd = self._hamming(new_hash, cached_hash)
                if hd > self._hamming_threshold:
                    continue
                confidence = 1.0 - (hd / self._hash_bits)
                for agent_id, cached_tokens in owners.items():
                    if exclude_agent and (
                        agent_id == exclude_agent
                        or (
                            exclude_prefix is not None
                            and agent_id.startswith(exclude_prefix)
                        )
                    ):
                        continue
                    matches.append(
                        TokenBlockMatch(
                            block_index=i // self._block_size,
                            cached_block_hash=cached_hash,
                            hamming_distance=hd,
                            reuse_confidence=confidence,
                            cached_agent_id=agent_id,
                        )
                    )

        # Sort by hamming distance (best = lowest)
        matches.sort(key=lambda m: m.hamming_distance)
        return matches

    async def get_shared_prefix_hash(self, text: str) -> str:
        """
        Compute a stable hash of the shared prefix (first block).
        Used for routing hints to llm-d/vLLM.

        Args:
            text: Prompt text

        Returns:
            SHA256 hex string of first block's tokens
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(None, self._token_counter.encode, text)

        if len(token_ids) < self._block_size:
            first_block = token_ids
        else:
            first_block = token_ids[: self._block_size]

        # Create deterministic hash
        hash_input = str(tuple(first_block)).encode()
        return hashlib.sha256(hash_input).hexdigest()[:32]  # First 32 chars

    def _simhash_block(self, token_ids: tuple[int, ...]) -> int:
        """
        Compute 64-bit SimHash fingerprint for a token block.

        Uses stable pseudo-random projection per token ID.
        Deterministic: same block always produces same hash.

        Args:
            token_ids: Tuple of token IDs

        Returns:
            64-bit integer hash
        """
        v = np.zeros(self._hash_bits, dtype=np.float32)

        for tid in token_ids:
            # Deterministic pseudo-random projection
            # Using xorshift for speed (avoids numpy RNG object creation)
            h = int(tid)
            for _ in range(4):  # Mix well
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF

            # Project onto hash bits
            for bit in range(self._hash_bits):
                if (h >> (bit % 32)) & 1:
                    v[bit] += 1
                else:
                    v[bit] -= 1

        # Binarize
        bits = (v > 0).astype(np.uint8)

        # Pack into int64
        result = 0
        for i, b in enumerate(bits):
            result |= int(b) << i

        return result

    async def stats(self) -> dict:
        """Return index statistics."""
        async with self._lock:
            return {
                "total_blocks": len(self._block_store),
                "total_agents": len(self._agent_blocks),
                "block_size": self._block_size,
                "hash_bits": self._hash_bits,
                "hamming_threshold": self._hamming_threshold,
            }

    async def clear_agent(self, agent_id: str) -> int:
        """
        Remove all blocks indexed for an agent.

        Args:
            agent_id: Agent to clear

        Returns:
            Number of blocks removed
        """
        async with self._lock:
            hashes = self._agent_blocks.pop(agent_id, [])
            for h in hashes:
                owners = self._block_store.get(h)
                if not owners:
                    continue
                # Drop only this agent's entry; keep blocks shared with others.
                owners.pop(agent_id, None)
                if not owners:
                    del self._block_store[h]
            return len(hashes)
