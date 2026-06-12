"""Pipeline orchestrator v3.0 - wired to ContextForge registry."""
import asyncio
import logging
import time
from typing import Any, Optional

from agents.demo_agents import create_agents

from apohara_context_forge.dedup.faiss_index import FAISSContextIndex
from apohara_context_forge.dedup.lsh_engine import LSHTokenMatcher
from apohara_context_forge.metrics.vram_monitor import VRAMMonitor
from apohara_context_forge.normalization.prefix_normalizer import PrefixNormalizer
from apohara_context_forge.pipeline_config import PipelineConfig
from apohara_context_forge.registry.context_registry import ContextRegistry
from apohara_context_forge.registry.vram_aware_cache import VRAMAwareCache
from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates 5-agent pipeline with ContextForge v3.0 registry.

    Uses LSHTokenMatcher + FAISSContextIndex + VRAMAwareCache for:
    - Token-level SimHash deduplication (LSH)
    - O(log n) ANN semantic search (FAISS)
    - VRAM-pressure-responsive eviction (VRAMAwareCache)

    Usage:
        config = PipelineConfig(model_id="Qwen/Qwen3-235B-A22B")
        pipeline = Pipeline(config=config)
        await pipeline.start()
        result = await pipeline.run("What is machine learning?")
        await pipeline.stop()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        enable_contextforge: bool = True,
    ):
        self._config = config or PipelineConfig()
        self._config.validate()
        self.enable_contextforge = enable_contextforge

        # Create ContextForge registry with dependency injection
        self._registry: Optional[ContextRegistry] = None
        self._vram_monitor: Optional[VRAMMonitor] = None

        # Create demo agents
        self.agents = create_agents()

        # Prefix-caching wiring (ROMY Fase 1): a single normalizer assembles a
        # byte-identical system prefix across all agents, and the salt planner
        # decides each agent's vLLM cache_salt. The anchor for the shared prefix
        # is the normalizer's canonical system-prompt hash (identical for every
        # agent → same salt → shared KV blocks), except for judge agents that
        # trip INV-15, which the planner isolates with a unique salt.
        self._prefix_normalizer = PrefixNormalizer(
            canonical_system_prompt=self._get_system_prompt()
        )
        self._salt_planner = PrefixSaltPlanner()

        # Metrics collection
        self.metrics = {
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "agent_ttft_ms": [],
            "strategies_used": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "lsh_matches": 0,
        }

    async def start(self) -> None:
        """Start ContextForge registry and VRAM monitor."""
        if not self.enable_contextforge:
            return

        # Initialize VRAM monitor
        self._vram_monitor = VRAMMonitor()
        await self._vram_monitor.start()

        # Initialize registry with wired components
        self._registry = ContextRegistry(
            lsh_matcher=LSHTokenMatcher(
                block_size=self._config.block_size,
                hamming_threshold=self._config.hamming_threshold,
            ),
            vram_cache=VRAMAwareCache(
                max_token_budget=self._config.vram_budget_tokens,
            ),
            faiss_index=FAISSContextIndex(dim=self._config.faiss_dim),
            vram_budget_tokens=self._config.vram_budget_tokens,
            block_size=self._config.block_size,
            hamming_threshold=self._config.hamming_threshold,
            faiss_nlist=self._config.faiss_nlist,
        )
        await self._registry.start()

        logger.info(f"Pipeline started with ContextForge v3.0 (model={self._config.model_id})")

    async def stop(self) -> None:
        """Stop ContextForge registry and VRAM monitor."""
        if self._registry:
            await self._registry.stop()
            self._registry = None
        if self._vram_monitor:
            await self._vram_monitor.stop()
            self._vram_monitor = None
        logger.info("Pipeline stopped")

    async def run(self, query: str) -> dict[str, Any]:
        """Run the full pipeline for a query."""
        logger.info(f"Starting pipeline for query: {query[:50]}...")

        input_data = {"query": query}
        pipeline_output = {}
        start_time = time.time()

        for i, agent in enumerate(self.agents):
            agent_start = time.time()

            # Assemble the byte-identical prefix and pick this agent's salt.
            # This runs regardless of ContextForge: the prefix + salt are the
            # serving-side artefacts vLLM needs for Automatic Prefix Caching.
            normalized_prompt = self._prefix_normalizer.normalize(
                agent_id=agent.agent_id,
                user_prompt=query,
                agent_role_prompt=self._build_role_prompt(agent),
            )
            # Anchor = canonical system-prompt hash (identical across agents →
            # shared salt). cla_group is a SINGLE shared cohort id ("pipeline")
            # so every non-judge agent lands on the SAME shared salt and they
            # actually share prefix KV blocks. The JCR gate keys "judge-ness"
            # off agent_id (JUDGE_ROLES = {"critic", "judge"}), NOT the
            # descriptive role string, so we pass agent_id there. request_id is
            # unique per (agent, step) so INV-15 isolation is per-request.
            salt_plan = self._salt_planner.plan(
                agent_role=agent.agent_id,
                anchor_hash=self._prefix_normalizer.get_canonical_hash(),
                cla_group="pipeline",
                request_id=f"{agent.agent_id}:{i}",
                candidate_count=len(self.agents),
            )

            # Build context for this agent
            if self.enable_contextforge and self._registry:
                shared_context = self._build_shared_context(input_data, agent)

                # Register with ContextForge
                try:
                    # Get shared system prompt from first agent or use default
                    system_prompt = self._get_system_prompt()
                    role_prompt = self._build_role_prompt(agent)

                    await self._registry.register_agent(
                        agent.agent_id,
                        system_prompt,
                        role_prompt,
                    )

                    # Query for shared context across agents
                    all_agents = await self._registry.get_all_agents()
                    if len(all_agents) >= 2:
                        shared_results = await self._registry.get_shared_context(
                            all_agents,
                            target_agent_id=agent.agent_id,
                        )
                        if shared_results:
                            self.metrics["lsh_matches"] += 1
                            self.metrics["cache_hits"] += 1
                        else:
                            self.metrics["cache_misses"] += 1
                except Exception as e:
                    logger.warning(f"ContextForge error for {agent.agent_id}: {e}")

            # Process agent
            result = await agent.process(input_data)
            agent_duration = (time.time() - agent_start) * 1000

            pipeline_output[f"{agent.agent_id}_output"] = result["result"]
            pipeline_output[f"{agent.agent_id}_metrics"] = {
                "ttft_ms": agent_duration,
                "strategy": result["strategy"],
                "tokens_before": result["tokens_before"],
                "tokens_after": result["tokens_after"],
                # APC serving artefacts (computed above; ready for vLLM).
                "prompt_hash": self._prefix_normalizer.compute_prompt_hash(
                    normalized_prompt
                ),
                "cache_salt": salt_plan.cache_salt,
                "salt_shared": salt_plan.shared,
            }

            self.metrics["total_tokens_before"] += result["tokens_before"]
            self.metrics["total_tokens_after"] += result["tokens_after"]
            self.metrics["agent_ttft_ms"].append(agent_duration)
            self.metrics["strategies_used"][result["strategy"]] = \
                self.metrics["strategies_used"].get(result["strategy"], 0) + 1

            input_data[f"{agent.agent_id}_output"] = result["result"]

        total_duration = (time.time() - start_time) * 1000

        return {
            "query": query,
            "final_output": pipeline_output.get("responder_output", ""),
            "pipeline_duration_ms": total_duration,
            "agent_metrics": pipeline_output,
            "summary": {
                "total_tokens_before": self.metrics["total_tokens_before"],
                "total_tokens_after": self.metrics["total_tokens_after"],
                "avg_ttft_ms": sum(self.metrics["agent_ttft_ms"]) / len(self.metrics["agent_ttft_ms"]),
                "strategies": self.metrics["strategies_used"],
                "token_savings_pct": (
                    (self.metrics["total_tokens_before"] - self.metrics["total_tokens_after"])
                    / self.metrics["total_tokens_before"] * 100
                    if self.metrics["total_tokens_before"] > 0 else 0
                ),
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "lsh_matches": self.metrics["lsh_matches"],
            },
            "contextforge": {
                "vram_pressure": self._vram_monitor.get_pressure() if self._vram_monitor else 0.0,
                "eviction_mode": self._registry.get_vram_mode() if self._registry else "unknown",
                "registry_size": self._registry.registry_size if self._registry else 0,
            } if self.enable_contextforge else None,
        }

    def _build_shared_context(self, input_data: dict, agent) -> str:
        """Build the shared context string for an agent."""
        prev_output = input_data.get(f"{agent.agent_id}_output", "")
        return f"Query: {input_data.get('query', '')}\nPrevious: {prev_output}\nRole: {agent.role}"

    def _get_system_prompt(self) -> str:
        """Get the canonical system prompt (shared across all agents)."""
        return (
            "You are a helpful AI assistant. "
            "Provide accurate, detailed, and thoughtful responses. "
            "Use chain-of-thought reasoning when appropriate."
        )

    def _build_role_prompt(self, agent) -> str:
        """Build agent-specific role prompt."""
        return f"You are a {agent.role}. {agent.agent_id}"

    @property
    def registry(self) -> Optional[ContextRegistry]:
        """Direct access to ContextRegistry (for advanced queries)."""
        return self._registry


async def run_pipeline_dry():
    """Dry run - prints agent plan without execution."""
    agents = create_agents()
    print("\n=== ContextForge v3.0 Pipeline - Dry Run ===")
    print(f"Total agents: {len(agents)}\n")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent.agent_id.upper()} ({agent.role})")
    print("\nPipeline flow:")
    print("  Query -> Retriever -> Reranker -> Summarizer -> Critic -> Responder")
    print("\nContextForge v3.0 wiring:")
    print("  - LSHTokenMatcher: SimHash on Qwen3 token IDs")
    print("  - FAISSContextIndex: O(log n) ANN search")
    print("  - VRAMAwareCache: 5-mode VRAM-pressure eviction")
    print("\nEach agent will:")
    print("  1. Register context with ContextForge (LSH + VRAM cache)")
    print("  2. Query shared context via FAISS ANN + LSH validation")
    print("  3. Return result with metrics\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContextForge v3.0 Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--query", default="What is machine learning?", help="Query to process")
    parser.add_argument(
        "--no-contextforge",
        action="store_true",
        help="Disable ContextForge (use raw pipeline)",
    )
    args = parser.parse_args()

    if args.dry_run:
        asyncio.run(run_pipeline_dry())
    else:
        config = PipelineConfig()
        pipeline = Pipeline(config=config, enable_contextforge=not args.no_contextforge)

        async def main():
            await pipeline.start()
            result = await pipeline.run(args.query)
            await pipeline.stop()
            return result

        result = asyncio.run(main())
        print(f"\n=== Pipeline Result ===")
        print(f"Token savings: {result['summary']['token_savings_pct']:.1f}%")
        print(f"Avg TTFT: {result['summary']['avg_ttft_ms']:.1f}ms")
        print(f"Strategies: {result['summary']['strategies']}")
        if result.get("contextforge"):
            print(f"VRAM pressure: {result['contextforge']['vram_pressure']:.2%}")
            print(f"Eviction mode: {result['contextforge']['eviction_mode']}")
            print(f"Registry size: {result['contextforge']['registry_size']}")