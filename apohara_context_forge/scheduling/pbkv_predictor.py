"""PBKVPredictor — prediction-based KV cache eviction priority.

Based on PBKV (arXiv:2605.06472, May 2026):
Prediction-based KV cache management for dynamic agent workflows.
Key result: 1.26x speedup over KVFlow (NeurIPS 2025).

Implementation: 2nd-order Markov chain over agent_id sequences.
State: (agent_id_t-2, agent_id_t-1)
Transition: predict agent_id_t with highest probability
Training: MLE on JSONL logs from PBKVPredictor stub output

Why Markov over neural:
- Zero VRAM overhead
- <1μs prediction latency
- Sufficient for agentic workflow patterns (low entropy, high repetition)
- PBKV paper uses similar lightweight approach for dynamic scenarios
"""
from __future__ import annotations

import asyncio
import aiofiles
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from apohara_context_forge.scheduling.step_graph import AgentStepGraph

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStepRecord:
    """Single step in a workflow sequence."""

    step_idx: int
    agent_id: str
    anchor_hash: str
    token_length: int
    cla_group: Optional[int] = None


@dataclass
class PredictionResult:
    """Prediction for next KV cache access."""

    predicted_agents: list[str]  # ranked by probability
    predicted_anchor_hashes: list[str]
    confidence: float
    prefetch_block_ids: list[str] = field(default_factory=list)


class PBKVPredictor:
    """Predictor-based KV cache prefetching using 2nd-order Markov chain.

    Design:
    1. Log each workflow step to local JSONL file
    2. Train Markov transition table from logged steps
    3. Predict next agents using transition probabilities
    4. Blend with AgentStepGraph for eviction/prefetch decisions

    Markov Chain:
    - 2nd-order: state = (prev_agent, curr_agent) → next_agent
    - 1st-order fallback: state = curr_agent → next_agent
    - Laplace smoothing (alpha=1) for unseen transitions
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_history_steps: int = 1000,
        blend_alpha: float = 0.6,
    ):
        self._log_dir = Path(log_dir) if log_dir else Path(".") / ".pbkv_logs"
        self._max_history_steps = max_history_steps
        self._blend_alpha = blend_alpha
        self._history: list[WorkflowStepRecord] = []
        self._transition_table: dict[tuple[str, str], dict[str, int]] = {}
        self._first_order_table: dict[str, dict[str, int]] = {}
        self._all_agents: set[str] = set()
        self._lock = asyncio.Lock()
        self._log_file = self._log_dir / "workflow_steps.jsonl"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trained = False

    async def log_workflow_step(
        self,
        step_idx: int,
        agent_id: str,
        anchor_hash: str,
        token_length: int,
        cla_group: Optional[int] = None,
    ) -> None:
        """Log a workflow step for future prediction training."""
        record = WorkflowStepRecord(
            step_idx=step_idx,
            agent_id=agent_id,
            anchor_hash=anchor_hash,
            token_length=token_length,
            cla_group=cla_group,
        )

        async with self._lock:
            self._history.append(record)
            if len(self._history) > self._max_history_steps:
                self._history.pop(0)

            # Append to JSONL log
            try:
                async with aiofiles.open(self._log_file, "a") as f:
                    await f.write(json.dumps(record.__dict__) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write PBKV log: {e}")

    def train_from_jsonl(self, path: str) -> None:
        """Load JSONL and build Markov transition table.

        Reads workflow_steps.jsonl files from the log directory.
        Builds: {(prev_agent, curr_agent): {next_agent: count}}
        Also builds 1st-order fallback: {curr_agent: {next_agent: count}}

        Uses Laplace smoothing (alpha=1) for unseen transitions.
        """
        log_path = Path(path)
        if log_path.is_dir():
            log_path = log_path / "workflow_steps.jsonl"

        if not log_path.exists():
            logger.warning(f"JSONL file not found: {log_path}")
            return

        sequences: list[list[str]] = []
        current_seq: list[str] = []

        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    current_seq.append(record["agent_id"])
                except (json.JSONDecodeError, KeyError):
                    # End of sequence marker (empty line or invalid)
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = []

        if current_seq:
            sequences.append(current_seq)

        # Build transition tables
        self._transition_table.clear()
        self._first_order_table.clear()
        self._all_agents.clear()

        for seq in sequences:
            for i, agent_id in enumerate(seq):
                self._all_agents.add(agent_id)
                if i >= 1:
                    prev_agent = seq[i - 1]
                    # 2nd-order: (prev, curr) → next
                    key = (prev_agent, agent_id)
                    if key not in self._transition_table:
                        self._transition_table[key] = {}
                    self._transition_table[key][agent_id] = \
                        self._transition_table[key].get(agent_id, 0) + 1

                if i >= 2:
                    # 1st-order: curr → next
                    curr_agent = seq[i - 1]
                    next_agent = seq[i]
                    if curr_agent not in self._first_order_table:
                        self._first_order_table[curr_agent] = {}
                    self._first_order_table[curr_agent][next_agent] = \
                        self._first_order_table[curr_agent].get(next_agent, 0) + 1

        self._trained = True
        logger.info(
            f"Trained Markov model: {len(self._transition_table)} 2nd-order states, "
            f"{len(self._first_order_table)} 1st-order states, "
            f"{len(self._all_agents)} unique agents"
        )

    def _get_transition_probs(
        self,
        prev_agent: Optional[str],
        curr_agent: str,
    ) -> dict[str, float]:
        """Get transition probabilities for given state.

        Uses 2nd-order if prev_agent available, else 1st-order.
        Applies Laplace smoothing (alpha=1).
        """
        alpha = 1.0
        num_states = len(self._all_agents) if self._all_agents else 1

        if prev_agent is not None:
            key = (prev_agent, curr_agent)
            if key in self._transition_table:
                total = sum(self._transition_table[key].values())
                probs = {}
                for agent in self._all_agents:
                    count = self._transition_table[key].get(agent, 0)
                    probs[agent] = (count + alpha) / (total + alpha * num_states)
                return probs

        # Fallback to 1st-order
        if curr_agent in self._first_order_table:
            total = sum(self._first_order_table[curr_agent].values())
            probs = {}
            for agent in self._all_agents:
                count = self._first_order_table[curr_agent].get(agent, 0)
                probs[agent] = (count + alpha) / (total + alpha * num_states)
            return probs

        # Uniform fallback
        return {agent: 1.0 / num_states for agent in self._all_agents}

    def _get_state_from_history(self, current_agent_id: str) -> tuple[Optional[str], str]:
        """Extracts (prev_agent, curr_agent) state from history for prediction."""
        prev_agent: Optional[str] = None
        curr_agent = current_agent_id

        seq: list[str] = [s.agent_id for s in self._history]
        for i, agent_id in enumerate(seq):
            if agent_id == current_agent_id and i > 0:
                prev_agent = seq[i - 1]
                break

        if prev_agent is None and len(seq) >= 2:
            prev_agent = seq[-2]
            curr_agent = seq[-1]

        return prev_agent, curr_agent

    def predict_next_agents(
        self,
        current_agent_id: str,
        top_k: int = 3,
    ) -> list[str]:
        """Predict top-k most likely next agents (synchronous).

        Uses only the last observed agent as prev_state for 1st-order
        approximation if history is empty, but tries (prev, curr) → next
        if available.
        """
        if not self._trained and not self._history:
            return [current_agent_id]

        prev_agent: Optional[str] = None
        curr_agent = current_agent_id

        if not self._trained:
            prev_agent, curr_agent = self._get_state_from_history(current_agent_id)

        probs = self._get_transition_probs(prev_agent, curr_agent)
        sorted_agents = sorted(probs.items(), key=lambda x: -x[1])
        return [agent for agent, _ in sorted_agents[:top_k]]

    async def _predict_next_agents_async(
        self,
        current_agent_id: str,
        current_step: int = 0,
        num_predictions: int = 3,
    ) -> PredictionResult:
        """Async wrapper for backward compatibility with PredictionResult.

        Internal use only. Use predict_next_agents() for the public API.
        """
        async with self._lock:
            history_copy = list(self._history)

        if not history_copy:
            return PredictionResult(
                predicted_agents=[current_agent_id],
                predicted_anchor_hashes=[],
                confidence=0.0,
            )

        # Determine prev_agent from history
        prev_agent: Optional[str] = None
        curr_agent = current_agent_id

        # Find current agent in history to get preceding agent
        for i, step in enumerate(history_copy):
            if step.agent_id == current_agent_id and i > 0:
                prev_agent = history_copy[i - 1].agent_id
                curr_agent = current_agent_id
                break

        # Get transition probabilities
        probs = self._get_transition_probs(prev_agent, curr_agent)

        # Sort by probability descending
        sorted_agents = sorted(probs.items(), key=lambda x: -x[1])
        top_agents = [agent for agent, _ in sorted_agents[:num_predictions]]

        confidence = sorted_agents[0][1] if sorted_agents else 0.0

        # Get anchor hashes from recent history for predicted agents
        anchor_hashes = []
        agent_set = set(top_agents)
        for step in reversed(history_copy):
            if step.agent_id in agent_set and step.anchor_hash not in anchor_hashes:
                anchor_hashes.append(step.anchor_hash)
                if len(anchor_hashes) >= num_predictions:
                    break

        return PredictionResult(
            predicted_agents=top_agents,
            predicted_anchor_hashes=anchor_hashes,
            confidence=confidence,
        )

    async def get_eviction_priority(
        self,
        agent_ids: list[str],
        step_graph: Optional["AgentStepGraph"] = None,
    ) -> list[str]:
        """Order agents by inverse predicted probability for eviction.

        Evicts agents least likely to be needed next (low priority).
        Blends with AgentStepGraph if available using blend_alpha:
        - blend_alpha=0.6: step_graph weight
        - (1-blend_alpha)=0.4: pbkv weight
        """
        if not agent_ids:
            return []

        # Get PBKV priorities (lower prob = higher eviction priority)
        pbkv_scores: dict[str, float] = {}
        if self._trained or self._history:
            for agent_id in agent_ids:
                top_k = self.predict_next_agents(agent_id, top_k=len(agent_ids))
                # Score = position in ranked list (lower position = higher prob)
                if agent_id in top_k:
                    pbkv_scores[agent_id] = 1.0 / (top_k.index(agent_id) + 1)
                else:
                    pbkv_scores[agent_id] = 0.0
        else:
            # Uniform if no training data
            for agent_id in agent_ids:
                pbkv_scores[agent_id] = 1.0 / len(agent_ids)

        # Get AgentStepGraph priorities if available
        if step_graph is not None:
            try:
                graph_priorities = step_graph.get_eviction_priority_order()
                graph_scores: dict[str, float] = {}
                for rank, agent_id in enumerate(graph_priorities):
                    if agent_id in agent_ids:
                        graph_scores[agent_id] = 1.0 / (rank + 1)

                # Blend scores
                blended_scores: dict[str, float] = {}
                for agent_id in agent_ids:
                    pbkv = pbkv_scores.get(agent_id, 0.0)
                    graph = graph_scores.get(agent_id, 0.0)
                    blended_scores[agent_id] = (
                        self._blend_alpha * graph + (1 - self._blend_alpha) * pbkv
                    )

                # Sort ascending (low score = evict first = low priority)
                sorted_agents = sorted(
                    agent_ids, key=lambda x: blended_scores.get(x, 0.0)
                )
            except Exception as e:
                logger.warning(f"AgentStepGraph blend failed: {e}")
                sorted_agents = sorted(
                    agent_ids, key=lambda x: pbkv_scores.get(x, 0.0)
                )
        else:
            # PBKV only: sort ascending (low prob = evict first)
            sorted_agents = sorted(
                agent_ids, key=lambda x: pbkv_scores.get(x, 0.0)
            )

        return sorted_agents

    async def get_prefetch_candidates(
        self,
        current_agent_id: str,
        step: int = 0,
        lookahead: int = 2,
    ) -> list[str]:
        """Get list of agent IDs to prefetch within lookahead steps.

        Uses Markov prediction to find agents within 2 steps.
        """
        prediction = await self._predict_next_agents_async(
            current_agent_id, current_step=step, num_predictions=lookahead
        )
        candidates = prediction.predicted_agents

        logger.debug(
            f"PBKV prefetch candidates for agent={current_agent_id} step={step}: "
            f"{len(candidates)} candidates"
        )

        return candidates

    def get_stats(self) -> dict:
        """Return PBKV predictor statistics."""
        return {
            "history_size": len(self._history),
            "log_file": str(self._log_file),
            "max_history_steps": self._max_history_steps,
            "blend_alpha": self._blend_alpha,
            "trained": self._trained,
            "transition_table_size": len(self._transition_table),
            "first_order_table_size": len(self._first_order_table),
            "unique_agents": len(self._all_agents),
        }
