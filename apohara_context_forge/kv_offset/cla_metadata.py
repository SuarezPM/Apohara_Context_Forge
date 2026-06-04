"""CLA Metadata Layer — Cross-Layer KV Cache Sharing hints for vLLM.

Based on:
- CLA (NeurIPS 2024): 2x KV cache reduction by sharing KVs between
  adjacent layer groups with negligible accuracy loss.
- NAACL 2025 systematic study: pairing queries of ALL layers with KVs of
  UPPER layers outperforms bottom-layer sharing at aggressive compression.
- LCKV (ACL 2024): Layer-Condensed KV, queries of all layers share KVs of
  only the top layer.

V4.0 CHANGES: New module for inference-time CLA hint injection.
"""
from dataclasses import dataclass
from typing import Optional

# Non-thinking roles (no chain-of-thought, can benefit from CLA)
NON_THOUGHT_ROLES = frozenset({"retriever", "summarizer", "formatter", "reviewer", "classifier"})


@dataclass
class HintRequest:
    """Request payload for emitting CLA hints."""
    agent_id: str
    model_id: str
    is_thinking_mode: bool
    model_layer_count: int = 64
    agent_role: str = "default"


@dataclass
class CLAGroupConfig:
    """Configuration for CLA layer grouping strategy."""
    group_size: int = 2          # layers per group (2 = 2x reduction)
    sharing_direction: str = "upper"  # "upper" | "lower" per NAACL 2025
    thinking_mode_bypass: bool = True  # never apply CLA in thinking mode
    min_layer: int = 0           # skip bottom N layers (attention sinks)
    max_layer: int = 64          # skip above this layer index


@dataclass
class CLAHint:
    """Metadata hint for vLLM attention backend to share KV across layers."""
    agent_id: str
    model_id: str
    layer_groups: list[tuple[int, int]]  # (start_layer, shared_kv_layer)
    estimated_vram_reduction_pct: float  # 0.0–0.5 for group_size=2
    is_thinking_mode: bool       # if True, hint is IGNORED by backend
    group_config: CLAGroupConfig


class CLAMetadataLayer:
    """
    Computes CLA metadata hints for agents based on their role and mode.
    
    Usage:
        cla = CLAMetadataLayer(CLAGroupConfig(group_size=2))
        request = HintRequest("agent1", "Qwen3.6-35B-A22B", is_thinking_mode=False, agent_role="retriever")
        hint = cla.emit_hint(request)
    """
    
    def __init__(self, config: CLAGroupConfig = CLAGroupConfig()):
        self._config = config
    
    def compute_layer_groups(
        self,
        model_layer_count: int,
        agent_role: str,
    ) -> list[tuple[int, int]]:
        """
        Compute layer sharing groups per NAACL 2025 'upper-layer' strategy.
        
        For group_size=2 and 64 layers:
            [(0,1), (2,3), (4,5), ..., (62,63)]
            → layer 0 queries use KV of layer 1, etc.
        Skip min_layer bottom layers to protect attention sinks.
        
        Args:
            model_layer_count: Total number of transformer layers in model
            agent_role: Agent role (e.g., "retriever", "summarizer") determines
                       whether this agent is in thinking or non-thinking mode
        
        Returns:
            List of (start_layer, shared_kv_layer) tuples
        """
        # Check if role is thinking or non-thinking
        is_non_thinking = agent_role in NON_THOUGHT_ROLES
        
        # Don't compute groups for thinking-mode agents (they bypass CLA)
        if not is_non_thinking:
            return []
        
        groups = []
        cfg = self._config
        # Start from min_layer, go up to max_layer, step by group_size
        for start in range(cfg.min_layer, min(cfg.max_layer, model_layer_count), cfg.group_size):
            end = min(start + cfg.group_size - 1, model_layer_count - 1)
            if cfg.sharing_direction == "upper":
                # NAACL 2025: queries of layer i share KV of layer i+1 (upper layer)
                shared_kv_layer = end
            else:
                # Alternative: share KV of lower layer
                shared_kv_layer = start
            groups.append((start, shared_kv_layer))
        
        return groups
    
    def emit_hint(self, request: HintRequest) -> CLAHint:
        """
        Emit a CLAHint for a given agent.
        
        If request.is_thinking_mode=True and thinking_mode_bypass is True,
        returns empty layer_groups and 0.0 vram_reduction.
        
        Args:
            request: The HintRequest encapsulating the agent and model details
        
        Returns:
            CLAHint with layer_groups and estimated VRAM reduction
        """
        # Bypass if thinking mode and config says to bypass
        if request.is_thinking_mode and self._config.thinking_mode_bypass:
            return CLAHint(
                agent_id=request.agent_id,
                model_id=request.model_id,
                layer_groups=[],
                estimated_vram_reduction_pct=0.0,
                is_thinking_mode=True,
                group_config=self._config,
            )
        
        layer_groups = self.compute_layer_groups(request.model_layer_count, request.agent_role)
        vram_reduction = self.estimated_vram_reduction(layer_groups)
        
        return CLAHint(
            agent_id=request.agent_id,
            model_id=request.model_id,
            layer_groups=layer_groups,
            estimated_vram_reduction_pct=vram_reduction,
            is_thinking_mode=request.is_thinking_mode,
            group_config=self._config,
        )
    
    def estimated_vram_reduction(self, layer_groups: list) -> float:
        """
        Estimate VRAM reduction factor from layer groups.
        
        group_size=2 → 50% of layers share KV → ~0.5 * KV_per_layer savings.
        Conservative estimate since actual savings depend on attention head count.
        
        Args:
            layer_groups: Output of compute_layer_groups()
        
        Returns:
            Float 0.0–0.5 representing VRAM fraction saved
        """
        if not layer_groups:
            return 0.0
        
        # Each group shares 1 layer's KV across group_size layers
        # Fraction saved = (group_size - 1) / group_size
        # For group_size=2: (2-1)/2 = 0.5 (50% savings)
        cfg = self._config
        return (cfg.group_size - 1) / cfg.group_size