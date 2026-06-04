"""Tests for CLAMetadataLayer — TASK-004."""
import pytest
from apohara_context_forge.kv_offset.cla_metadata import CLAMetadataLayer, CLAGroupConfig, CLAHint, NON_THOUGHT_ROLES, HintRequest


class TestCLAMetadataLayer:
    """Tests for CLA metadata layer."""

    def test_non_thought_roles_frozenset(self):
        """NON_THOUGHT_ROLES is a frozenset with expected members."""
        assert isinstance(NON_THOUGHT_ROLES, frozenset)
        assert "retriever" in NON_THOUGHT_ROLES
        assert "summarizer" in NON_THOUGHT_ROLES
        assert "critic" not in NON_THOUGHT_ROLES  # thinking agent

    def test_cla_group_config_defaults(self):
        """CLAGroupConfig has sensible defaults."""
        config = CLAGroupConfig()
        assert config.group_size == 2
        assert config.sharing_direction == "upper"
        assert config.thinking_mode_bypass == True

    @pytest.mark.asyncio
    async def test_compute_layer_groups_upper_direction(self):
        """compute_layer_groups returns upper-layer sharing pairs."""
        config = CLAGroupConfig(group_size=2, sharing_direction="upper", min_layer=0, max_layer=64)
        layer = CLAMetadataLayer(config)
        groups = layer.compute_layer_groups(model_layer_count=32, agent_role="retriever")
        assert len(groups) > 0
        # Each group: (start, shared_kv_layer)
        for start, shared in groups:
            assert shared > start  # upper direction: KV from higher layer

    @pytest.mark.asyncio
    async def test_compute_layer_groups_non_thinking_only(self):
        """compute_layer_groups returns empty for thinking agents."""
        config = CLAGroupConfig(group_size=2, thinking_mode_bypass=True)
        layer = CLAMetadataLayer(config)
        groups = layer.compute_layer_groups(model_layer_count=32, agent_role="retriever")
        assert len(groups) > 0  # retriever is non-thinking
        groups_thinking = layer.compute_layer_groups(model_layer_count=32, agent_role="critic")
        assert len(groups_thinking) == 0  # critic is thinking

    def test_emit_hint_returns_cla_hint(self):
        """emit_hint returns CLAHint with correct fields."""
        config = CLAGroupConfig(group_size=2)
        layer = CLAMetadataLayer(config)
        request = HintRequest(
            agent_id="agent1",
            model_id="Qwen3.6-35B-A22B",
            is_thinking_mode=False,
            model_layer_count=32,
            agent_role="retriever",
        )
        hint = layer.emit_hint(request)
        assert isinstance(hint, CLAHint)
        assert hint.agent_id == "agent1"
        assert hint.model_id == "Qwen3.6-35B-A22B"
        assert hint.is_thinking_mode == False
        assert len(hint.layer_groups) > 0

    def test_emit_hint_thinking_mode_bypass(self):
        """emit_hint returns empty groups for thinking mode when bypass=True."""
        config = CLAGroupConfig(group_size=2, thinking_mode_bypass=True)
        layer = CLAMetadataLayer(config)
        request = HintRequest(
            agent_id="agent1",
            model_id="Qwen3.6-35B-A22B",
            is_thinking_mode=True,
            model_layer_count=32,
            agent_role="critic",
        )
        hint = layer.emit_hint(request)
        assert len(hint.layer_groups) == 0
        assert hint.estimated_vram_reduction_pct == 0.0
        assert hint.is_thinking_mode == True

    def test_estimated_vram_reduction(self):
        """estimated_vram_reduction returns correct fraction for group_size=2."""
        config = CLAGroupConfig(group_size=2)
        layer = CLAMetadataLayer(config)
        groups = [(0, 1), (2, 3), (4, 5)]
        reduction = layer.estimated_vram_reduction(groups)
        assert reduction == 0.5  # (2-1)/2 = 0.5 → 50% VRAM reduction