"""Tests for vLLMRomyPlugin.

V6.1+ semantics: the plugin's metadata flags are honest. ``quantized``
(and the new ``quantization_applied`` alias) is True iff a quantizer
was wired AND it actually ran, not just because ``enable_quantization``
is set in the config. Tests cover both the unwired no-op path and the
wired-with-fakes happy path.
"""
import logging

import numpy as np
import pytest

from apohara_context_forge.serving.romy_plugin import (
    ROMYConfig,
    PostAttentionHook,
    PreAttentionHook,
    register,
    vLLMRomyPlugin,
)


# ---------------------------------------------------------------------------
# ROMYConfig                                                                 #
# ---------------------------------------------------------------------------

class TestROMYConfig:
    def test_romy_config_defaults(self):
        config = ROMYConfig()
        assert config.enable_quantization is True
        assert config.enable_anchor_routing is True
        assert config.enable_cla_injection is True
        assert config.enable_jcr_gate is True
        assert config.quantization_mode == "rotate_kv"


# ---------------------------------------------------------------------------
# Fake dependencies                                                          #
# ---------------------------------------------------------------------------

class _FakeQuantizer:
    """Honest fake: records that quantization actually ran."""

    def __init__(self):
        self.calls = 0

    def quantize_pre_rope(self, keys, values, positions):
        self.calls += 1
        return keys, values  # pass-through, but the call IS recorded


class _RaisingQuantizer:
    def quantize_pre_rope(self, keys, values, positions):
        raise RuntimeError("simulated kernel crash")


class _FakeJCRDecision:
    def __init__(self, use_dense: bool, risk_score: float):
        self.use_dense = use_dense
        self.risk_score = risk_score


class _FakeJCRGate:
    def __init__(self, *, fire_on_role: str = "critic"):
        self._fire = fire_on_role
        self.calls: list[str] = []

    def gate_decision(self, agent_role, candidate_count, reuse_rate, layout_shuffled):
        self.calls.append(agent_role)
        if agent_role == self._fire:
            return _FakeJCRDecision(use_dense=True, risk_score=0.95)
        return _FakeJCRDecision(use_dense=False, risk_score=0.10)


class _FakeMetrics:
    def __init__(self):
        self.records: list[bool] = []

    def record_register(self, matched: bool):
        self.records.append(matched)


# ---------------------------------------------------------------------------
# Plugin lifecycle                                                           #
# ---------------------------------------------------------------------------

class TestvLLMRomyPlugin:
    def test_plugin_initialization(self):
        config = ROMYConfig()
        plugin = vLLMRomyPlugin(config)
        assert plugin._config is config
        assert plugin.is_initialized() is False

    def test_initialize_sets_worker_id(self):
        plugin = vLLMRomyPlugin(ROMYConfig())
        plugin.initialize("worker_0", {})
        assert plugin.is_initialized() is True
        stats = plugin.get_stats()
        assert stats["worker_id"] == "worker_0"
        assert stats["initialized"] is True

    def test_dependency_status_reflects_construction(self):
        bare = vLLMRomyPlugin(ROMYConfig())
        assert bare.get_stats()["dependencies"] == {
            "quantizer": False, "lsh_matcher": False,
            "jcr_gate": False, "metrics": False,
        }
        wired = vLLMRomyPlugin(
            ROMYConfig(),
            quantizer=_FakeQuantizer(),
            jcr_gate=_FakeJCRGate(),
            metrics=_FakeMetrics(),
        )
        deps = wired.get_stats()["dependencies"]
        assert deps["quantizer"] is True
        assert deps["jcr_gate"] is True
        assert deps["metrics"] is True
        assert deps["lsh_matcher"] is False

    def test_plugin_pre_attention_hook_property(self):
        plugin = vLLMRomyPlugin(ROMYConfig())
        assert callable(plugin.pre_attention_hook)

    def test_plugin_post_attention_hook_property(self):
        plugin = vLLMRomyPlugin(ROMYConfig())
        assert callable(plugin.post_attention_hook)

    def test_get_stats_returns_config_and_state(self):
        config = ROMYConfig(
            enable_quantization=True,
            enable_anchor_routing=False,
            enable_cla_injection=True,
            quantization_mode="rotate_kv",
        )
        plugin = vLLMRomyPlugin(config)
        plugin.initialize("worker_test", {})
        stats = plugin.get_stats()
        assert stats["config"]["enable_quantization"] is True
        assert stats["config"]["enable_anchor_routing"] is False
        assert stats["config"]["quantization_mode"] == "rotate_kv"


# ---------------------------------------------------------------------------
# Pre-attention hook — HONEST semantics                                      #
# ---------------------------------------------------------------------------

class TestPreAttentionHookHonest:
    def test_unwired_no_op_reports_quantization_not_applied(self):
        """No quantizer wired → quantization_applied=False even with
        enable_quantization=True. This is the truth-up: the old hook
        always returned quantized=True regardless."""
        hook = PreAttentionHook(ROMYConfig(enable_quantization=True))
        result = hook(["b0", "b1"], [101, 2003], layer_idx=0)
        assert result["quantization_attempted"] is False
        assert result["quantization_applied"] is False
        assert result["quantized"] is False  # alias matches honest flag
        assert result["pre_rope"] is True    # INV-10 still holds
        assert result["layer_idx"] == 0
        assert result["num_blocks"] == 2

    def test_quantization_applied_when_quantizer_wired_and_invoked(self):
        """A wired quantizer with valid pre-RoPE tensors → True."""
        q = _FakeQuantizer()
        hook = PreAttentionHook(ROMYConfig(enable_quantization=True), quantizer=q)
        keys = np.zeros((4, 64), dtype=np.float32)
        values = np.zeros((4, 64), dtype=np.float32)
        positions = np.arange(4, dtype=np.float32)
        result = hook(["b0", "b1"], [101, 2003], layer_idx=2,
                      keys=keys, values=values, positions=positions)
        assert result["quantization_attempted"] is True
        assert result["quantization_applied"] is True
        assert result["quantized"] is True
        assert q.calls == 1

    def test_quantization_not_applied_when_tensors_missing(self):
        """Quantizer wired but caller didn't hand us pre-RoPE tensors
        → quantization_applied=False (we refuse to invent inputs)."""
        q = _FakeQuantizer()
        hook = PreAttentionHook(ROMYConfig(), quantizer=q)
        result = hook(["b0"], [101], layer_idx=0)  # no keys/values/positions
        assert result["quantization_attempted"] is True
        assert result["quantization_applied"] is False
        assert q.calls == 0

    def test_quantization_failure_reported_truthfully(self):
        """Quantizer raises → applied=False, hook does not propagate."""
        q = _RaisingQuantizer()
        hook = PreAttentionHook(ROMYConfig(), quantizer=q)
        result = hook(
            ["b0"], [101], layer_idx=0,
            keys=np.zeros((1, 4), dtype=np.float32),
            values=np.zeros((1, 4), dtype=np.float32),
            positions=np.zeros(1, dtype=np.float32),
        )
        assert result["quantization_applied"] is False

    def test_jcr_gate_fires_on_critic_role(self):
        gate = _FakeJCRGate(fire_on_role="critic")
        hook = PreAttentionHook(ROMYConfig(), jcr_gate=gate)
        result = hook(
            ["b0"], [101], layer_idx=0,
            agent_role="critic", candidate_count=5,
            reuse_rate=0.9, layout_shuffled=True,
        )
        assert result["jcr_dense"] is True
        assert result["jcr_risk"] >= 0.7

    def test_jcr_gate_passes_through_for_non_judge(self):
        gate = _FakeJCRGate(fire_on_role="critic")
        hook = PreAttentionHook(ROMYConfig(), jcr_gate=gate)
        result = hook(
            ["b0"], [101], layer_idx=0,
            agent_role="retriever", candidate_count=2,
            reuse_rate=0.1, layout_shuffled=False,
        )
        assert result["jcr_dense"] is False

    def test_jcr_dense_disables_anchor_routing(self):
        """INV-15 path: when JCR fires dense, the LSH lookup is skipped
        — judges that need dense prefill must NOT consult the shared
        registry."""

        class _LSHThatShouldNotBeCalled:
            async def find_reusable_blocks(self, text, exclude_agent=None):
                raise AssertionError("LSH must not be queried when JCR dense fires")

        hook = PreAttentionHook(
            ROMYConfig(),
            lsh_matcher=_LSHThatShouldNotBeCalled(),
            jcr_gate=_FakeJCRGate(fire_on_role="critic"),
        )
        result = hook(
            ["b0"], [101], layer_idx=0,
            agent_role="critic", candidate_count=5,
            reuse_rate=0.9, layout_shuffled=True,
        )
        assert result["jcr_dense"] is True
        assert result["anchor_match"] is None  # routing skipped

    def test_pre_rope_invariant_always_true(self):
        """INV-10: this hook never quantises post-RoPE tensors. The
        contract surfaces in the result dict as pre_rope=True."""
        hook = PreAttentionHook(ROMYConfig())
        assert hook(["b0"], [1], layer_idx=0)["pre_rope"] is True


# ---------------------------------------------------------------------------
# Post-attention hook                                                        #
# ---------------------------------------------------------------------------

class TestPostAttentionHook:
    def test_returns_processed_block_count(self):
        result = PostAttentionHook(ROMYConfig())(["b0", "b1"], [], layer_idx=0)
        assert result["processed_blocks"] == 2
        assert result["layer_idx"] == 0
        assert result["matched"] is False

    def test_matched_flag_propagates_to_metrics(self):
        metrics = _FakeMetrics()
        hook = PostAttentionHook(ROMYConfig(), metrics=metrics)
        hook(["b0"], [], layer_idx=0, matched=True)
        hook(["b1"], [], layer_idx=1, matched=False)
        assert metrics.records == [True, False]

    def test_stats_accumulate_across_calls(self):
        hook = PostAttentionHook(ROMYConfig())
        hook(["b0"], [], layer_idx=0)
        hook(["b1", "b2"], [], layer_idx=1, matched=True)
        result = hook(["b3"], [], layer_idx=2)
        assert result["blocks_processed_total"] == 4

    def test_metrics_record_register_exception_is_caught(self, caplog):
        """A failing metrics sink must not break the post-attention hook."""

        class _BrokenMetrics:
            def record_register(self, matched: bool):
                raise ValueError("boom")

        hook = PostAttentionHook(ROMYConfig(), metrics=_BrokenMetrics())
        with caplog.at_level(logging.WARNING):
            result = hook(["b0"], [], layer_idx=0, matched=True)
        assert result["processed_blocks"] == 1
        assert "ROMY metrics.record_register failed" in caplog.text


# ---------------------------------------------------------------------------
# Entry-point function                                                       #
# ---------------------------------------------------------------------------

class TestEntryPoint:
    def test_register_returns_initialised_plugin(self):
        """register() is what vllm.general_plugins invokes. It must
        return an initialised plugin and must not touch vLLM at all:
        KV interception is config-driven (--kv-transfer-config + LMCache),
        not attention hooks (that platform API never existed)."""
        plugin = register()
        assert isinstance(plugin, vLLMRomyPlugin)
        assert plugin.is_initialized() is True
        assert plugin.get_stats()["worker_id"] == "default"
