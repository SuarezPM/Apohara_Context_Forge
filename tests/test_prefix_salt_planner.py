"""Tests for PrefixSaltPlanner (Automatic Prefix Caching salt mapping).

Covers:
- Same anchor + cla_group => same shared salt (agents share KV blocks)
- INV-15 dense judge => UNIQUE isolated salt, distinct from the shared group
- The mapping runs against the REAL JCRSafetyGate (not a fake)
- Determinism and purity (no GPU, no I/O)
"""
from __future__ import annotations

from apohara_context_forge.safety.jcr_gate import JCRSafetyGate
from apohara_context_forge.serving.prefix_salt_planner import (
    PrefixSaltPlanner,
    SaltPlan,
)


class TestSharedSalt:
    def test_same_anchor_same_salt(self):
        """Two non-judge agents on the same anchor get the SAME salt."""
        planner = PrefixSaltPlanner()
        a = planner.plan(
            agent_role="retriever",
            anchor_hash=12345,
            cla_group="g1",
            request_id="req-a",
        )
        b = planner.plan(
            agent_role="summarizer",
            anchor_hash=12345,
            cla_group="g1",
            request_id="req-b",
        )
        assert a.shared is True
        assert b.shared is True
        # Identical salt despite different role and different request_id.
        assert a.cache_salt == b.cache_salt

    def test_shared_salt_independent_of_request_id(self):
        planner = PrefixSaltPlanner()
        s1 = planner.shared_salt(anchor_hash=999, cla_group="gX")
        s2 = planner.shared_salt(anchor_hash=999, cla_group="gX")
        assert s1 == s2

    def test_different_anchor_different_salt(self):
        planner = PrefixSaltPlanner()
        a = planner.shared_salt(anchor_hash=1, cla_group="g1")
        b = planner.shared_salt(anchor_hash=2, cla_group="g1")
        assert a != b

    def test_different_cla_group_different_salt(self):
        planner = PrefixSaltPlanner()
        a = planner.shared_salt(anchor_hash=1, cla_group="g1")
        b = planner.shared_salt(anchor_hash=1, cla_group="g2")
        assert a != b


class TestINV15IsolatedSalt:
    """A judge that trips INV-15 must get a UNIQUE salt, isolated from the
    shared group AND from other judge requests."""

    def test_dense_judge_unique_salt_distinct_from_shared(self):
        planner = PrefixSaltPlanner()
        # Shared group: a retriever on the anchor.
        shared = planner.plan(
            agent_role="retriever",
            anchor_hash=42,
            cla_group="g1",
            request_id="req-shared",
        )
        # A critic with risk pushed above threshold (5 candidates, shuffled,
        # high reuse) => JCR gate fires INV-15 => use_dense=True.
        judge = planner.plan(
            agent_role="critic",
            anchor_hash=42,
            cla_group="g1",
            request_id="req-judge",
            candidate_count=5,
            reuse_rate=0.95,
            layout_shuffled=True,
        )
        assert shared.shared is True
        assert judge.shared is False
        # Physical isolation: the judge's salt is NOT the shared group's salt,
        # even though it sits on the SAME anchor and cla_group.
        assert judge.cache_salt != shared.cache_salt

    def test_two_dense_judges_get_distinct_salts(self):
        """Distinct judge requests must not collide with each other."""
        planner = PrefixSaltPlanner()
        j1 = planner.plan(
            agent_role="critic",
            anchor_hash=7,
            cla_group="g1",
            request_id="req-1",
            candidate_count=5,
            reuse_rate=0.95,
            layout_shuffled=True,
        )
        j2 = planner.plan(
            agent_role="critic",
            anchor_hash=7,
            cla_group="g1",
            request_id="req-2",
            candidate_count=5,
            reuse_rate=0.95,
            layout_shuffled=True,
        )
        assert j1.shared is False and j2.shared is False
        assert j1.cache_salt != j2.cache_salt

    def test_low_risk_critic_still_shares(self):
        """A critic UNDER the threshold is not dense => still shares blocks."""
        planner = PrefixSaltPlanner()
        # critic, 2 candidates, no shuffle, low reuse => risk 0.6 <= 0.7.
        p = planner.plan(
            agent_role="critic",
            anchor_hash=5,
            cla_group="g1",
            request_id="req-x",
            candidate_count=2,
            reuse_rate=0.5,
            layout_shuffled=False,
        )
        assert p.shared is True
        expected = planner.shared_salt(anchor_hash=5, cla_group="g1")
        assert p.cache_salt == expected


class TestUsesRealJCRGate:
    """The INV-15 mapping must run against the REAL JCRSafetyGate."""

    def test_planner_default_gate_is_real(self):
        planner = PrefixSaltPlanner()
        assert isinstance(planner.gate, JCRSafetyGate)

    def test_gate_decision_logged_on_real_gate(self):
        """plan() drives gate_decision(), so the real gate's audit log grows."""
        gate = JCRSafetyGate()
        planner = PrefixSaltPlanner(jcr_gate=gate)
        assert len(gate.gate_log) == 0
        planner.plan(
            agent_role="critic",
            anchor_hash=1,
            cla_group="g1",
            request_id="r",
            candidate_count=5,
            reuse_rate=0.95,
            layout_shuffled=True,
        )
        # The real gate recorded the INV-15 decision.
        assert len(gate.gate_log) == 1
        assert gate.gate_log[0].use_dense is True
        assert "INV-15" in gate.gate_log[0].reason

    def test_custom_threshold_gate_changes_isolation(self):
        """Swapping the real gate's threshold changes the salt decision —
        proof the planner defers to the gate, not a hardcoded rule."""
        # With a very high threshold, even a high-risk critic won't go dense.
        lenient = PrefixSaltPlanner(JCRSafetyGate(jcr_threshold=1.0))
        p = lenient.plan(
            agent_role="critic",
            anchor_hash=1,
            cla_group="g1",
            request_id="r",
            candidate_count=5,
            reuse_rate=0.95,
            layout_shuffled=True,
        )
        # risk clamps to 1.0, threshold is 1.0, gate uses strict > => not dense.
        assert p.shared is True


class TestSaltPlanShape:
    def test_returns_salt_plan(self):
        planner = PrefixSaltPlanner()
        p = planner.plan(
            agent_role="responder",
            anchor_hash=1,
            cla_group="g1",
            request_id="r",
        )
        assert isinstance(p, SaltPlan)
        assert isinstance(p.cache_salt, str)
        assert p.cache_salt
        assert isinstance(p.shared, bool)
        assert isinstance(p.reason, str)

    def test_shared_and_isolated_salts_have_distinct_prefixes(self):
        planner = PrefixSaltPlanner()
        assert planner.shared_salt(1, "g").startswith("shared:")
        assert planner.isolated_salt(1, "r").startswith("iso:")


# ---------------------------------------------------------------------------
# US-007 / Phase 5 — AUDIT #19 regression at the planner level               #
# ---------------------------------------------------------------------------

class TestPlannerJudgeIsolationRegression:
    """Regression test on the salt planner itself (the source of the
    `cache_salt` ROMY hands vLLM). Pins the AUDIT #19 baseline:

    * 100 distinct judge requests → 100 unique isolated salts.
    * 10 calls to the shared path with the same inputs → 10 identical
      salts (the determinism that gives vLLM APC the 84.7 % shared
      hit rate).

    Reference: AUDIT.md §19. Post-ABANDON reframe (US-007 / Phase 5)
    keeps the planner's contract unchanged; this test guards against
    drift in the planner's deterministic / unique paths.
    """

    def test_prefix_salt_planner_judge_isolation_unique(self):
        """100 isolated salts with distinct request_ids are all unique.

        This is the planner-level contract that backs the
        `tests/test_romy_plugin.py::test_romy_judge_isolation_zero_hit_rate_regression_on_audit_19`
        test: every judge request gets a fresh salt → vLLM APC
        allocates fresh KV blocks → 0.0 % hit rate between judges.
        """
        planner = PrefixSaltPlanner()
        n = 100
        salts = [
            planner.isolated_salt(anchor_hash="x", request_id=f"req_{i}")
            for i in range(n)
        ]
        assert len(salts) == n
        assert len(set(salts)) == n, (
            f"isolated salts must be unique per request_id: "
            f"got {len(set(salts))}/{n} unique "
            f"(AUDIT #19 baseline requires 0.0 % hit rate between judges)"
        )

    def test_prefix_salt_planner_shared_path_deterministic(self):
        """10 calls to shared_salt with the same (anchor_hash,
        cla_group) inputs all return the same salt.

        This is the determinism precondition for the AUDIT #19
        84.7 % shared APC hit rate: if shared_salt were not
        deterministic, vLLM would not key the same anchor to the
        same block-hash and the 84.7 % baseline would break.
        """
        planner = PrefixSaltPlanner()
        n = 10
        salts = [
            planner.shared_salt(anchor_hash="x", cla_group="default")
            for _ in range(n)
        ]
        assert len(salts) == n
        assert len(set(salts)) == 1, (
            f"shared salts must be deterministic: expected 1 unique "
            f"shared salt across {n} calls, got {len(set(salts))}"
        )
        # And: the shared path must NOT include a request_id signal
        # (otherwise it would behave like the isolated path).
        first = salts[0]
        for s in salts:
            assert s == first
