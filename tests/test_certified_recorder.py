import pytest
pytest.importorskip("z3")
from apohara_context_forge.observability import recorders
from apohara_context_forge.observability.ledger import Ledger


def test_certified_decision_appends_verifiable_ledger_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    recorders._reset_singletons()
    cert = recorders.record_certified_inv15_decision(
        agent_id="critic", anchor_hash="abc", risk_score=0.9,
        gate_action="block", predicted_jcr_delta=0.0,
        candidate_count=5, reuse_rate=0.75, layout_shuffled=False, use_dense=True)
    assert cert["satisfies_inv15"] is True
    led = Ledger(tmp_path / "inv15_ledger.jsonl")
    v = led.verify()
    assert v["valid"] is True and v["entries"] == 1
    # the ledger payload carries the full certificate + decision metadata
    entry = list(led)[0]["payload"]
    assert entry["kind"] == "inv15_certificate" and entry["agent_id"] == "critic"
    assert entry["satisfies_inv15"] is True


def test_reset_singletons_resets_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    recorders._reset_singletons()
    assert recorders._ledger is None


def test_out_of_domain_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    recorders._reset_singletons()
    with pytest.raises(ValueError):
        recorders.record_certified_inv15_decision(
            agent_id="critic", anchor_hash="x", risk_score=0.9,
            gate_action="block", predicted_jcr_delta=0.0,
            candidate_count=5, reuse_rate=1.5,  # out of [0,1]
            layout_shuffled=False, use_dense=True)
    assert not (tmp_path / "inv15_ledger.jsonl").exists()


def test_path_traversal_via_symlink_raises_value_error(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    recorders._reset_singletons()

    # Create a symlink that points outside the audit directory
    target = tmp_path.parent / "sensitive.jsonl"
    target.touch()

    # Symlink the inv15.jsonl inside our tmp_path to point to the external file
    symlink_path = tmp_path / "inv15.jsonl"
    os.symlink(target, symlink_path)

    # Calling _get_audit_log should catch that the resolved path is no longer in tmp_path
    with pytest.raises(ValueError):
        recorders._get_audit_log()

    # Same check for ledger
    symlink_ledger = tmp_path / "inv15_ledger.jsonl"
    os.symlink(target, symlink_ledger)

    with pytest.raises(ValueError):
        recorders._get_ledger()
