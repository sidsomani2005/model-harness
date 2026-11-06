"""Scorer arithmetic and honesty rules, using hand-made verdicts in a tmp runs dir."""

import yaml

from medproof import benchmark
from medproof.runs import bank_item_id, save_target_run, save_verifier_run
from medproof.schema import Citation, ClaimStatus, ClaimVerdict, Overall, TargetRun, Trap, VerifierRun


def _vr(item_id, overall, variant="haiku", cits=(), traps=(), cost=0.001, latency=1000, case_id="med_01", fake=False):
    cv = [ClaimVerdict(claim_id="c1", claim_text="x", status=ClaimStatus.CONTRADICTED, citations=list(cits))] if cits else []
    return VerifierRun(target_run_id=item_id, case_id=case_id, verifier_variant=variant, overall=overall,
                       claim_verdicts=cv, traps_found=list(traps), estimated_cost_usd=cost, latency_ms=latency,
                       total_model_calls=3, fake_backend=fake)


def test_recall_false_alarm_and_insufficient_rules(tmp_path, cases_root):
    b = lambda a: bank_item_id("med_01", a)
    good_cit = Citation(file="lab_results.txt", line_id="L006", quote="potassium 5.7 mmol/L (H)")
    save_verifier_run(_vr(b("w1"), Overall.ERROR_DETECTED, cits=[good_cit], traps=[Trap.TEMPORAL_CAUSALITY]), tmp_path)
    save_verifier_run(_vr(b("w2"), Overall.INSUFFICIENT_TO_VERIFY), tmp_path)
    save_verifier_run(_vr(b("c1"), Overall.NO_ISSUE_FOUND), tmp_path)
    save_verifier_run(_vr(b("c2"), Overall.ERROR_DETECTED, cits=[Citation(file="lab_results.txt", line_id="L099", quote="x")]), tmp_path)
    m = benchmark.evaluate(cases_root, tmp_path)
    g = m["tables"]["bank"]["variants"]["haiku"]
    assert g["error_recall"]["k"] == 1 and g["error_recall"]["n"] == 2  # INSUFFICIENT is not a detection
    assert g["abstention"]["insufficient_on_wrong"] == 1
    assert g["false_alarm_rate"]["k"] == 1 and g["false_alarm_rate"]["n"] == 2
    assert g["trap_attribution"]["matched"] == 1 and g["trap_attribution"]["checked"] == 1
    assert g["evidence_validity"]["citations"] == 2 and g["evidence_validity"]["valid"] == 1
    assert g["missing_verdicts"] > 0  # other cases' bank items have no verdict
    assert m["tables"]["target"]["variants"]["haiku"]["n_scored"] == 0


def _target(run_id, case_id="med_02", stop="end_turn"):
    return TargetRun(run_id=run_id, case_id=case_id, case_version_sha256="x", model="claude-sonnet-5-5",
                     timestamp_utc="2026-11-06T18:00:00Z", exact_prompt="p", answer_raw="a", stop_reason=stop)


def test_target_labels_unlabeled_and_refusal(tmp_path, cases_root):
    save_target_run(_target("r1"), tmp_path)
    save_target_run(_target("r2"), tmp_path)
    save_target_run(_target("r3", stop="refusal"), tmp_path)
    for rid in ("r1", "r2"):
        save_verifier_run(_vr(rid, Overall.ERROR_DETECTED, case_id="med_02"), tmp_path)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "target_labels.yaml").write_text(yaml.safe_dump(
        {"labels": {"r1": {"label": "wrong", "trap": "SILENT_CONTRADICTION", "labeled_by": ["a", "b"]}}}))
    m = benchmark.evaluate(cases_root, tmp_path)
    g = m["tables"]["target"]["variants"]["haiku"]
    assert g["error_recall"]["k"] == 1 and g["error_recall"]["n"] == 1
    assert g["unlabeled"] == 1 and g["target_refusals"] == 1


def test_label_template_never_fills_labels(tmp_path, cases_root):
    save_target_run(_target("r9"), tmp_path)
    path = benchmark.write_label_template(cases_root, tmp_path)
    data = yaml.safe_load(path.read_text())
    assert data["labels"]["r9"]["label"] is None


def test_citation_validity_rules():
    lines = {"a.txt": {"A001": "SYNTHETIC CASE — NOT A REAL PATIENT", "A002": "Allergies: NKDA."}}
    ok = lambda f, l, q: benchmark.citation_is_valid(Citation(file=f, line_id=l, quote=q), lines)[0]
    assert ok("a.txt", "A002", "allergies:  nkda")
    assert ok("a.txt", "A002", "[A002] Allergies: NKDA.")
    assert ok("a.txt", "A001", "SYNTHETIC CASE - NOT ... PATIENT")
    assert not ok("a.txt", "A002", "penicillin")
    assert not ok("a.txt", "A003", "NKDA")
    assert not ok("b.txt", "A002", "NKDA")
    assert not ok("a.txt", "A002", "")


def test_markdown_renders_without_verdicts(tmp_path, cases_root):
    md = benchmark.to_markdown(benchmark.evaluate(cases_root, tmp_path))
    assert "No verdicts saved yet." in md


def test_fake_backend_and_stub_verdicts_are_not_results(tmp_path, cases_root):
    b = lambda a: bank_item_id("med_01", a)
    save_verifier_run(_vr(b("w1"), Overall.ERROR_DETECTED, fake=True), tmp_path)
    stub = VerifierRun(target_run_id=b("w2"), case_id="med_01", verifier_variant="haiku",
                       overall=Overall.INSUFFICIENT_TO_VERIFY, explanation="STUB verifier: not a real verdict.")
    save_verifier_run(stub, tmp_path)
    g = benchmark.evaluate(cases_root, tmp_path)["tables"]["bank"]["variants"]["haiku"]
    assert g["fake_backend_excluded"] == 1 and g["stub_verdicts"] == 1
    assert g["error_recall"]["k"] == 0 and g["error_recall"]["n"] == 1


def test_omission_citations_and_claim_traps_count(tmp_path, cases_root):
    from medproof.schema import OmissionFinding
    vr = _vr(bank_item_id("med_02", "w2"), Overall.ERROR_DETECTED, case_id="med_02")
    vr.omissions = [OmissionFinding(description="NKDA conflicts", kind="contradiction", material=True, confirmed=True,
                                    citations=[Citation(file="ed_triage_note.txt", line_id="N006", quote="Allergies: NKDA.")])]
    vr.claim_verdicts = [ClaimVerdict(claim_id="c1", claim_text="x", status=ClaimStatus.CONFLICTED,
                                      traps=[Trap.SILENT_CONTRADICTION])]
    save_verifier_run(vr, tmp_path)
    g = benchmark.evaluate(cases_root, tmp_path)["tables"]["bank"]["variants"]["haiku"]
    assert g["evidence_validity"]["valid"] == 1 and g["evidence_validity"]["citations"] == 1
    assert g["trap_attribution"]["matched"] == 1
