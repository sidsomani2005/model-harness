"""End-to-end flow with a scripted fake model. Exercises plumbing, not model quality.
Every result here has fake_backend=True and must never be reported as an experimental number."""
import json
from pathlib import Path

from medproof import llm
from medproof.schema import ClaimStatus, Overall, Trap
from medproof.verifier.pipeline import averify

WRONG = "Spironolactone [M004] caused the initial potassium rise documented in [L004]. The patient has no known drug allergies [P007]."
RIGHT = ("The first elevated potassium (5.9) was on 2026-03-02 [L004], before spironolactone was started on 2026-03-03 [M004], "
         "so it cannot explain the initial onset. Allergy documentation conflicts: the allergy list records penicillin "
         "anaphylaxis [A003] while a later note says NKDA [P007].")


def scripted(answer_is_wrong: bool):
    def handler(req):
        schema = req["schema"]
        user = req["messages"][0]["content"]
        if schema == "ClaimList":
            if answer_is_wrong:
                return {"answer_abstains": False, "answer_reports_conflict": False, "claims": [
                    {"claim_id": "c1", "text": "Spironolactone caused the initial potassium rise.", "kind": "causal_attribution",
                     "certainty": "definitive", "importance": "high", "is_final_conclusion": True,
                     "entities": ["spironolactone", "potassium"], "cited_lines": ["M004", "L004"]},
                    {"claim_id": "c2", "text": "The patient has no known drug allergies.", "kind": "absence",
                     "certainty": "definitive", "importance": "high", "is_final_conclusion": True,
                     "entities": ["drug allergies"], "cited_lines": ["P007"]}]}
            return {"answer_abstains": False, "answer_reports_conflict": True, "claims": [
                {"claim_id": "c1", "text": "The first elevated potassium preceded spironolactone, so it cannot explain the initial onset.",
                 "kind": "temporal", "certainty": "definitive", "importance": "high", "is_final_conclusion": True,
                 "entities": ["potassium", "spironolactone"], "cited_lines": ["L004", "M004"]},
                {"claim_id": "c2", "text": "Allergy documentation conflicts between A003 and P007.", "kind": "conflict_report",
                 "certainty": "definitive", "importance": "high", "is_final_conclusion": True,
                 "entities": ["penicillin"], "cited_lines": ["A003", "P007"]}]}
        if schema == "TemporalExtraction":
            causal = "caused the initial" in user
            return {"applicable": causal, "claim_asserts_causation": causal, "about_patient_events": True, "cause_line_id": "M004" if causal else None,
                    "effect_line_id": "L004" if causal else None, "effect_scope": "initial_onset"}
        if schema == "ClaimCheck":
            if answer_is_wrong and "no known drug allergies" in user:
                return {"status": "CONFLICTED", "reason": "Allergy list documents penicillin anaphylaxis [A003]; note says NKDA [P007].",
                        "uncertainty": "", "suspected_treated_as_confirmed": False, "different_patient_or_encounter": False,
                        "traps": ["SILENT_CONTRADICTION"], "material": True, "confidence": "high",
                        "evidence": [{"line_id": "A003", "quote": "Penicillin - anaphylaxis", "role": "contradicts"},
                                     {"line_id": "P007", "quote": "Allergies: NKDA.", "role": "supports"}]}
            if answer_is_wrong:
                return {"status": "SUPPORTED", "reason": "(naive checker misses the timeline)", "uncertainty": "",
                        "suspected_treated_as_confirmed": False, "different_patient_or_encounter": False,
                        "traps": [], "material": True, "confidence": "medium",
                        "evidence": [{"line_id": "M004", "quote": "spironolactone 25 mg PO", "role": "supports"}]}
            return {"status": "SUPPORTED", "reason": "Lines state it.", "uncertainty": "",
                    "suspected_treated_as_confirmed": False, "different_patient_or_encounter": False,
                    "traps": [], "material": True, "confidence": "high",
                    "evidence": [{"line_id": "L004", "quote": "potassium 5.9 mmol/L", "role": "supports"}]}
        if schema == "SweepResult":
            return {"items": []}
        raise AssertionError(f"unexpected schema {schema}")
    return handler


async def test_wrong_answer_flags_temporal_and_silent_contradiction(fixture_case):
    llm.use_fake_backend(scripted(answer_is_wrong=True))
    run = await averify(fixture_case, WRONG, target_run_id="t_wrong")
    assert run.overall == Overall.ERROR_DETECTED
    assert run.fake_backend
    by_id = {c.claim_id: c for c in run.claim_verdicts}
    assert by_id["c1"].status == ClaimStatus.CONTRADICTED
    assert Trap.TEMPORAL_CAUSALITY in by_id["c1"].traps
    assert {c.line_id for c in by_id["c1"].citations} >= {"M004", "L004"}
    assert by_id["c2"].status == ClaimStatus.CONFLICTED
    assert "confirmation" in by_id["c2"].checks
    assert set(run.traps_found) >= {Trap.TEMPORAL_CAUSALITY, Trap.SILENT_CONTRADICTION}
    raw = json.loads(Path(run.raw_model_outputs_path).read_text())
    assert raw["model_calls"] and raw["stages"]["claims"]


async def test_correct_answer_passes(fixture_case):
    llm.use_fake_backend(scripted(answer_is_wrong=False))
    run = await averify(fixture_case, RIGHT, target_run_id="t_right")
    assert run.overall == Overall.NO_ISSUE_FOUND, run.explanation


async def test_refusal_is_recorded_not_interpreted(fixture_case):
    llm.use_fake_backend(lambda req: (_ for _ in ()).throw(AssertionError("no calls expected")))
    run = await averify(fixture_case, "", target_stop_reason="refusal")
    assert run.overall == Overall.TARGET_REFUSAL and run.total_model_calls == 0


async def test_checker_failure_is_insufficient(fixture_case):
    llm.use_fake_backend(lambda req: "not json")
    run = await averify(fixture_case, WRONG)
    assert run.overall == Overall.INSUFFICIENT_TO_VERIFY


def test_raw_log_follows_runs_root(fixture_case, tmp_path):
    from medproof.verifier.pipeline import verify
    llm.use_fake_backend(scripted(answer_is_wrong=False))
    run = verify(fixture_case, RIGHT, target_run_id="t_root", runs_root=tmp_path / "scratch")
    assert Path(run.raw_model_outputs_path) == tmp_path / "scratch" / "verifier" / "t_root" / "haiku" / "raw.json"
    run2 = verify(fixture_case, RIGHT, target_run_id="t_nosave", save=False)
    assert run2.raw_model_outputs_path is None


async def test_case_id_never_reaches_a_prompt(fixture_case):
    """case ids like med_01_twin must not leak into checker prompts."""
    seen = []
    base = scripted(answer_is_wrong=True)

    def spy(req):
        seen.append(req["system"] + req["messages"][0]["content"])
        return base(req)
    llm.use_fake_backend(spy)
    await averify(fixture_case, WRONG, target_run_id="t_leak")
    assert seen and not any(fixture_case.case_id in s for s in seen)
