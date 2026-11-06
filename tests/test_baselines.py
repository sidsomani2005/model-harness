"""Baseline plumbing with a fake LLM (no network). Checks inputs, outputs and honesty."""

import json

import pytest

from medproof import baselines, llm
from medproof.load_case import load_public
from medproof.schema import Overall


@pytest.fixture
def case(fixture_case):
    return fixture_case


def _fake(parsed, stop_reason="end_turn", seen=None):
    async def fake(model, system, user, *, effort=None, schema=None, max_tokens=1024, **kw):
        if seen is not None:
            seen.update(model=model, system=system, user=user, effort=effort, schema=schema)
        return llm.LLMResult(model=model, text=parsed.model_dump_json() if parsed else "", parsed=parsed,
                             stop_reason=stop_reason, input_tokens=100, output_tokens=20,
                             latency_ms=50, cost_usd=0.00002)
    return fake


def test_prompt_has_case_and_answer_but_no_key(case, cases_root):
    user = baselines.build_user_prompt(case, "The patient has no known drug allergies [P007].")
    assert "[A003]" in user and "NKDA" in user and "[P007]" in user
    assert case.question in user
    assert "required_propositions" not in user and "unacceptable" not in user


def test_haiku_baseline_converts_judgment(tmp_path, monkeypatch, case):
    seen = {}
    judgment = baselines.BaselineJudgment(
        overall="ERROR_DETECTED",
        findings=[baselines.BaselineFinding(
            claim_text="No known drug allergies", status="CONTRADICTED", severity="material",
            citations=[baselines.BaselineCitation(file="allergies.txt", line_id="A003", quote="Penicillin - anaphylaxis")],
            why="Allergy list documents penicillin anaphylaxis.")],
        error_types=["SILENT_CONTRADICTION"],
        explanation="Answer erases documented allergy.",
    )
    monkeypatch.setattr(llm, "call", _fake(judgment, seen=seen))
    run = baselines.run_baseline(case, "NKDA [P007]", "direct-haiku", "bank___fixture__w1", runs_root=tmp_path)
    assert seen["model"] == "claude-haiku-5-5" and seen["effort"] == "low"
    assert seen["schema"] is baselines.BaselineJudgment
    assert run.overall == Overall.ERROR_DETECTED and run.verifier_variant == "direct-haiku"
    assert run.claim_verdicts[0].citations[0].line_id == "A003"
    assert run.claim_verdicts[0].severity.value == "material" and run.fake_backend is False
    assert run.total_model_calls == 1 and run.estimated_cost_usd == 0.00002
    raw = json.loads((tmp_path / "verifier" / "bank___fixture__w1" / "direct-haiku" / "raw.json").read_text())
    assert raw["model_calls"][0]["parsed"]["overall"] == "ERROR_DETECTED" and raw["config"]["effort"] == "low"


def test_opus_baseline_uses_default_effort(tmp_path, monkeypatch, case):
    seen = {}
    j = baselines.BaselineJudgment(overall="NO_ISSUE_FOUND", findings=[], error_types=[], explanation="ok")
    monkeypatch.setattr(llm, "call", _fake(j, seen=seen))
    run = baselines.run_baseline(case, "answer", "direct-opus", "t1", runs_root=tmp_path)
    assert seen["model"] == "claude-opus-5-5" and seen["effort"] is None
    assert run.overall == Overall.NO_ISSUE_FOUND


def test_unparsable_or_refused_judgment_is_insufficient_not_guessed(tmp_path, monkeypatch, case):
    monkeypatch.setattr(llm, "call", _fake(None, stop_reason="refusal"))
    run = baselines.run_baseline(case, "answer", "direct-haiku", "t2", runs_root=tmp_path)
    assert run.overall == Overall.INSUFFICIENT_TO_VERIFY
    assert "refusal" in run.needs_review_reasons[0]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        baselines.variant_config("direct-gpt")
