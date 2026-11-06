"""Every official target prompt uses the identical neutral template, includes the full
record, and leaks nothing from the private key (spec sections 4 and 7)."""

import json

import pytest

from medproof import llm
from medproof.load_case import case_and_twin_dirs, load_private, load_public
from medproof.runs import load_target_runs
from medproof.schema import Trap
from medproof.targets import anthropic_runner as runner

SPEC_TEMPLATE_LINES = [
    "You are reviewing a SYNTHETIC clinical record for documentation accuracy.",
    "Answer the user's question using the supplied source documents only.",
    "If evidence is absent, contradictory or tentative, say so.",
    "Cite relevant file/line IDs. Provide a concise answer and evidence summary.",
    "Do not invent missing details.",
]


def _all_public(cases_root):
    return [(d, load_public(d)) for d in case_and_twin_dirs(cases_root, include_fixture=True)]


def test_template_matches_spec_section_7():
    header = runner.PROMPT_TEMPLATE.split("\n\nSOURCES:\n")[0]
    assert header.splitlines() == SPEC_TEMPLATE_LINES
    assert runner.PROMPT_TEMPLATE.endswith("QUESTION:\n{question}")


def test_same_template_for_every_case_and_twin(cases_root):
    headers = set()
    for _, case in _all_public(cases_root):
        prompt = runner.build_prompt(case)
        headers.add(prompt.split("SOURCES:\n")[0])
        assert prompt.endswith("QUESTION:\n" + case.question)
        assert prompt == runner.build_prompt(case), "prompt must be deterministic"
    assert len(headers) == 1


def test_prompt_contains_every_source_line_verbatim(cases_root):
    for _, case in _all_public(cases_root):
        prompt = runner.build_prompt(case)
        for name, text in case.sources.items():
            assert f"=== FILE: {name} ===" in prompt
            for line in text.splitlines():
                assert line in prompt


def test_prompt_leaks_no_private_information(cases_root):
    for d, case in _all_public(cases_root):
        prompt = " ".join(runner.build_prompt(case).split())
        priv = load_private(d)
        secrets = []
        for field in ("required_propositions", "unacceptable_claims", "permissible_uncertainty", "alternative_correct_answers"):
            secrets += [str(s) for s in priv.key.get(field) or []]
        secrets += [str(priv.key.get("rationale", ""))] + [a.text for a in priv.answer_bank] + [a.rationale for a in priv.answer_bank]
        for s in secrets:
            s = " ".join(s.split())
            if len(s) > 20:
                assert s not in prompt, f"{case.case_id}: leaked {s[:50]!r}"
        for trap in Trap:
            assert trap.value not in prompt
        for word in ("answer key", "trap", "rubric", "ground truth", "verifier"):
            assert word not in prompt.lower(), f"{case.case_id}: prompt mentions {word!r}"


def test_twin_prompt_differs_only_inside_sources(cases_root):
    by_id = {c.case_id: c for _, c in _all_public(cases_root)}
    for cid, case in by_id.items():
        twin = by_id.get(f"{cid}_twin")
        if twin is None:
            continue
        a, b = runner.build_prompt(case), runner.build_prompt(twin)
        assert a.split("SOURCES:\n")[0] == b.split("SOURCES:\n")[0]
        assert a.split("QUESTION:\n")[1] == b.split("QUESTION:\n")[1]


def test_request_config_is_model_default():
    cfg = runner.request_config("claude-sonnet-5-5")
    assert cfg["system"] is None
    assert "default" in cfg["effort"] and "default" in cfg["thinking"]
    assert cfg["fallbacks"].startswith("none")


# ----------------------------------------------------------------- runner behavior (fake LLM, tmp dir)


@pytest.fixture
def fixture_case(cases_root):
    return load_public(cases_root / "_fixture")


def test_dry_run_never_calls_model(tmp_path, monkeypatch, fixture_case):
    async def boom(*a, **k):
        raise AssertionError("dry run must not call the model")

    monkeypatch.setattr(llm, "call", boom)
    out = runner.run_targets([(fixture_case, False)], runs_root=tmp_path, dry_run=True)
    assert out[0].status == "dry_run"
    assert (tmp_path / "dry_run" / "_fixture.prompt.txt").read_text().rstrip("\n") == runner.build_prompt(fixture_case)
    assert not (tmp_path / "target").exists()


def _fake(stop_reason="end_turn", text="An answer [A005].", calls=None):
    async def fake(model, system, user, *, effort=None, schema=None, max_tokens=1024):
        if calls is not None:
            calls.append(dict(model=model, system=system, user=user, effort=effort, schema=schema, max_tokens=max_tokens))
        return llm.LLMResult(model=model, response_model=model, text=text, input_tokens=10,
                             output_tokens=5, stop_reason=stop_reason, latency_ms=7, cost_usd=0.0)
    return fake


def test_first_answer_saved_untouched_and_never_rerun(tmp_path, monkeypatch, fixture_case):
    calls = []
    monkeypatch.setattr(llm, "call", _fake(calls=calls))
    out = runner.run_targets([(fixture_case, False)], runs_root=tmp_path)
    assert out[0].status == "saved"
    assert calls == [dict(model="claude-sonnet-5-5", system=None, user=runner.build_prompt(fixture_case),
                          effort=None, schema=None, max_tokens=runner.TARGET_MAX_TOKENS)]
    [run] = load_target_runs(tmp_path)
    assert run.answer_raw == "An answer [A005]."
    assert run.exact_prompt == runner.build_prompt(fixture_case)
    assert run.case_version_sha256 == fixture_case.source_sha256
    assert run.first_attempt and run.stop_reason == "end_turn"
    again = runner.run_targets([(fixture_case, False)], runs_root=tmp_path)
    assert again[0].status == "skipped_existing" and len(calls) == 1
    assert len(load_target_runs(tmp_path)) == 1


def test_refusal_recorded_separately(tmp_path, monkeypatch, fixture_case):
    monkeypatch.setattr(llm, "call", _fake(stop_reason="refusal", text=""))
    out = runner.run_targets([(fixture_case, False)], runs_root=tmp_path)
    assert out[0].status == "refusal"
    [run] = load_target_runs(tmp_path)
    assert run.stop_reason == "refusal" and run.answer_raw == ""


def test_transport_error_saves_nothing(tmp_path, monkeypatch, fixture_case):
    async def fail(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(llm, "call", fail)
    out = runner.run_targets([(fixture_case, False)], runs_root=tmp_path)
    assert out[0].status == "error" and "network down" in out[0].detail
    assert load_target_runs(tmp_path) == []
