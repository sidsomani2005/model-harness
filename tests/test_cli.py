"""CLI exposes every spec command and enforces the official-run safety gates."""

import shutil

from medproof import cli, llm
from medproof.runs import load_target_runs, load_verifier_runs


def test_all_spec_commands_exist():
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    assert {"validate-cases", "run-targets", "verify", "baselines", "evaluate", "report", "demo", "freeze"} <= set(sub.choices)


def test_validate_cases_passes(cases_root):
    assert cli.main(["--cases", str(cases_root), "validate-cases"]) == 0


def test_official_run_refused_when_unfrozen(tmp_path, monkeypatch, cases_root):
    async def boom(*a, **k):
        raise AssertionError("must not call the model")
    monkeypatch.setattr(llm, "call", boom)
    rc = cli.main(["--cases", str(cases_root), "run-targets", "--first-only", "--yes", "--runs", str(tmp_path)])
    assert rc == 1 and load_target_runs(tmp_path) == []


def test_official_run_needs_yes_even_when_frozen(tmp_path, monkeypatch, cases_root):
    cases = tmp_path / "cases"
    shutil.copytree(cases_root / "_fixture", cases / "fx")
    (cases / "fx" / "case.yaml").write_text((cases / "fx" / "case.yaml").read_text().replace("id: _fixture", "id: fx"))
    (cases / "fx" / "answer_bank.yaml").write_text((cases / "fx" / "answer_bank.yaml").read_text().replace("case_id: _fixture", "case_id: fx"))
    from medproof import load_case
    monkeypatch.setattr(load_case, "freeze_status", lambda d: ("frozen", []))
    monkeypatch.setattr(cli, "freeze_status", lambda d: ("frozen", []))
    calls = []
    async def fake(model, system, user, **k):
        calls.append(model)
        return llm.LLMResult(model=model, text="answer", parsed=None, stop_reason="end_turn")
    monkeypatch.setattr(llm, "call", fake)
    runs = tmp_path / "runs"
    assert cli.main(["--cases", str(cases), "run-targets", "--runs", str(runs)]) == 2 and calls == []
    assert cli.main(["--cases", str(cases), "run-targets", "--runs", str(runs), "--yes"]) == 0 and calls == ["claude-sonnet-5-5"]
    assert cli.main(["--cases", str(cases), "run-targets", "--runs", str(runs), "--yes"]) == 0 and len(calls) == 1


def test_failed_baseline_saves_nothing(tmp_path, monkeypatch, cases_root):
    async def fail(*a, **k):
        raise RuntimeError("simulated API failure")
    monkeypatch.setattr(llm, "call", fail)
    rc = cli.main(["--cases", str(cases_root), "baselines", "--models", "haiku", "--runs-root", str(tmp_path), "--case", "med_01"])
    assert rc == 1
    assert load_verifier_runs(tmp_path) == []
