import hashlib
import os
from pathlib import Path

import pytest

from medproof import llm
from medproof.schema import PublicCase

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"

# Verifier unit-test chart (Person 1). Kept outside cases/ so it is never part of the
# benchmark; built directly into a PublicCase so it does not depend on case.yaml rules.
P1_CHART = Path(__file__).resolve().parent / "data" / "p1_chart"
P1_QUESTION = ("Using only the supplied record, summarize what the chart establishes about the patient's drug "
               "allergies, and state whether spironolactone can explain the initial onset of the elevated potassium.")


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def cases_root() -> Path:
    return CASES


@pytest.fixture
def fixture_case() -> PublicCase:
    sources = {p.name: p.read_text() for p in sorted(P1_CHART.glob("*.txt"))}
    sha = hashlib.sha256(repr(sorted(sources.items())).encode()).hexdigest()
    return PublicCase(case_id="_p1_chart", question=P1_QUESTION, sources=sources, source_sha256=sha)


@pytest.fixture
def index(fixture_case):
    from medproof.source_index import SourceIndex
    return SourceIndex(fixture_case)


@pytest.fixture(autouse=True)
def _isolate_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDPROOF_RUNS_DIR", str(tmp_path / "runs"))
    import medproof.verifier.pipeline as p
    monkeypatch.setattr(p, "RUNS_DIR", tmp_path / "runs")
    yield
    llm.use_fake_backend(None)


def live_enabled() -> bool:
    return os.environ.get("MEDPROOF_LIVE") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


live = pytest.mark.skipif(not live_enabled(), reason="set MEDPROOF_LIVE=1 and ANTHROPIC_API_KEY to run live API tests")
