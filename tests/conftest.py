import os
from pathlib import Path

import pytest

from medproof import llm
from medproof.load_case import load_public
from medproof.source_index import SourceIndex

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "cases" / "_fixture"


@pytest.fixture
def fixture_case():
    return load_public(FIXTURE)


@pytest.fixture
def index(fixture_case):
    return SourceIndex(fixture_case)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDPROOF_RUNS_DIR", str(tmp_path / "runs"))
    import medproof.verifier.pipeline as p
    monkeypatch.setattr(p, "RUNS_DIR", tmp_path / "runs")
    yield
    llm.use_fake_backend(None)


def live_enabled() -> bool:
    return os.environ.get("MEDPROOF_LIVE") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


live = pytest.mark.skipif(not live_enabled(), reason="set MEDPROOF_LIVE=1 and ANTHROPIC_API_KEY to run live API tests")
