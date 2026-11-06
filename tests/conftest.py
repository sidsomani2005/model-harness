import os
from pathlib import Path

import pytest

from medproof import llm
from medproof.load_case import load_public

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "cases" / "_fixture"


@pytest.fixture
def fixture_case():
    return load_public(FIXTURE)


@pytest.fixture(autouse=True)
def _reset_llm_backend():
    yield
    llm.use_fake_backend(None)


def live_enabled() -> bool:
    return os.environ.get("MEDPROOF_LIVE") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


live = pytest.mark.skipif(not live_enabled(), reason="set MEDPROOF_LIVE=1 and ANTHROPIC_API_KEY to run live API tests")


CASES = ROOT / "cases"


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def cases_root() -> Path:
    return CASES
