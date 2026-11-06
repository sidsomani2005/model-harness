from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def cases_root() -> Path:
    return CASES
